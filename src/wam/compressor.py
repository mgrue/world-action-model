"""Frozen frame compressor: a small convolutional VAE, 64×64 Frame ↔ 8×8×16 Latent (Step 4 SPEC).

    caffeinate -i uv run python -m wam.compressor train [--name NAME] [--steps 12000] [--seed 0] [--data DIR] [--cpu]
    uv run python -m wam.compressor train   --resume runs/vae/<run>
    uv run python -m wam.compressor eval    <run-dir | checkpoint-name> [--eval-frames 4096] [--json OUT]
    uv run --group viz python -m wam.compressor show <run-dir | checkpoint-name> [--out contact.png]
    uv run python -m wam.compressor promote runs/vae/<run> --name vae-v1 [--force]

Later Steps use `load(name)`, `Compressor.encode` / `decode` / `encode_windows`.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from toycar import State
from toycar.dataset import PROBES, open_dataset

from . import carcheck as cc

# -- constants (SPEC §3, §5) ----------------------------------------------------------------
ARCH: dict = {"channels": [32, 64, 128, 128], "blocks": 2, "z": 16}
Z = 16
GRID = 8
LATENT_SHAPE = (Z, GRID, GRID)
LOGVAR_RANGE = (-30.0, 20.0)
COLOUR_PROBES = [n for n, p in PROBES.items() if p.track_params is None]
NOISE_STD = 0.1  # noisy-latent check, normalised units
NOISE_SEED = 7
WARMUP = 200
ENCODE_CHUNK = 1024

REPO = Path(__file__).resolve().parents[2]
CODE_FILE = Path(__file__).resolve()
DEFAULT_DATA = "data/step3-v1"
RUNS_DIR = Path("runs/vae")
CHECKPOINTS = Path("checkpoints")

DEFAULT_CONFIG: dict = {
    "arch": ARCH,
    "lam": 30.0,  # car-region weight λ
    "kl_weight": 1e-3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup": WARMUP,
    "lr_final": 0.05,
    "steps": 12_000,
    "batch": 128,
    "seed": 0,
    "data": DEFAULT_DATA,
    "split": "train",
    "stats_frames": 65_536,
    "eval_frames": 4096,
    "val_frames": 1024,
    "log_every": 100,
    "val_every": 1000,
    "resume_every": 1000,
}

LOG_FIELDS = ["step", "loss", "rec", "kl", "lr", "frames_per_s", "elapsed_s", "val_rec"]


# -- network (SPEC §3.1) --------------------------------------------------------------------
def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c // 4), c)


class Res(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.n1, self.c1 = _gn(c), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = _gn(c), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


def _check_arch(arch: dict) -> tuple[list[int], int, int]:
    try:
        ch = [int(c) for c in arch["channels"]]
        blocks, z = int(arch["blocks"]), int(arch["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bad arch {arch!r}: {exc}") from None
    if len(ch) != 4 or any(c < 4 for c in ch) or blocks < 0:
        raise ValueError(f"bad arch {arch!r}: need 4 channel counts ≥ 4 and blocks ≥ 0")
    if z != Z:
        raise ValueError(f"bad arch {arch!r}: z is fixed at {Z} by the latent contract")
    return ch, blocks, z


class VAE(nn.Module):
    """64×64×3 → 8×8×(Z means + Z log-variances) → 64×64×3."""

    def __init__(self, arch: dict = ARCH) -> None:
        super().__init__()
        (c0, c1, c2, c3), blocks, z = _check_arch(arch)
        self.arch = {"channels": [c0, c1, c2, c3], "blocks": blocks, "z": z}

        def res(c: int) -> list[nn.Module]:
            return [Res(c) for _ in range(blocks)]

        def up(a: int, b: int) -> list[nn.Module]:
            return [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(a, b, 3, padding=1)]

        self.enc = nn.Sequential(
            nn.Conv2d(3, c0, 3, padding=1),
            nn.Conv2d(c0, c1, 3, stride=2, padding=1), *res(c1),  # 32×32
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), *res(c2),  # 16×16
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), *res(c3),  # 8×8
            _gn(c3), nn.SiLU(), nn.Conv2d(c3, 2 * z, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(z, c3, 3, padding=1), *res(c3),  # 8×8
            *up(c3, c2), *res(c2),  # 16×16
            *up(c2, c1), *res(c1),  # 32×32
            *up(c1, c0),  # 64×64
            _gn(c0), nn.SiLU(), nn.Conv2d(c0, 3, 3, padding=1),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = self.enc(x).chunk(2, dim=1)
        return mu, logvar.clamp(*LOGVAR_RANGE)

    def decode(self, z: Tensor) -> Tensor:
        return self.dec(z)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pick_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def to_x(frames: Tensor) -> Tensor:
    """uint8 (B, 64, 64, 3) → float (B, 3, 64, 64) in [−1, 1], on the frames' device."""
    return frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0


def to_u8(x: Tensor) -> Tensor:
    """float (B, 3, 64, 64) → uint8 (B, 64, 64, 3), clamped and rounded."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1)


# -- the frozen compressor (SPEC §7, §8) ----------------------------------------------------
class Compressor:
    """A frozen VAE with its latent normalisation. Encodes to normalised mean Latents."""

    def __init__(self, vae: VAE, latent_mean: Tensor, latent_std: Tensor, *, device=None,
                 name: str | None = None, card: dict | None = None) -> None:
        self.device = pick_device(device)
        self.vae = vae.to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.latent_mean = latent_mean.detach().to(self.device, torch.float32).reshape(Z)
        self.latent_std = latent_std.detach().to(self.device, torch.float32).reshape(Z)
        self.name = name
        self.card = card
        self.latent_shape = LATENT_SHAPE

    @property
    def arch(self) -> dict:
        return self.vae.arch

    def _frames(self, frames) -> Tensor:
        t = torch.from_numpy(np.ascontiguousarray(frames)) if isinstance(frames, np.ndarray) else frames
        if not isinstance(t, Tensor):
            raise TypeError(f"frames must be a numpy array or torch tensor, got {type(frames).__name__}")
        if t.dtype != torch.uint8 or t.shape[-3:] != (64, 64, 3):
            raise ValueError(f"frames must be uint8 (..., 64, 64, 3), got {t.dtype} {tuple(t.shape)}")
        return t

    def encode(self, frames) -> Tensor:
        """uint8 (..., 64, 64, 3) → float32 (..., 16, 8, 8), normalised mean Latents on self.device."""
        t = self._frames(frames)
        lead = t.shape[:-3]
        flat = t.reshape(-1, 64, 64, 3)
        m, s = self.latent_mean.view(1, Z, 1, 1), self.latent_std.view(1, Z, 1, 1)
        with torch.inference_mode():
            outs = []
            for i in range(0, flat.shape[0], ENCODE_CHUNK):
                mu, _ = self.vae.encode(to_x(flat[i : i + ENCODE_CHUNK].to(self.device)))
                outs.append((mu - m) / s)
            z = torch.cat(outs) if outs else torch.empty((0, *LATENT_SHAPE), device=self.device)
        return z.reshape(*lead, *LATENT_SHAPE)

    def _decode_raw(self, z: Tensor) -> Tensor:
        lead = z.shape[:-3]
        flat = z.reshape(-1, *LATENT_SHAPE).to(self.device, torch.float32)
        mu = flat * self.latent_std.view(1, Z, 1, 1) + self.latent_mean.view(1, Z, 1, 1)
        x = torch.cat([self.vae.decode(mu[i : i + ENCODE_CHUNK]) for i in range(0, mu.shape[0], ENCODE_CHUNK)]) \
            if mu.shape[0] else torch.empty((0, 3, 64, 64), device=self.device)
        return x.reshape(*lead, 3, 64, 64)

    def decode(self, z: Tensor, *, as_float: bool = False) -> Tensor:
        """Normalised (..., 16, 8, 8) → uint8 (..., 64, 64, 3); `as_float` → float32 (..., 3, 64, 64) in [−1, 1],
        differentiable in z (the weights stay frozen)."""
        if tuple(z.shape[-3:]) != LATENT_SHAPE:
            raise ValueError(f"latents must be (..., {Z}, {GRID}, {GRID}), got {tuple(z.shape)}")
        if as_float:
            return self._decode_raw(z).clamp(-1, 1)
        with torch.inference_mode():
            x = self._decode_raw(z)
            lead = x.shape[:-3]
            return to_u8(x.reshape(-1, 3, 64, 64)).reshape(*lead, 64, 64, 3)

    def encode_windows(self, batch: dict) -> tuple[Tensor, Tensor]:
        """Step 3 Window batch → obs (B, 16, 8, 8), future (B, H, 16, 8, 8), in one encoder call."""
        obs = self._frames(batch["obs_frame"])
        fut = self._frames(batch["future_frames"])
        B, H = fut.shape[0], fut.shape[1]
        if obs.shape[0] != B:
            raise ValueError(f"obs_frame batch {obs.shape[0]} != future_frames batch {B}")
        z = self.encode(torch.cat([obs[:, None], fut], dim=1))
        return z[:, 0], z[:, 1:].reshape(B, H, *LATENT_SHAPE)

    def reconstruct(self, frames, noise: float = 0.0, generator: torch.Generator | None = None) -> np.ndarray:
        """uint8 Frames → uint8 reconstructions (numpy), optionally with N(0, noise²) added to the Latents."""
        z = self.encode(frames)
        if noise:
            z = z + noise * torch.randn(z.shape, generator=generator).to(self.device)
        return self.decode(z).cpu().numpy()


# -- provenance helpers ---------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha256() -> str:
    return _sha256(CODE_FILE)


def _rel(path: str | Path) -> str:
    """Repo-relative path (never absolute)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, REPO)).as_posix()


def data_path(data: str | Path) -> Path:
    """A dataset path from a config or card (repo-relative) or the command line (as given)."""
    p = Path(data)
    return p if p.is_absolute() or p.exists() else REPO / p


def _git_info() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit.stdout.strip(), bool(status.stdout.strip())


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _provenance(data: str | Path, device: torch.device, stats_frames: int) -> dict:
    commit, dirty = _git_info()
    return {
        "git_commit": commit, "git_dirty": dirty, "code_sha256": code_sha256(),
        "torch_version": torch.__version__, "device": device.type,
        "dataset": _rel(data_path(data)), "manifest_sha256": _sha256(data_path(data) / "manifest.json"),
        "split": "train", "stats_frames": int(stats_frames),
    }


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


# -- training data (SPEC §3.2, §5.1) --------------------------------------------------------
class FrameSampler:
    """Train Frames drawn uniformly with replacement, rendered on the fly, with their car-region masks."""

    def __init__(self, es, seed: int, stream: int) -> None:
        from toycar.dataset import frame_dataset

        self.es = es
        self.items = frame_dataset(es)
        self.rng = np.random.default_rng([seed, stream])
        self._bg: dict[int, np.ndarray] = {}

    def background(self, e: int) -> np.ndarray:
        if e not in self._bg:
            self._bg[e] = cc.background(self.es.track, self.es.render_params(e))
        return self._bg[e]

    def batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.rng.integers(len(self.items), size=n)
        frames = np.empty((n, 64, 64, 3), np.uint8)
        bgs = np.empty_like(frames)
        for j, i in enumerate(idx):
            e, t = self.items.locate(int(i))
            frames[j] = self.es._render(e, t, self.es.render_params(e))
            bgs[j] = self.background(e)
        return frames, cc.car_region(frames, bgs)

    def get_state(self) -> str:
        return json.dumps(self.rng.bit_generator.state)

    def set_state(self, s: str) -> None:
        self.rng.bit_generator.state = json.loads(s)


def rec_loss(xhat: Tensor, x: Tensor, mask: Tensor, lam: float) -> Tensor:
    """Weighted mean L1: weight 1 + λ in the car region, 1 elsewhere (SPEC §3.2)."""
    w = 1.0 + lam * mask.float()[:, None]
    return (w * (xhat - x).abs()).sum() / (w.sum() * 3)


def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean()


def lr_factor(s: int, steps: int, warmup: int = WARMUP, final: float = 0.05) -> float:
    if s < warmup:
        return (s + 1) / warmup
    return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))


class _Tee(io.TextIOBase):
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def _new_run_dir(root: Path, name: str) -> Path:
    base = root / f"{datetime.datetime.now():%Y%m%d-%H%M}-{name}"
    out, k = base, 1
    while out.exists():
        k += 1
        out = base.with_name(f"{base.name}-{k}")
    out.mkdir(parents=True)
    return out


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(config: dict, out: Path, *, resume: bool = False, device=None, _stop_at: int | None = None) -> Path:
    """Run the §5 recipe. `out` is the parent of a new Run directory, or the Run directory with `resume`.
    Returns the Run directory. `_stop_at` (tests) saves `resume.pt` and stops after that step."""
    out = Path(out)
    if resume:
        run = out
        cfg = json.loads((run / "config.json").read_text())
        state = torch.load(run / "resume.pt", map_location="cpu", weights_only=True)
        dev = pick_device(device if device is not None else cfg["provenance"]["device"])
    else:
        cfg = {**DEFAULT_CONFIG, **config}
        dev = pick_device(device)
        cfg["name"] = cfg.get("name") or "vae"
        cfg["data"] = _rel(data_path(cfg["data"]))
        run = _new_run_dir(out, cfg["name"])
        cfg["provenance"] = _provenance(cfg["data"], dev, cfg["stats_frames"])
        cfg["created"] = _now()
        cfg["resumes"] = []
        state = None

    with open(run / "stdout.txt", "a") as so, contextlib.redirect_stdout(_Tee(sys.stdout, so)):
        if resume:
            cfg["resumes"].append({"step": int(state["step"]), "at": _now(), "device": dev.type})
            print(f"resuming {_rel(run)} at step {int(state['step'])} on {dev}", flush=True)
        (run / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        model, info = _train_loop(cfg, run, dev, state, _stop_at)
        if model is None:
            return run
        _finish(cfg, run, dev, model, info)
    return run


def _train_loop(cfg: dict, run: Path, dev: torch.device, state: dict | None, stop_at: int | None):
    steps, batch, seed = int(cfg["steps"]), int(cfg["batch"]), int(cfg["seed"])
    torch.manual_seed(seed)
    model = VAE(cfg["arch"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, steps, int(cfg["warmup"]), float(cfg["lr_final"])))
    ds = open_dataset(data_path(cfg["data"]), check=False)
    sampler = FrameSampler(ds.split(cfg["split"]), seed, 0)
    gen = torch.Generator(device=dev).manual_seed(seed)
    val_es = ds.split("val")
    vframes = cc.eval_frames(val_es, int(cfg["val_frames"]))
    val_x = np.stack([val_es._render(e, t, val_es.render_params(e)) for e, t in vframes]) if vframes else None
    val_m = cc.car_region(val_x, np.stack([cc.background(val_es.track, val_es.render_params(e)) for e, _ in vframes])) \
        if vframes else None

    start, acc, n_acc, elapsed0 = 0, np.zeros(3), 0, 0.0
    log_path = run / "log.csv"
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        sampler.set_state(state["sampler"])
        gen.set_state(state["gen"])
        start, n_acc, elapsed0 = int(state["step"]), int(state["n_acc"]), float(state["elapsed"])
        acc = state["acc"].numpy().astype(np.float64)
        with open(log_path, "r+b") as f:
            f.truncate(int(state["log_bytes"]))
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)

    lam, klw = float(cfg["lam"]), float(cfg["kl_weight"])
    log_every, val_every, resume_every = int(cfg["log_every"]), int(cfg["val_every"]), int(cfg["resume_every"])
    print(f"training {_rel(run)}: {n_params(model):,} params, {steps} steps × batch {batch} on {dev}", flush=True)
    t0 = time.time() - elapsed0
    last = {"loss": math.nan, "rec": math.nan, "kl": math.nan, "val_rec": math.nan}
    model.train()
    for step in range(start + 1, steps + 1):
        frames, mask = sampler.batch(batch)
        x = to_x(torch.from_numpy(frames).to(dev))
        m = torch.from_numpy(mask).to(dev)
        mu, logvar = model.encode(x)
        z = mu + torch.randn(mu.shape, generator=gen, device=dev) * (0.5 * logvar).exp()
        rec = rec_loss(model.decode(z), x, m, lam)
        kl = kl_loss(mu, logvar)
        loss = rec + klw * kl
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}; rerun with --resume from the last resume.pt")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        acc += [loss.item(), rec.item(), kl.item()]
        n_acc += 1

        if step % log_every == 0 or step == steps:
            val = ""
            if val_x is not None and (step % val_every == 0 or step == steps):
                val = _val_rec(model, val_x, val_m, dev, lam)
                last["val_rec"] = val
            el = time.time() - t0
            lr = sched.get_last_lr()[0]
            a = acc / n_acc
            last.update(loss=float(a[0]), rec=float(a[1]), kl=float(a[2]))
            fps = step * batch / max(el, 1e-9)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, *(round(float(v), 6) for v in a), f"{lr:.3e}", round(fps, 1),
                                        round(el, 1), "" if val == "" else round(val, 6)])
            eta = el / step * (steps - step)
            print(f"  step {step:6d}/{steps}  loss {a[0]:.4f}  rec {a[1]:.4f}  kl {a[2]:8.3f}  lr {lr:.1e}  "
                  f"{fps:5.0f} Frames/s  ETA {eta / 60:5.1f} min" + ("" if val == "" else f"  val_rec {val:.4f}"),
                  flush=True)
            acc[:] = 0
            n_acc = 0

        if (step % resume_every == 0 and step < steps) or step == stop_at:
            _save_atomic({
                "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                "sampler": sampler.get_state(), "gen": gen.get_state(), "acc": torch.from_numpy(acc.copy()),
                "n_acc": n_acc, "elapsed": time.time() - t0, "log_bytes": log_path.stat().st_size,
            }, run / "resume.pt")
            if step == stop_at:
                print(f"stopped at step {step}; continue with: train --resume {_rel(run)}", flush=True)
                return None, None
    _sync(dev)
    train_s = time.time() - t0
    model.eval()
    return model, {"train_s": round(train_s, 1), "params": n_params(model),
                   "frames_per_s": round((steps * batch) / max(train_s, 1e-9), 1), "final": last}


@torch.no_grad()
def _val_rec(model: VAE, frames: np.ndarray, mask: np.ndarray, dev: torch.device, lam: float) -> float:
    num = den = 0.0
    for i in range(0, len(frames), 256):
        x = to_x(torch.from_numpy(frames[i : i + 256]).to(dev))
        m = torch.from_numpy(mask[i : i + 256]).to(dev)
        mu, _ = model.encode(x)
        w = 1.0 + lam * m.float()[:, None]
        num += float((w * (model.decode(mu) - x).abs()).sum())
        den += float(w.sum() * 3)
    return num / den


@torch.no_grad()
def latent_stats(model: VAE, es, n: int, seed: int, dev: torch.device) -> tuple[Tensor, Tensor]:
    """Per-channel mean / std of μ over n train Frames from stream [seed, 1] (SPEC §3.3)."""
    sampler = FrameSampler(es, seed, 1)
    s1 = torch.zeros(Z, dtype=torch.float64)
    s2 = torch.zeros(Z, dtype=torch.float64)
    cnt = 0
    for i in range(0, n, 256):
        frames, _ = sampler.batch(min(256, n - i))
        mu, _ = model.encode(to_x(torch.from_numpy(frames).to(dev)))
        mu = mu.cpu().double()
        s1 += mu.sum(dim=(0, 2, 3))
        s2 += mu.pow(2).sum(dim=(0, 2, 3))
        cnt += mu.shape[0] * mu.shape[2] * mu.shape[3]
    mean = s1 / cnt
    std = (s2 / cnt - mean.pow(2)).clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


def _finish(cfg: dict, run: Path, dev: torch.device, model: VAE, info: dict) -> None:
    print(f"trained in {info['train_s'] / 60:.1f} min; latent stats over {cfg['stats_frames']} train Frames …", flush=True)
    ds = open_dataset(data_path(cfg["data"]), check=False)
    mean, std = latent_stats(model, ds.split(cfg["split"]), int(cfg["stats_frames"]), int(cfg["seed"]), dev)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "arch": model.arch,
                "latent_mean": mean, "latent_std": std}, run / "checkpoint.pt")
    comp = Compressor(model, mean, std, device=dev)
    res = evaluate(comp, data_path(cfg["data"]), int(cfg["eval_frames"]))
    metrics = {
        "check": res["summary"],
        "latent_mean": [round(float(v), 5) for v in mean], "latent_std": [round(float(v), 5) for v in std],
        **info,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"latent std per channel: {' '.join(f'{v:.3g}' for v in std)}")
    try:
        cc.contact_sheet(run / "contact.png", f"{run.name}: clean {res['summary']['clean']['all']['verdict']}",
                         res["clean"], res["probes"])
    except ImportError:
        print("contact sheet skipped (matplotlib missing; uv sync --group viz, then `show`)")
    (run / "resume.pt").unlink(missing_ok=True)
    print(f"clean check: {res['summary']['clean']['all']['verdict']}")
    print(_rel(run), flush=True)


# -- evaluation (SPEC §4.4) -----------------------------------------------------------------
def evaluate(comp: Compressor, data: str | Path, n_frames: int = cc.N_FRAMES, *, noisy: bool = True,
             probes: Sequence[str] = COLOUR_PROBES, quiet: bool = False) -> dict:
    """The full check on n val Frames: clean, noisy, per colour Probe. Returns
    {"summary": {"clean", "noisy", "probes"}, "clean": (rows, orig, rec), "probes": {name: (rows, orig, rec)}}."""
    es = open_dataset(data, check=False).split("val")
    frames = cc.eval_frames(es, n_frames)
    track = es.track
    states = [State(*map(float, es.arrays["states"][e, t])) for e, t in frames]
    branch = [int(es.arrays["branch"][e, t]) for e, t in frames]
    tag = comp.name or "run"

    def one(probe: str | None, noise: float, label: str):
        rps = [es.render_params(e, probe) for e, _ in frames]
        orig = np.stack([es._render(e, t, rp) for (e, t), rp in zip(frames, rps)])
        rec = comp.reconstruct(orig, noise, torch.Generator().manual_seed(NOISE_SEED))
        res = cc.check(orig, rec, states, rps, track, branch=branch, label=None if quiet else f"{tag} — {label}")
        return res, orig, rec

    clean, orig, rec = one(None, 0.0, f"clean latents ({len(frames)} val Frames)")
    summary = {"clean": cc.jsonable(clean)}
    if noisy:
        summary["noisy"] = cc.jsonable(one(None, NOISE_STD, f"noisy latents (+{NOISE_STD}·N)")[0])
    summary["probes"] = {}
    sheets = {}
    for p in probes:
        r, po, pr = one(p, 0.0, f"probe {p} (report only)")
        summary["probes"][p] = cc.jsonable(r)
        sheets[p] = (r["rows"], po, pr)
    return {"summary": summary, "clean": (clean["rows"], orig, rec), "probes": sheets}


# -- Checkpoints (SPEC §6) ------------------------------------------------------------------
def _build(arch: dict, state_dict: dict, where: str) -> VAE:
    try:
        vae = VAE(arch)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{where}: arch does not build: {exc}") from None
    try:
        vae.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(f"{where}: state dict does not fit the arch {arch}: {exc}") from None
    return vae


def _read_weights(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def load(name: str | Path, device: str | torch.device | None = None, *, root: Path = CHECKPOINTS) -> Compressor:
    """A frozen Compressor from `checkpoints/<name>/`, or from a Run directory (name = None)."""
    p = Path(name)
    if p.is_dir() and (p / "checkpoint.pt").exists():
        ck = _read_weights(p / "checkpoint.pt")
        vae = _build(ck["arch"], ck["state_dict"], str(p))
        card = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else None
        return Compressor(vae, ck["latent_mean"], ck["latent_std"], device=device, name=None, card=card)
    d = Path(root) / str(name)
    if not (d / "model.pt").exists():
        raise FileNotFoundError(f"no Checkpoint {name!r} in {root} (and not a Run directory)")
    card = json.loads((d / "card.json").read_text())
    ck = _read_weights(d / "model.pt")
    vae = _build(ck["arch"], ck["state_dict"], str(d))
    if card.get("code_sha256") != code_sha256():
        warnings.warn(f"Checkpoint {name!r}: src/wam/compressor.py changed since it was trained "
                      "(code_sha256 differs); the weights still load", stacklevel=2)
    return Compressor(vae, ck["latent_mean"], ck["latent_std"], device=device, name=str(name), card=card)


def promote(run: Path, name: str, *, force: bool = False, root: Path = CHECKPOINTS) -> Path:
    """Copy a finished Run's weights to `root/<name>/` with its provenance card. Never overwrites."""
    run, root = Path(run), Path(root)
    dest = root / name
    if dest.exists():
        raise FileExistsError(f"Checkpoint {name!r} already exists in {root}; pick a new name")
    for f in ("checkpoint.pt", "config.json", "metrics.json"):
        if not (run / f).exists():
            raise FileNotFoundError(f"{run} is not a finished Run (missing {f})")
    cfg = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    verdict = metrics["check"]["clean"]["all"]["verdict"]
    if verdict != "PASS" and not force:
        raise ValueError(f"{run}: clean check is {verdict}, not PASS; pass --force to promote anyway")
    ck = _read_weights(run / "checkpoint.pt")
    _build(ck["arch"], ck["state_dict"], str(run))
    prov = cfg["provenance"]
    final = metrics.get("final", {})
    check = {k: metrics["check"][k]["all"] if k != "probes" else
             {p: v["all"] for p, v in metrics["check"]["probes"].items()} for k in metrics["check"]}
    check["clean_slices"] = {k: v for k, v in metrics["check"]["clean"].items() if k != "all"}
    card = {
        "name": name, "created": _now(), "source_run": _rel(run),
        **{k: prov[k] for k in ("git_commit", "git_dirty", "code_sha256", "torch_version", "device", "dataset",
                                "manifest_sha256", "split", "stats_frames")},
        "config": {k: v for k, v in cfg.items() if k not in ("provenance",)},
        "final": {k: final.get(k) for k in ("loss", "rec", "kl", "val_rec")},
        "params": metrics.get("params"), "train_s": metrics.get("train_s"),
        "check": check, "verdict": verdict, "forced": bool(force and verdict != "PASS"),
        "latent_std": metrics["latent_std"],
    }
    tmp = root / f".{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copyfile(run / "checkpoint.pt", tmp / "model.pt")
    (tmp / "card.json").write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
    tmp.rename(dest)
    return dest


# -- CLI (SPEC §9) --------------------------------------------------------------------------
def _target_data(comp: Compressor, override: str | None) -> str:
    if override:
        return override
    card = comp.card or {}  # a Checkpoint card ("dataset") or a Run's config.json ("data"), both repo-relative
    return str(data_path(card.get("dataset") or card.get("data") or DEFAULT_DATA))


def cmd_train(a) -> int:
    if a.resume:
        run = train({}, a.resume, resume=True, device="cpu" if a.cpu else None)
    else:
        cfg = {"name": a.name, "steps": a.steps, "seed": a.seed, "data": a.data}
        for k in ("eval_frames", "stats_frames", "batch", "val_frames"):
            if getattr(a, k) is not None:
                cfg[k] = getattr(a, k)
        if a.arch:
            cfg["arch"] = json.loads(a.arch)
        run = train(cfg, a.runs, device="cpu" if a.cpu else None)
    if not (run / "metrics.json").exists():
        return 1
    m = json.loads((run / "metrics.json").read_text())
    return 0 if m["check"]["clean"]["all"]["verdict"] == "PASS" else 1


def cmd_eval(a) -> int:
    comp = load(a.target, device="cpu" if a.cpu else None, root=a.root)
    res = evaluate(comp, _target_data(comp, a.data), a.eval_frames)
    verdict = res["summary"]["clean"]["all"]["verdict"]
    if a.json:
        Path(a.json).write_text(json.dumps(res["summary"], indent=2, allow_nan=False) + "\n")
    print(f"\nclean check: {verdict}")
    return 0 if verdict == "PASS" else 1


def cmd_show(a) -> int:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("show needs matplotlib: uv sync --group viz (or uv run --group viz …)", file=sys.stderr)
        return 2
    comp = load(a.target, device="cpu" if a.cpu else None, root=a.root)
    res = evaluate(comp, _target_data(comp, a.data), a.eval_frames, noisy=False, quiet=True)
    is_run = Path(a.target).is_dir()
    out = a.out or (Path(a.target) / "contact.png" if is_run else Path("contact.png"))
    title = f"{comp.name or Path(a.target).name}: clean {res['summary']['clean']['all']['verdict']}"
    cc.contact_sheet(out, title, res["clean"], res["probes"])
    print(out)
    return 0


def cmd_promote(a) -> int:
    try:
        dest = promote(a.run, a.name, force=a.force, root=a.root)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(dest)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wam.compressor", description="Train, check and freeze the frame compressor.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    hide = argparse.SUPPRESS

    tr = sub.add_parser("train", help="train a VAE into a new Run directory (or --resume one)")
    tr.add_argument("--name", default="vae")
    tr.add_argument("--steps", type=int, default=DEFAULT_CONFIG["steps"])
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--data", default=DEFAULT_DATA)
    tr.add_argument("--cpu", action="store_true")
    tr.add_argument("--resume", type=Path, metavar="RUN")
    tr.add_argument("--runs", type=Path, default=RUNS_DIR, help=hide)
    tr.add_argument("--eval-frames", type=int, help=hide)
    tr.add_argument("--stats-frames", type=int, help=hide)
    tr.add_argument("--val-frames", type=int, help=hide)
    tr.add_argument("--batch", type=int, help=hide)
    tr.add_argument("--arch", help=hide)

    for cmd, hlp in (("eval", "print the car-visibility check; exit 0 on clean PASS"),
                     ("show", "write the contact sheet (needs --group viz)")):
        p = sub.add_parser(cmd, help=hlp)
        p.add_argument("target", help="Run directory or Checkpoint name")
        p.add_argument("--eval-frames", type=int, default=cc.N_FRAMES)
        p.add_argument("--data", help="dataset directory (default: the one the compressor was trained on)")
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--root", type=Path, default=CHECKPOINTS, help=hide)
        if cmd == "eval":
            p.add_argument("--json", type=Path, metavar="OUT")
        else:
            p.add_argument("--out", type=Path)

    pr = sub.add_parser("promote", help="freeze a Run as checkpoints/<name>/")
    pr.add_argument("run", type=Path)
    pr.add_argument("--name", required=True)
    pr.add_argument("--force", action="store_true")
    pr.add_argument("--root", type=Path, default=CHECKPOINTS, help=hide)

    a = ap.parse_args(argv)
    return {"train": cmd_train, "eval": cmd_eval, "show": cmd_show, "promote": cmd_promote}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
