"""Training loop for the world action model (SPEC §6, §10.1).

`train(config, out)` creates `out/<YYYYMMDD-HHMM>-<name>/`, trains, evaluates (§9), writes the contact sheet
and returns the Run directory; `train({}, run, resume=True)` continues an unfinished Run bit for bit.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from toycar.dataset import open_dataset

from . import compressor as C
from . import evaluate as E
from . import flow
from .model import ModelConfig, build, n_params
from .runs import (CHECKPOINTS, REPO, _git_info, _new_run_dir, _now, _read_weights, _rel, _save_atomic, _sha256,
                   _sync, _Tee, data_path)

DEFAULT_DATA = "data/step3-v1"
RUNS_DIR = Path("runs/wam")
QUICK_STEPS = 3_200
DEFAULT_RECIPE: dict = {
    "steps": 32_000,
    "batch": 32,
    "lr": 3e-4,
    "betas": [0.9, 0.99],
    "weight_decay": 0.01,  # Linear weights only
    "warmup": 1_000,
    "lr_final": 0.05,
    "grad_clip": 1.0,
    "ema_max": 0.999,
    "seed": 0,
    "data": DEFAULT_DATA,
    "compressor": "vae-v1",
    "log_every": 100,
    "val_every": 2_000,
    "val_windows": 1024,
    "val_seed": 0,
    "periodic_every": 8_000,
    "resume_every": 1_000,
    "eval_windows": None,  # None → the §9 set sizes
}
VAL_CHUNK = 128
STREAM_COLS = ("loss_video", "loss_action", "loss_action_hi_t", "loss_action_lo_t")
PERIODIC_COLS = ("rmse_steer", "rmse_throttle", "commit", "follow", "any_balance", "any_two_sided")
LOG_FIELDS = (["step", "loss", *STREAM_COLS, "grad_norm", "lr", "windows_per_s", "elapsed_s"]
              + [f"val_{c}" for c in STREAM_COLS] + [f"ema_val_{c}" for c in STREAM_COLS] + list(PERIODIC_COLS))


# -- provenance ------------------------------------------------------------------------------
def code_sha256() -> dict[str, str]:
    """sha256 of every src/wam/*.py, by file name."""
    return {p.name: _sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))}


def compressor_card_sha256(name: str, root: Path = CHECKPOINTS) -> str:
    card = Path(root) / name / "card.json"
    if not card.exists():
        raise FileNotFoundError(f"no compressor Checkpoint {name!r} in {root} (card.json missing)")
    return _sha256(card)


def _provenance(cfg: dict, dev: torch.device, root: Path) -> dict:
    commit, dirty = _git_info()
    data = data_path(cfg["data"])
    return {"git_commit": commit, "git_dirty": dirty, "code_sha256": code_sha256(), "torch_version": torch.__version__,
            "device": dev.type, "dataset": _rel(data), "manifest_sha256": _sha256(data / "manifest.json"),
            "compressor": cfg["compressor"], "compressor_card_sha256": compressor_card_sha256(cfg["compressor"], root)}


# -- optimisation pieces (SPEC §6.1) ---------------------------------------------------------
def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on Linear weights only (not norms, the Instruction embedding or biases)."""
    decay = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    params = list(model.parameters())
    return [{"params": [p for p in params if id(p) in decay], "weight_decay": weight_decay},
            {"params": [p for p in params if id(p) not in decay], "weight_decay": 0.0}]


def ema_decay(s: int, ema_max: float = 0.999) -> float:
    """EMA decay after optimiser step s (0-based): min(ema_max, (1 + s) / (10 + s))."""
    return min(ema_max, (1 + s) / (10 + s))


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1 - decay)


# -- data (SPEC §6.1, §6.2) ------------------------------------------------------------------
class WindowSampler:
    """Uniform over train Windows: a seeded permutation per pass from numpy default_rng([seed, 1])."""

    def __init__(self, windows, seed: int) -> None:
        self.w = windows
        self.rng = np.random.default_rng([seed, 1])
        self.pass_state = json.dumps(self.rng.bit_generator.state)
        self.perm = self.rng.permutation(len(windows))
        self.pos = 0
        self.passes = 0

    def indices(self, n: int) -> np.ndarray:
        out = []
        while n:
            if self.pos == len(self.perm):
                self.pass_state = json.dumps(self.rng.bit_generator.state)
                self.perm = self.rng.permutation(len(self.w))
                self.pos = 0
                self.passes += 1
            k = min(n, len(self.perm) - self.pos)
            out.append(self.perm[self.pos : self.pos + k])
            self.pos += k
            n -= k
        return np.concatenate(out)

    def get_state(self) -> str:
        return json.dumps({"pass_state": self.pass_state, "pos": self.pos, "passes": self.passes})

    def set_state(self, s: str) -> None:
        d = json.loads(s)
        self.rng.bit_generator.state = json.loads(d["pass_state"])
        self.pass_state = d["pass_state"]
        self.perm = self.rng.permutation(len(self.w))
        self.pos, self.passes = int(d["pos"]), int(d["passes"])


def render_batch(w, idx) -> dict[str, np.ndarray]:
    items = [w[int(i)] for i in idx]
    return {k: np.stack([it[k] for it in items]) for k in ("obs_frame", "future_frames", "state", "instruction",
                                                           "actions")}


def to_batch(raw: dict, comp: C.Compressor, cfg: ModelConfig, dev: torch.device) -> dict:
    """Encode with the frozen compressor; Latents are cloned out of inference mode (SPEC §16.9)."""
    obs, fut = comp.encode_windows({"obs_frame": raw["obs_frame"], "future_frames": raw["future_frames"]})
    return {"obs": obs.clone(), "fut": fut.clone() if cfg.video_stream else None,
            "state": torch.as_tensor(raw["state"], dtype=torch.float32, device=dev),
            "instr": torch.as_tensor(raw["instruction"], dtype=torch.long, device=dev),
            "actions": torch.as_tensor(raw["actions"], dtype=torch.float32, device=dev)}


class ValSet:
    """`eval_subset(val, n)` at the Run's H, encoded once, with fixed t and noise from a CPU generator seeded 0
    (independent of the Run seed) and p_uncond = 0 (SPEC §6.3, §16.3)."""

    def __init__(self, ds, cfg: ModelConfig, comp: C.Compressor, n: int, dev: torch.device, seed: int = 0) -> None:
        w = ds.split("val").eval_subset(n, horizon=cfg.horizon, future_frames=cfg.future_frames)
        self.cfg, self.n = cfg, len(w)
        parts = [to_batch(render_batch(w, range(i, min(i + VAL_CHUNK, len(w)))), comp, cfg, dev)
                 for i in range(0, len(w), VAL_CHUNK)]
        self.batch = {k: (None if parts[0][k] is None else torch.cat([p[k] for p in parts])) for k in parts[0]} \
            if parts else None
        self.draw = flow.draw(cfg, self.n, torch.Generator().manual_seed(seed), p_uncond=0.0).to(dev)

    @torch.no_grad()
    def losses(self, model: nn.Module) -> dict[str, float]:
        if not self.n:
            return {c: math.nan for c in STREAM_COLS}
        was = model.training
        model.eval()
        sv = sa = 0.0
        hi = lo = 0.0
        nhi = nlo = 0
        for i in range(0, self.n, VAL_CHUNK):
            sl = slice(i, i + VAL_CHUNK)
            b = {k: (None if v is None else v[sl]) for k, v in self.batch.items()}
            d = self.draw
            dd = flow.Draw(d.t[sl], d.uncond[sl], d.eps_a[sl], None if d.eps_z is None else d.eps_z[sl])
            _, parts = flow.loss(model, self.cfg, b, dd)
            m = b["obs"].shape[0]
            if parts["loss_video"] is not None:
                sv += float(parts["loss_video"]) * m
            sa += float(parts["loss_action"]) * m
            per, t = parts["action_per"].cpu().double(), parts["t"].cpu()
            h = t > flow.HI_T
            hi += float(per[h].sum())
            lo += float(per[~h].sum())
            nhi += int(h.sum())
            nlo += int((~h).sum())
        model.train(was)
        reg = self.cfg.objective == "regression"
        return {"loss_video": sv / self.n if self.cfg.video_stream else math.nan, "loss_action": sa / self.n,
                "loss_action_hi_t": hi / nhi if nhi and not reg else math.nan,
                "loss_action_lo_t": lo / nlo if nlo and not reg else math.nan}


# -- the Run --------------------------------------------------------------------------------
def make_config(config: dict) -> dict:
    """Defaults + overrides; `config["model"]` is a partial ModelConfig dict (validated)."""
    model = ModelConfig.from_dict(dict(config.get("model") or {}))
    cfg = {**DEFAULT_RECIPE, **{k: v for k, v in config.items() if k != "model"}}
    cfg["model"] = model.to_dict()
    cfg["name"] = cfg.get("name") or "wam"
    return cfg


def train(config: dict, out: Path, *, resume: bool = False, device=None, root: Path = CHECKPOINTS,
          _stop_at: int | None = None) -> Path:
    """Run the §6 recipe. `out` is the parent of a new Run directory, or the Run directory with `resume`.
    `root` holds the compressor Checkpoint. `_stop_at` (tests) saves `resume.pt` and stops after that step."""
    out = Path(out)
    if resume:
        run = out
        cfg = json.loads((run / "config.json").read_text())
        state = _read_weights(run / "resume.pt")
        dev = C.pick_device(device if device is not None else cfg["provenance"]["device"])
    else:
        cfg = make_config(config)
        dev = C.pick_device(device)
        cfg["data"] = _rel(data_path(cfg["data"]))
        cfg["provenance"] = _provenance(cfg, dev, root)
        run = _new_run_dir(out, cfg["name"])
        cfg["created"] = _now()
        cfg["resumes"] = []
        state = None

    with open(run / "stdout.txt", "a") as so, contextlib.redirect_stdout(_Tee(sys.stdout, so)):
        if resume:
            cfg["resumes"].append({"step": int(state["step"]), "at": _now(), "device": dev.type})
            print(f"resuming {_rel(run)} at step {int(state['step'])} on {dev}", flush=True)
        (run / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        res = _train_loop(cfg, run, dev, state, _stop_at, Path(root))
        if res is None:
            return run
        _finish(cfg, run, dev, *res)
    return run


class _Eval:
    """Lazily built eval data for the periodic sampling metrics."""

    def __init__(self, ds, cfg: ModelConfig, sz: dict) -> None:
        self.ds, self.cfg, self.sz = ds, cfg, sz
        self._sub = self._ref = None

    @property
    def sub(self) -> E.Subsets:
        if self._sub is None:
            self._sub = E.subsets(self.ds, self.cfg.horizon, self.sz)
        return self._sub

    @property
    def ref(self) -> np.ndarray:
        if self._ref is None:
            self._ref = E.fork_references(self.ds, self.cfg.horizon)
        return self._ref


def _load_compressor(cfg: dict, dev: torch.device, root: Path) -> C.Compressor:
    want = cfg["provenance"]["compressor_card_sha256"]
    if compressor_card_sha256(cfg["compressor"], root) != want:
        raise ValueError(f"compressor {cfg['compressor']!r} in {root} differs from the one this Run started with "
                         "(card.json sha256 mismatch)")
    return C.load(cfg["compressor"], dev, root=root)


def _train_loop(cfg: dict, run: Path, dev: torch.device, state: dict | None, stop_at: int | None, root: Path):
    from .policy import Policy

    mcfg = ModelConfig.from_dict(cfg["model"])
    steps, B, seed = int(cfg["steps"]), int(cfg["batch"]), int(cfg["seed"])
    model = build(mcfg, seed).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, float(cfg["weight_decay"])), lr=float(cfg["lr"]),
                            betas=tuple(cfg["betas"]))
    warm, final = int(cfg["warmup"]), float(cfg["lr_final"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: C.lr_factor(s, steps, warm, final))
    comp = _load_compressor(cfg, dev, root)
    ds = open_dataset(data_path(cfg["data"]), check=False)
    windows = ds.split("train").windows(horizon=mcfg.horizon, future_frames=mcfg.future_frames)
    sampler = WindowSampler(windows, seed)
    gen = torch.Generator().manual_seed(seed)
    val = ValSet(ds, mcfg, comp, int(cfg["val_windows"]), dev, int(cfg["val_seed"]))
    ev = _Eval(ds, mcfg, E.sizes(cfg["eval_windows"]))

    log_path = run / "log.csv"
    acc = torch.zeros(8, dtype=torch.float32, device=dev)  # loss, lv, la, gn, hi_sum, hi_n, lo_sum, lo_n
    start, n_acc, elapsed0 = 0, 0, 0.0
    if state is not None:
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        sampler.set_state(state["sampler"])
        gen.set_state(state["gen"])
        start, n_acc, elapsed0 = int(state["step"]), int(state["n_acc"]), float(state["elapsed"])
        acc = state["acc"].to(dev)
        with open(log_path, "r+b") as f:
            f.truncate(int(state["log_bytes"]))
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)

    log_every, val_every = int(cfg["log_every"]), int(cfg["val_every"])
    per_every, resume_every = int(cfg["periodic_every"]), int(cfg["resume_every"])
    ema_max, clip = float(cfg["ema_max"]), float(cfg["grad_clip"])
    print(f"training {_rel(run)}: {n_params(model):,} params, {mcfg.n_tokens} tokens, {len(windows):,} train "
          f"Windows, {steps} steps × batch {B} on {dev}", flush=True)

    def write_row(row: dict) -> None:
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(["" if row.get(k) is None or (isinstance(row.get(k), float) and math.isnan(row[k]))
                                    else (round(row[k], 6) if isinstance(row[k], float) else row[k])
                                    for k in LOG_FIELDS])

    def val_row(row: dict) -> dict:
        r, e = val.losses(model), val.losses(ema)
        row.update({f"val_{k}": v for k, v in r.items()})
        row.update({f"ema_val_{k}": v for k, v in e.items()})
        return e

    if state is None:
        row = {"step": 0}
        e = val_row(row)
        write_row(row)
        print(f"  step      0  val (EMA = raw): action {e['loss_action']:.4f}"
              + (f"  video {e['loss_video']:.4f}" if mcfg.video_stream else ""), flush=True)

    t0 = time.time() - elapsed0
    t_loop = t_int = time.time()
    w_int = 0
    model.train()
    for step in range(start + 1, steps + 1):
        raw = render_batch(windows, sampler.indices(B))
        batch = to_batch(raw, comp, mcfg, dev)
        d = flow.draw(mcfg, B, gen).to(dev)
        loss, parts = flow.loss(model, mcfg, batch, d)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        ema_update(ema, model, ema_decay(step - 1, ema_max))
        hi = (parts["t"] > flow.HI_T).float()
        per = parts["action_per"].float()
        lv = parts["loss_video"]
        acc += torch.stack([loss.detach().float(), (lv if lv is not None else torch.zeros((), device=dev)).float(),
                            parts["loss_action"].float(), gn.detach().float(), (per * hi).sum(), hi.sum(),
                            (per * (1 - hi)).sum(), (1 - hi).sum()])
        n_acc += 1
        w_int += B

        if step % log_every == 0 or step == steps:
            a = acc.cpu().double().numpy()
            if not np.isfinite(a).all():
                raise FloatingPointError(f"non-finite loss or gradient before step {step}; "
                                         "rerun with --resume from the last resume.pt")
            el = time.time() - t0
            wps = w_int / max(time.time() - t_int, 1e-9)
            lr = sched.get_last_lr()[0]
            reg = mcfg.objective == "regression"
            row = {"step": step, "loss": a[0] / n_acc, "loss_video": a[1] / n_acc if mcfg.video_stream else None,
                   "loss_action": a[2] / n_acc, "grad_norm": a[3] / n_acc,
                   "loss_action_hi_t": a[4] / a[5] if a[5] and not reg else None,
                   "loss_action_lo_t": a[6] / a[7] if a[7] and not reg else None,
                   "lr": float(f"{lr:.4e}"), "windows_per_s": round(wps, 1), "elapsed_s": round(el, 1)}
            msg = ""
            if step % val_every == 0 or step == steps:
                e = val_row(row)
                msg += f"  val(EMA) a {e['loss_action']:.4f}" + (f" v {e['loss_video']:.4f}" if mcfg.video_stream else "")
            if step % per_every == 0 or step == steps:
                pm = E.periodic(Policy(ema, comp), ev.sub, ev.ref, ev.sz)
                row.update(pm)
                msg += f"  rmse {pm['rmse_steer']:.3f}/{pm['rmse_throttle']:.3f} commit {pm['commit']:.2f}"
            write_row(row)
            eta = (time.time() - t_loop) / (step - start) * (steps - step)
            print(f"  step {step:6d}/{steps}  loss {row['loss']:.4f}"
                  + (f"  video {row['loss_video']:.4f}" if mcfg.video_stream else "")
                  + f"  action {row['loss_action']:.4f}  lr {lr:.1e}  {wps:5.1f} W/s  ETA {eta / 60:5.1f} min" + msg,
                  flush=True)
            acc.zero_()
            n_acc = 0
            t_int, w_int = time.time(), 0

        if (step % resume_every == 0 and step < steps) or step == stop_at:
            _save_atomic({
                "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step, "sampler": sampler.get_state(), "gen": gen.get_state(),
                "acc": acc.cpu(), "n_acc": n_acc, "elapsed": time.time() - t0, "log_bytes": log_path.stat().st_size,
            }, run / "resume.pt")
            if step == stop_at:
                print(f"stopped at step {step}; continue with: train --resume {_rel(run)}", flush=True)
                return None
    _sync(dev)
    train_s = time.time() - t0
    info = {"train_s": round(train_s, 1), "params": n_params(model), "tokens": mcfg.n_tokens,
            "windows_per_s": round(steps * B / max(train_s, 1e-9), 1), "passes": sampler.passes}
    return ema, comp, ds, ev, info


def read_log(run: Path) -> list[dict]:
    with open(Path(run) / "log.csv", newline="") as f:
        return list(csv.DictReader(f))


def checkpoint_dict(ema: nn.Module, cfg: dict) -> dict:
    return {"state_dict": {k: v.detach().cpu() for k, v in ema.state_dict().items()}, "config": cfg["model"],
            "compressor": cfg["compressor"], "compressor_card_sha256": cfg["provenance"]["compressor_card_sha256"]}


def _finish(cfg: dict, run: Path, dev: torch.device, ema: nn.Module, comp: C.Compressor, ds, ev: _Eval,
            info: dict) -> None:
    from .policy import Policy

    print(f"trained in {info['train_s'] / 60:.1f} min; evaluating …", flush=True)
    torch.save(checkpoint_dict(ema, cfg), run / "checkpoint.pt")
    policy = Policy(ema, comp)
    res = E.evaluate(policy, ds, sz=ev.sz, sub=ev.sub, ref=ev.ref, quiet=True)
    s7 = E.step7(read_log(run), policy.config.video_stream)
    verdict = "PASS" if s7["verdict"] == "PASS" and res["verdict"] == "PASS" else "FAIL"
    metrics = {"verdict": verdict, "step7": s7, **{k: v for k, v in res.items() if k != "verdict"},
               "gates_verdict": res["verdict"], **info}
    (run / "metrics.json").write_text(json.dumps(E.jsonable(metrics), indent=2, allow_nan=False) + "\n")
    E.print_report(res, s7)
    try:
        E.contact_sheet(run / "samples.png", f"{run.name}: {verdict}", E.sheet_data(policy, ds, ev.sub, ev.ref, ev.sz))
    except ImportError:
        print("contact sheet skipped (matplotlib missing; uv sync --group viz, then `show`)")
    (run / "resume.pt").unlink(missing_ok=True)
    print(f"verdict: {verdict}")
    print(_rel(run), flush=True)


__all__ = ["DEFAULT_RECIPE", "QUICK_STEPS", "REPO", "RUNS_DIR", "make_config", "train"]
