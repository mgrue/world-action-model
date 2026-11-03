"""PROTOTYPE — throwaway. Ticket 01 "PyTorch on Python 3.14 and MPS".

Question: how fast does a small stand-in conv autoencoder (64×64×3 → 8×8×16 → 64×64×3) train on
this Mac (MPS vs CPU), which precisions and torch.compile work, is a seeded run reproducible, and
does frame_dataset (re-rendering) keep up?

Run (from repo root; torch is NOT a project dependency yet):
  uv run --with torch==2.15.0 python .scratch/step4-vae/prototypes/torch_bench.py all
Commands: info | train | precision | compile | encode | data | e2e | determinism | all
Writes prototypes/out/torch_bench_<command>.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).parent / "out"
DATA = "data/step3-v1"
SECONDS = 4.0  # timed window per config (after warm-up)


# ---- stand-in model ----------------------------------------------------------------------------
def gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c // 4), c)


class Res(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.n1, self.c1, self.n2, self.c2 = gn(c), nn.Conv2d(c, c, 3, padding=1), gn(c), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class AE(nn.Module):
    """3 stride-2 stages: 64→32→16→8. Channels ch, 2ch, 4ch. 16 latent channels."""

    def __init__(self, ch: int = 32, z: int = 16, lean: bool = False) -> None:
        super().__init__()
        cs = [ch, ch * 2, ch * 4, ch * 4]
        full = (lambda c: nn.Identity()) if lean else Res  # lean: no residual block at 64×64
        enc = [nn.Conv2d(3, cs[0], 3, padding=1)]
        for i in range(3):
            enc += [full(cs[i]) if i == 0 else Res(cs[i]), nn.Conv2d(cs[i], cs[i + 1], 3, stride=2, padding=1)]
        enc += [Res(cs[3]), gn(cs[3]), nn.SiLU(), nn.Conv2d(cs[3], z, 1)]
        self.enc = nn.Sequential(*enc)
        dec = [nn.Conv2d(z, cs[3], 3, padding=1), Res(cs[3])]
        for i in (3, 2, 1):
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cs[i], cs[i - 1], 3, padding=1),
                    full(cs[0]) if i == 1 else Res(cs[i - 1])]
        dec += [gn(cs[0]), nn.SiLU(), nn.Conv2d(cs[0], 3, 3, padding=1)]
        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        return self.dec(self.enc(x))


SIZES = {"small": (32, False), "large": (48, False), "lean": (32, True)}


def build(size: str) -> AE:
    ch, lean = SIZES[size]
    return AE(ch, lean=lean)


def train_gflops(size: str) -> float:
    """Forward+backward GFLOP per Frame (FlopCounterMode, convs and matmuls only)."""
    from torch.utils.flop_counter import FlopCounterMode

    m, x = build(size), torch.zeros(1, 3, 64, 64)
    with FlopCounterMode(display=False) as fc:
        F.mse_loss(m(x), x).backward()
    return fc.get_total_flops() / 1e9


def enc_gflops(size: str) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    m = build(size)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m.enc(torch.zeros(1, 3, 64, 64))
    return fc.get_total_flops() / 1e9


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def to_input(u8: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC → float NCHW in [-1, 1] (done on the device)."""
    return u8.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()


def mem_mb(dev: str) -> tuple[float, float]:
    if dev != "mps":
        return (0.0, 0.0)
    return torch.mps.driver_allocated_memory() / 2**20, torch.mps.current_allocated_memory() / 2**20


def make_step(model, opt, dev: str, amp: torch.dtype | None):
    def step(xb):
        x = to_input(xb)
        if amp is None:
            loss = F.mse_loss(model(x), x)
        else:
            with torch.autocast(device_type=dev, dtype=amp):
                loss = F.mse_loss(model(x).float(), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    return step


def bench_train(dev: str, size: str, batch: int, amp: torch.dtype | None = None, compile_: bool = False,
                seconds: float = SECONDS) -> dict:
    torch.manual_seed(0)
    model = build(size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    step = make_step(model, opt, dev, amp)
    if compile_:
        step = torch.compile(step)
    xs = torch.randint(0, 256, (8, batch, 64, 64, 3), dtype=torch.uint8, device=dev)
    res = dict(device=dev, size=size, params=n_params(model), batch=batch,
               precision=str(amp).replace("torch.", "") if amp else "float32", compile=compile_)
    try:
        t0 = time.perf_counter()
        for i in range(3):  # warm-up (and compile)
            loss = step(xs[i % 8])
        float(loss.detach())
        sync(dev)
        res["warmup_s"] = round(time.perf_counter() - t0, 2)
        n, peak, t0 = 0, 0.0, time.perf_counter()
        while time.perf_counter() - t0 < seconds or n < 3:
            loss = step(xs[n % 8])
            n += 1
            if n % 5 == 0:
                sync(dev)
                peak = max(peak, mem_mb(dev)[0])
        float(loss.detach())
        sync(dev)
        dt = time.perf_counter() - t0
        drv, cur = mem_mb(dev)
        res.update(ok=True, steps=n, steps_s=round(n / dt, 2), frames_s=round(n * batch / dt),
                   ms_step=round(1000 * dt / n, 1), loss=round(float(loss.detach()), 4),
                   finite=bool(torch.isfinite(loss)))
        if dev == "mps":
            res.update(driver_mb_peak=round(max(peak, drv)), current_mb=round(cur))
        res["tflops"] = round(res["frames_s"] * train_gflops(size) / 1000, 2)
    except Exception as e:  # noqa: BLE001
        res.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
    del model, opt, xs
    if dev == "mps":
        torch.mps.empty_cache()
    print(json.dumps(res), flush=True)
    return res


# ---- commands ----------------------------------------------------------------------------------
def cmd_info(a) -> dict:
    import numpy

    import toycar

    r = dict(torch=torch.__version__, python=sys.version.split()[0], numpy=numpy.__version__, macos=platform.mac_ver()[0],
             machine=platform.machine(), mps_available=torch.backends.mps.is_available(),
             mps_recommended_max_gb=round(torch.mps.recommended_max_memory() / 2**30, 2), cpu_threads=torch.get_num_threads(),
             toycar=getattr(toycar, "__version__", "?"),
             params={k: n_params(build(k)) for k in SIZES}, enc_params={k: n_params(build(k).enc) for k in SIZES},
             train_gflop_per_frame={k: round(train_gflops(k), 3) for k in SIZES},
             enc_gflop_per_frame={k: round(enc_gflops(k), 3) for k in SIZES})
    print(json.dumps(r))
    return r


def cmd_train(a) -> list:
    rows = []
    for size in SIZES:
        for b in (64, 128, 256):
            rows.append(bench_train("mps", size, b))
    for size, b in (("small", 64), ("small", 128), ("large", 64), ("lean", 64), ("lean", 128)):
        rows.append(bench_train("cpu", size, b, seconds=5.0))
    return rows


def cmd_precision(a) -> list:
    rows = []
    for size in SIZES:
        for amp in (torch.float16, torch.bfloat16):
            rows.append(bench_train("mps", size, 128, amp=amp))
    rows.append(bench_train("mps", "small", 256, amp=torch.float16))
    rows.append(bench_train("cpu", "small", 64, amp=torch.bfloat16, seconds=5.0))
    return rows


def cmd_compile(a) -> list:
    rows = []
    for size in SIZES:
        rows.append(bench_train("mps", size, 256, compile_=True))
    rows.append(bench_train("mps", "small", 256, amp=torch.float16, compile_=True))
    rows.append(bench_train("cpu", "small", 64, compile_=True, seconds=5.0))
    return rows


def cmd_encode(a) -> list:
    """Encoder-only forward (no grad), what Steps 6–8 pay per Window (17 Frames) if encoding on the fly."""
    rows = []
    for dev in ("mps", "cpu"):
        for size in SIZES:
            model = build(size).to(dev).eval()
            for amp in (None, torch.float16):
                if dev == "cpu" and amp is not None:
                    continue
                for windows in (8, 32):
                    xs = torch.randint(0, 256, (windows * 17, 64, 64, 3), dtype=torch.uint8, device=dev)
                    with torch.no_grad(), torch.autocast(dev, dtype=amp or torch.float32, enabled=amp is not None):
                        for _ in range(3):
                            z = model.enc(to_input(xs))
                        sync(dev)
                        n, t0 = 0, time.perf_counter()
                        while time.perf_counter() - t0 < 2.0:
                            z = model.enc(to_input(xs))
                            n += 1
                        sync(dev)
                        dt = time.perf_counter() - t0
                    r = dict(device=dev, size=size, precision=str(amp or "float32").replace("torch.", ""), windows=windows,
                             frames=windows * 17, enc_params=n_params(model.enc), latent_shape=list(z.shape[1:]),
                             ms_batch=round(1000 * dt / n, 2), ms_per_window=round(1000 * dt / n / windows, 3),
                             frames_s=round(n * windows * 17 / dt))
                    print(json.dumps(r), flush=True)
                    rows.append(r)
            del model
    return rows


def _frames():
    from toycar.dataset import frame_dataset, open_dataset

    return frame_dataset(open_dataset(DATA, check=False).split("train"))


def cmd_data(a) -> list:
    from torch.utils.data import DataLoader, RandomSampler

    fd = _frames()
    rows = []
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(fd), 20_000)
    t0 = time.perf_counter()
    for i in idx:
        fd[int(i)]
    dt = time.perf_counter() - t0
    r = dict(mode="in-process loop", n_frames=len(fd), frames_s=round(len(idx) / dt))
    print(json.dumps(r), flush=True)
    rows.append(r)
    for nw in (0, 2, 4, 6):
        for b in (256,):
            g = torch.Generator().manual_seed(0)
            dl = DataLoader(fd, batch_size=b, sampler=RandomSampler(fd, generator=g), num_workers=nw,
                            persistent_workers=nw > 0, prefetch_factor=4 if nw else None)
            t0 = time.perf_counter()
            it = iter(dl)
            first = next(it)
            t_first = time.perf_counter() - t0
            nb, t0 = 0, time.perf_counter()
            while nb < 60:
                xb = next(it)
                nb += 1
            dt = time.perf_counter() - t0
            r = dict(mode="DataLoader", num_workers=nw, batch=b, startup_s=round(t_first, 2),
                     frames_s=round(nb * b / dt), dtype=str(xb.dtype), shape=list(first.shape))
            print(json.dumps(r), flush=True)
            rows.append(r)
            del it, dl
    return rows


def cmd_e2e(a) -> list:
    """Real Frames from frame_dataset through a DataLoader into an MPS training step."""
    from torch.utils.data import DataLoader, RandomSampler

    fd = _frames()
    rows = []
    for size, amp in (("small", None), ("lean", None), ("lean", torch.float16)):
        for nw in (0, 4):
            torch.manual_seed(0)
            model = build(size).to("mps")
            opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
            step = make_step(model, opt, "mps", amp)
            b = 256
            dl = DataLoader(fd, batch_size=b, sampler=RandomSampler(fd, generator=torch.Generator().manual_seed(0)),
                            num_workers=nw, persistent_workers=nw > 0, prefetch_factor=4 if nw else None)
            it = iter(dl)
            for _ in range(3):
                loss = step(next(it).to("mps", non_blocking=True))
            float(loss)
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 8.0:
                loss = step(next(it).to("mps", non_blocking=True))
                n += 1
            float(loss)
            dt = time.perf_counter() - t0
            r = dict(size=size, precision=str(amp or "float32").replace("torch.", ""), num_workers=nw, batch=b,
                     steps_s=round(n / dt, 2), frames_s=round(n * b / dt), loss=round(float(loss), 4))
            print(json.dumps(r), flush=True)
            rows.append(r)
            del it, dl, model, opt
            torch.mps.empty_cache()
    return rows


def _seeded_run(dev: str, steps: int = 20, deterministic: bool = False) -> list[float]:
    torch.manual_seed(123)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    try:
        model = build("small").to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        step = make_step(model, opt, dev, None)
        g = torch.Generator().manual_seed(7)
        xs = torch.randint(0, 256, (steps, 64, 64, 64, 3), dtype=torch.uint8, generator=g).to(dev)
        losses = [float(step(xs[i]).detach()) for i in range(steps)]
        losses.append(float(sum(p.detach().cpu().double().sum() for p in model.parameters())))  # final param checksum
        return losses
    finally:
        torch.use_deterministic_algorithms(False)


def cmd_determinism(a) -> dict:
    r = {}
    for dev in ("mps", "cpu"):
        a1, a2 = _seeded_run(dev), _seeded_run(dev)
        r[dev] = dict(identical=a1 == a2, max_abs_diff=max(abs(x - y) for x, y in zip(a1, a2)),
                      first_losses=[round(x, 6) for x in a1[:3]], final_loss_1=a1[-2], final_loss_2=a2[-2],
                      param_sum_1=a1[-1], param_sum_2=a2[-1])
        try:
            d1, d2 = _seeded_run(dev, deterministic=True), _seeded_run(dev, deterministic=True)
            r[dev]["use_deterministic_algorithms"] = dict(ok=True, identical=d1 == d2)
        except Exception as e:  # noqa: BLE001
            r[dev]["use_deterministic_algorithms"] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    m, c = _seeded_run("mps"), _seeded_run("cpu")
    r["mps_vs_cpu"] = dict(max_abs_loss_diff=max(abs(x - y) for x, y in zip(m[:-1], c[:-1])),
                           rel_final_loss_diff=abs(m[-2] - c[-2]) / c[-2])
    print(json.dumps(r, indent=1))
    return r


CMDS = dict(info=cmd_info, train=cmd_train, precision=cmd_precision, compile=cmd_compile, encode=cmd_encode,
            data=cmd_data, e2e=cmd_e2e, determinism=cmd_determinism)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=[*CMDS, "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in CMDS if a.cmd == "all" else [a.cmd]:
        print(f"== {name}", flush=True)
        t0 = time.perf_counter()
        res = CMDS[name](a)
        (OUT / f"torch_bench_{name}.json").write_text(json.dumps(res, indent=1, default=str))
        print(f"== {name} done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
