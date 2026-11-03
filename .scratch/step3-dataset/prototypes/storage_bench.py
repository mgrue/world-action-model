"""PROTOTYPE — throwaway. Ticket 01 "Storage format and loading throughput".

Question: which on-disk format lets Step 4 (random Frames) and Steps 6–8 (random windows of
1 + 16 Frames, State, 16 actions) read fast enough on a 16 GB Mac with ~34 GB free disk?

Run (from repo root):
  uv run --with torch --with h5py --with zarr python .scratch/step3-dataset/prototypes/storage_bench.py all --data DIR [--n 150]
Commands: record | write | read | loader | render | train | all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from toycar import CarEnv, RenderParams, State, Track, render
from toycar.expert import Expert, run_episode, start_spec  # noqa: F401

T = 900  # ticks per episode
H = 16  # future length
W = H + 1  # frames per window
COMBOS = [(i, d) for i in ("outer", "inner", "any") for d in ("cw", "ccw")]


# ---- recording ---------------------------------------------------------------------------------
def record_one(ep: int) -> dict:
    instr, direction = COMBOS[ep % 6]
    seed = 10_000 + ep
    env = CarEnv()
    state, info = env.reset(seed, start_spec(instr, direction))
    ex = Expert(env.track, instr)
    ex.reset(seed, state, info)
    frames = [env.render()]
    states = [[state.x, state.y, state.theta, state.v]]
    clean, executed, pert = [], [], []
    while True:
        a = ex.act(state, info)
        r = env.step(a.executed)
        state, info = r.state, r.info
        frames.append(env.render())
        states.append([state.x, state.y, state.theta, state.v])
        clean.append(a.clean), executed.append(a.executed), pert.append(a.perturbed)
        if r.terminated or r.truncated:
            break
    return dict(frames=np.stack(frames), states=np.array(states), clean=np.array(clean, np.float32),
                executed=np.array(executed, np.float32), perturbed=np.array(pert), seed=seed, instr=instr)


def record(data: Path, n: int) -> None:
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        for ep, d in enumerate(pool.map(record_one, range(n), chunksize=4)):
            np.savez(raw / f"{ep:05d}.npz", **{k: v for k, v in d.items() if k != "instr"}, instr=d["instr"])
    print(f"recorded {n} episodes in {time.time() - t0:.1f} s (sim + render, parallel)")


def episodes(data: Path):
    for p in sorted((data / "raw").glob("*.npz")):
        with np.load(p) as z:
            yield {k: z[k] for k in z.files}


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


# ---- formats -----------------------------------------------------------------------------------
FORMATS = ["npz_ep", "npy_memmap", "h5_lzf", "h5_gzip", "zarr_blosc", "states_only"]


def write(data: Path, fmt: str, n: int) -> tuple[float, int]:
    out = data / fmt
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    t0 = time.time()
    if fmt == "npz_ep":
        for ep, d in enumerate(episodes(data)):
            np.savez_compressed(out / f"{ep:05d}.npz", **d)
    elif fmt == "npy_memmap":
        F = np.lib.format.open_memmap(out / "frames.npy", "w+", np.uint8, (n, T + 1, 64, 64, 3))
        S = np.zeros((n, T + 1, 4)); A = np.zeros((n, T, 2), np.float32)
        for ep, d in enumerate(episodes(data)):
            F[ep] = d["frames"]; S[ep] = d["states"]; A[ep] = d["clean"]
        F.flush(); del F
        np.save(out / "states.npy", S); np.save(out / "actions.npy", A)
    elif fmt.startswith("h5"):
        import h5py

        comp = dict(compression="lzf") if fmt == "h5_lzf" else dict(compression="gzip", compression_opts=4)
        with h5py.File(out / "data.h5", "w") as f:
            F = f.create_dataset("frames", (n, T + 1, 64, 64, 3), np.uint8, chunks=(1, W, 64, 64, 3), **comp)
            S = f.create_dataset("states", (n, T + 1, 4), np.float64)
            A = f.create_dataset("actions", (n, T, 2), np.float32)
            for ep, d in enumerate(episodes(data)):
                F[ep] = d["frames"]; S[ep] = d["states"]; A[ep] = d["clean"]
    elif fmt == "zarr_blosc":
        import zarr

        root = zarr.open_group(out / "data.zarr", mode="w")
        F = root.create_array("frames", shape=(n, T + 1, 64, 64, 3), chunks=(1, W, 64, 64, 3), dtype="uint8")
        S = root.create_array("states", shape=(n, T + 1, 4), chunks=(1, T + 1, 4), dtype="float64")
        A = root.create_array("actions", shape=(n, T, 2), chunks=(1, T, 2), dtype="float32")
        for ep, d in enumerate(episodes(data)):
            F[ep] = d["frames"]; S[ep] = d["states"]; A[ep] = d["clean"]
    elif fmt == "states_only":
        S = np.zeros((n, T + 1, 4)); A = np.zeros((n, T, 2), np.float32)
        for ep, d in enumerate(episodes(data)):
            S[ep] = d["states"]; A[ep] = d["clean"]
        np.save(out / "states.npy", S); np.save(out / "actions.npy", A)
    return time.time() - t0, dir_size(out)


class Reader:
    """Random window access: frames[t : t+17], state[t], actions[t : t+16]."""

    def __init__(self, data: Path, fmt: str) -> None:
        self.fmt, self.dir = fmt, data / fmt
        self.track = Track.default()
        self._open = False

    def _lazy(self) -> None:  # open per process (DataLoader workers)
        if self._open:
            return
        d, f = self.dir, self.fmt
        if f == "npz_ep":
            self.files = sorted(d.glob("*.npz"))
        elif f in ("npy_memmap", "states_only"):
            if f == "npy_memmap":
                self.F = np.load(d / "frames.npy", mmap_mode="r")
            self.S = np.load(d / "states.npy", mmap_mode="r"); self.A = np.load(d / "actions.npy", mmap_mode="r")
        elif f.startswith("h5"):
            import h5py

            self.h = h5py.File(d / "data.h5", "r")
            self.F, self.S, self.A = self.h["frames"], self.h["states"], self.h["actions"]
        elif f == "zarr_blosc":
            import zarr

            g = zarr.open_group(d / "data.zarr", mode="r")
            self.F, self.S, self.A = g["frames"], g["states"], g["actions"]
        self._open = True

    def window(self, ep: int, t: int):
        self._lazy()
        if self.fmt == "npz_ep":
            with np.load(self.files[ep]) as z:
                return z["frames"][t : t + W], z["states"][t], z["clean"][t : t + H]
        if self.fmt == "states_only":
            st = self.S[ep, t : t + W]
            fr = np.stack([render(State(*row), self.track) for row in st])
            return fr, np.array(self.S[ep, t]), np.array(self.A[ep, t : t + H])
        return np.asarray(self.F[ep, t : t + W]), np.asarray(self.S[ep, t]), np.asarray(self.A[ep, t : t + H])

    def frame(self, ep: int, t: int):
        self._lazy()
        if self.fmt == "npz_ep":
            with np.load(self.files[ep]) as z:
                return z["frames"][t]
        if self.fmt == "states_only":
            return render(State(*self.S[ep, t]), self.track)
        return np.asarray(self.F[ep, t])


def bench_read(data: Path, fmt: str, n: int, seconds: float = 3.0) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    r = Reader(data, fmt)
    r.window(0, 0)
    out = []
    for what in ("window", "frame"):
        k, t0 = 0, time.time()
        while time.time() - t0 < seconds:
            ep, t = int(rng.integers(n)), int(rng.integers(T - H))
            r.window(ep, t) if what == "window" else r.frame(ep, t)
            k += 1
        out.append(k / (time.time() - t0))
    return out[0], out[1]


# ---- torch DataLoader ---------------------------------------------------------------------------
class WindowDS:
    def __init__(self, data, fmt, n):
        self.r, self.n = Reader(data, fmt), n

    def __len__(self):
        return self.n * (T - H)

    def __getitem__(self, i):
        ep, t = divmod(i, T - H)
        return self.r.window(ep, t)


def bench_loader(data: Path, fmt: str, n: int, workers: int, batch: int = 64, batches: int = 30) -> float:
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(WindowDS(data, fmt, n), batch_size=batch, shuffle=True, num_workers=workers,
                    persistent_workers=workers > 0, generator=torch.Generator().manual_seed(0))
    it = iter(dl)
    next(it)  # warm-up / worker start
    t0 = time.time()
    for _ in range(batches):
        next(it)
    return batch * batches / (time.time() - t0)


# ---- render determinism across processes --------------------------------------------------------
def _hash_render(rows):
    track = Track.default()
    h = hashlib.sha256()
    for row in rows:
        h.update(render(State(*row), track, RenderParams()).tobytes())
    return h.hexdigest()


def bench_render(data: Path) -> dict:
    d = next(episodes(data))
    rows = [tuple(map(float, r)) for r in d["states"]]
    stored = hashlib.sha256(b"".join(f.tobytes() for f in d["frames"])).hexdigest()
    with ProcessPoolExecutor(2) as pool:
        remote = list(pool.map(_hash_render, [rows, rows]))
    local = _hash_render(rows)
    track = Track.default()
    t0 = time.time()
    for row in rows:
        render(State(*row), track)
    ms = 1e3 * (time.time() - t0) / len(rows)
    return dict(bit_identical=len({stored, local, *remote}) == 1, render_ms=ms)


# ---- training step budget on MPS ----------------------------------------------------------------
def bench_train() -> dict:
    import torch
    from torch import nn

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    res = {}
    # PLAN: 8×8 latents per Frame → 64 tokens; 1 obs Frame + 16 future + state + instr + 16 actions
    seq = 64 * 17 + 1 + 1 + 16
    for d_model, layers in ((256, 8), (384, 8)):
        layer = nn.TransformerEncoderLayer(d_model, 8, 4 * d_model, batch_first=True, norm_first=True)
        model = nn.TransformerEncoder(layer, layers).to(dev)
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), 1e-4)
        for batch in (4, 8, 16, 32):
            key = f"d{d_model}x{layers} ({params / 1e6:.1f} M) b{batch}"
            try:
                x = torch.randn(batch, seq, d_model, device=dev)
                for i in range(8):
                    if i == 3:
                        torch.mps.synchronize() if dev == "mps" else None
                        t0 = time.time()
                    loss = model(x).pow(2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                torch.mps.synchronize() if dev == "mps" else None
                dt = (time.time() - t0) / 5
                res[key] = f"{dt * 1e3:.0f} ms/step, {batch / dt:.0f} windows/s"
            except RuntimeError as e:
                res[key] = "OOM" if "out of memory" in str(e) else repr(e)[:80]
            finally:
                x = loss = None
                torch.mps.empty_cache() if dev == "mps" else None
            print(key, res[key], flush=True)
    return res


# ---- main ----------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "write", "read", "loader", "render", "train", "all"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    n, data = a.n, a.data
    results = {}
    if a.cmd in ("record", "all"):
        record(data, n)
    if a.cmd in ("write", "all"):
        for fmt in FORMATS:
            dt, size = write(data, fmt, n)
            results.setdefault(fmt, {})["write_s_per_ep"] = dt / n
            results[fmt]["MB_per_ep"] = size / n / 1e6
            results[fmt]["GB_at_3000"] = size / n * 3000 / 1e9
            print(fmt, results[fmt], flush=True)
    if a.cmd in ("read", "all"):
        for fmt in FORMATS:
            w, f = bench_read(data, fmt, n)
            results.setdefault(fmt, {}).update(windows_per_s=w, frames_per_s=f)
            print(fmt, f"{w:8.0f} windows/s  {f:8.0f} frames/s (1 process, random)", flush=True)
    if a.cmd in ("loader", "all"):
        for fmt in FORMATS:
            for workers in (0, 4):
                wps = bench_loader(data, fmt, n, workers)
                results.setdefault(fmt, {})[f"loader_w{workers}_windows_per_s"] = wps
                print(fmt, f"DataLoader workers={workers}: {wps:8.0f} windows/s", flush=True)
    if a.cmd in ("render", "all"):
        results["render"] = bench_render(data)
        print("render", results["render"], flush=True)
    if a.cmd in ("train", "all"):
        results["train"] = bench_train()
        print("train", results["train"], flush=True)
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    (out / f"storage_bench_{a.cmd}.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
