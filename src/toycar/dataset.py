"""Recorded expert episodes and training Windows (Step 3 SPEC, `.scratch/step3-dataset/SPEC.md`).

    uv run python -m toycar.dataset record [--out data/step3-v1] [--workers K] [--force]
    uv run python -m toycar.dataset verify [DIR]
    uv run python -m toycar.dataset stats  [DIR]
    uv run --group viz python -m toycar.dataset plot [DIR] [--split val] [--episode I --t T | --seed N]
                                                    [--approaching] [--probe NAME] [--zoom M] [--out window.png]

numpy only; matplotlib is imported lazily by `plot`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import numpy as np

from . import __version__
from .env import CarEnv
from .expert import INSTRUCTIONS, Expert, ExpertParams, Instruction, start_spec
from .physics import CAR_LENGTH
from .render import RGB, RenderParams, render
from .state import Branch, Direction, State
from .track import WORLD_M, Track, TrackParams

# -- recorded set (SPEC §2) -----------------------------------------------------------------
Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")
SEED_BLOCKS: dict[Split, int] = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000}
PROBE_SEED_BLOCK = 4_000_000  # optional wide_fork recordings (Step 10)
CLOSED_LOOP_SEED_BLOCK = 9_000_000  # Step 9 closed-loop evaluation; never recorded
DEFAULT_EPISODES: dict[Split, int] = {"train": 4800, "val": 600, "test": 600}
EPISODE_TICKS = 300
DIRECTIONS: tuple[Direction, ...] = ("cw", "ccw")  # index = integer id
COMBINATIONS: tuple[tuple[Instruction, Direction], ...] = tuple((i, d) for i in INSTRUCTIONS for d in DIRECTIONS)

DEFAULT_DIR = Path("data/step3-v1")
DEFAULT_HORIZON = 16
FORMAT_VERSION = 1
SOURCE_FILES = ("render.py", "track.py", "physics.py", "env.py", "expert.py")
OPEN_CHECK_EPISODES = 4

_BRANCH_ID: dict[Branch | None, int] = {None: -1, "outer": 0, "inner": 1}
_BRANCHES: tuple[Branch, ...] = ("outer", "inner")

# -- visual randomisation (SPEC §3.1) -------------------------------------------------------
COLOUR_FIELDS = ("grass", "road", "kerb", "body", "nose")  # order is part of the contract
BRIGHTNESS_RANGE = (0.85, 1.15)
CHANNEL_OFFSET = 10.0


def sample_render_params(seed: int) -> RenderParams:
    """Per-episode colour jitter from the visual stream [seed, 3]."""
    rng = np.random.default_rng([seed, 3])
    colours = {}
    for name in COLOUR_FIELDS:
        default = getattr(RenderParams(), name)
        b = rng.uniform(*BRIGHTNESS_RANGE)
        colours[name] = tuple(
            int(np.clip(round(c * b + rng.uniform(-CHANNEL_OFFSET, CHANNEL_OFFSET)), 0, 255)) for c in default
        )
    return RenderParams(**colours)


# -- Probes (SPEC §3.2) ---------------------------------------------------------------------
@dataclass(frozen=True)
class Probe:
    name: str
    colours: dict[str, RGB | str]  # field → RGB, or the name of another field of the same episode to copy
    track_params: TrackParams | None  # None = default Track

    def apply(self, rp: RenderParams) -> RenderParams:
        new = {f: (getattr(rp, v) if isinstance(v, str) else tuple(v)) for f, v in self.colours.items()}
        return dataclasses.replace(rp, **new)


PROBES: dict[str, Probe] = {
    "brown_grass": Probe("brown_grass", {"grass": (120, 100, 50)}, None),
    "light_road": Probe("light_road", {"road": (140, 140, 145)}, None),
    "blue_car": Probe("blue_car", {"body": (40, 80, 220)}, None),
    "no_kerb": Probe("no_kerb", {"kerb": "road"}, None),
    "wide_fork": Probe("wide_fork", {}, TrackParams(bulge_m=7.0)),
}


def _colour_probe(probe: str | None) -> Probe | None:
    if probe is None:
        return None
    p = PROBES.get(probe)
    if p is None:
        raise ValueError(f"unknown probe {probe!r}; known: {sorted(PROBES)}")
    if p.track_params is not None:
        raise ValueError(f"probe {probe!r} changes the Track (and the driving); it needs closed-loop runs or a new recording")
    return p


def _rp_to_array(rp: RenderParams) -> np.ndarray:
    return np.array([getattr(rp, f) for f in COLOUR_FIELDS], dtype=np.uint8)


def _rp_from_array(a: np.ndarray) -> RenderParams:
    return RenderParams(**{f: tuple(int(c) for c in a[j]) for j, f in enumerate(COLOUR_FIELDS)})


@lru_cache(maxsize=4)
def _track(params: TrackParams) -> Track:
    return Track(params)


# -- recording one episode (SPEC §5.1) ------------------------------------------------------
_PER_TICK = ("states", "clean", "executed", "perturbed", "planned_branch", "station_m", "branch")


def _record_episode(job: tuple[Split, int, int]) -> dict:
    """Pure function of (split, k, length): the episode's arrays plus Passage bookkeeping."""
    split, k, T = job
    seed = SEED_BLOCKS[split] + k
    instruction, direction = COMBINATIONS[k % len(COMBINATIONS)]
    rp = sample_render_params(seed)
    env = CarEnv(track=_track(TrackParams()), max_steps=T)
    track = env.track
    state, info = env.reset(seed, start_spec(instruction, direction), render_params=rp)
    expert = Expert(track, instruction, ExpertParams())
    expert.reset(seed, state, info)

    states = np.empty((T + 1, 4), np.float64)
    clean = np.empty((T, 2), np.float32)
    executed = np.empty((T, 2), np.float32)
    perturbed = np.empty(T, bool)
    planned = np.empty(T, np.int8)
    station = np.empty(T + 1, np.float32)
    branch = np.empty(T + 1, np.int8)
    h = hashlib.sha256()
    passages: list[int] = []

    def put(t: int, s: State, inf) -> None:
        states[t] = (s.x, s.y, s.theta, s.v)
        station[t] = inf.station_m
        branch[t] = _BRANCH_ID[inf.branch]
        h.update(render(s, track, rp).tobytes())

    put(0, state, info)
    r = None
    for t in range(T):
        a = expert.act(state, info)
        clean[t] = a.clean
        executed[t] = a.executed
        perturbed[t] = a.perturbed
        planned[t] = _BRANCH_ID[expert.planned_branch]
        prev_passages = info.passages
        r = env.step(a.executed)
        state, info = r.state, r.info
        put(t + 1, state, info)
        if info.passages > prev_passages:
            passages.append(_BRANCH_ID[info.last_branch])
        if r.terminated:
            raise RuntimeError(
                f"expert crashed while recording: split {split}, k {k}, seed {seed}, "
                f"instruction {instruction}, direction {direction}, tick {t + 1}"
            )
    assert r is not None and r.truncated and info.steps == T, (split, k, info)

    return {
        "states": states, "clean": clean, "executed": executed, "perturbed": perturbed,
        "planned_branch": planned, "station_m": station, "branch": branch,
        "k": k, "seed": seed,
        "instruction": INSTRUCTIONS.index(instruction), "direction": DIRECTIONS.index(direction),
        "start_route": -1 if instruction == "any" else _BRANCH_ID[instruction],
        "length": T, "render_params": _rp_to_array(rp), "frames_sha256": h.hexdigest(),
        "passages": passages,
    }


def _stack(results: Sequence[dict], T: int) -> dict[str, np.ndarray]:
    E = len(results)
    if E == 0:
        shapes = {"states": ((0, T + 1, 4), np.float64), "clean": ((0, T, 2), np.float32),
                  "executed": ((0, T, 2), np.float32), "perturbed": ((0, T), bool),
                  "planned_branch": ((0, T), np.int8), "station_m": ((0, T + 1), np.float32),
                  "branch": ((0, T + 1), np.int8)}
        out = {n: np.empty(s, d) for n, (s, d) in shapes.items()}
    else:
        out = {n: np.stack([r[n] for r in results]) for n in _PER_TICK}
    out["k"] = np.array([r["k"] for r in results], np.int32)
    out["seed"] = np.array([r["seed"] for r in results], np.int64)
    out["instruction"] = np.array([r["instruction"] for r in results], np.int8)
    out["direction"] = np.array([r["direction"] for r in results], np.int8)
    out["start_route"] = np.array([r["start_route"] for r in results], np.int8)
    out["length"] = np.array([r["length"] for r in results], np.int32)
    out["render_params"] = np.array([r["render_params"] for r in results], np.uint8).reshape(E, len(COLOUR_FIELDS), 3)
    out["frames_sha256"] = np.array([r["frames_sha256"] for r in results], "<U64")
    return out


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Same layout as `np.savez` (uncompressed, no pickle) but with fixed zip timestamps,
    so the bytes don't depend on when the file was written."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            with zf.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)


# -- Window index (SPEC §6.2) ---------------------------------------------------------------
def _window_index(
    perturbed: np.ndarray, branch: np.ndarray, length: np.ndarray, horizon: int, clean_future: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(index (N, 2) int32, approaches_section (N,), in_section (N,)), vectorised over episodes."""
    E, T = perturbed.shape
    P = T - horizon + 1
    if E == 0 or P <= 0:
        return np.empty((0, 2), np.int32), np.empty(0, bool), np.empty(0, bool)
    ts = np.arange(P)
    valid = ts[None, :] <= (length[:, None] - horizon)
    if clean_future:
        c = np.concatenate([np.zeros((E, 1), np.int64), np.cumsum(perturbed, axis=1)], axis=1)
        valid &= (c[:, horizon : horizon + P] - c[:, :P]) == 0
    e, t = np.nonzero(valid)
    in_sec = branch != -1
    cs = np.concatenate([np.zeros((E, 1), np.int64), np.cumsum(in_sec, axis=1)], axis=1)
    ahead = cs[e, t + horizon + 1] - cs[e, t + 1]  # count over branch[t+1 … t+H]
    here = in_sec[e, t]
    return np.stack([e, t], axis=1).astype(np.int32), (~here) & (ahead > 0), here


# -- manifest (SPEC §4.3) -------------------------------------------------------------------
def _source_hashes() -> dict[str, str]:
    pkg = Path(__file__).parent
    return {name: hashlib.sha256((pkg / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _git_info() -> tuple[str | None, bool | None]:
    pkg = Path(__file__).parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=pkg, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=pkg, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit.stdout.strip(), bool(status.stdout.strip())


def _jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    return obj


def _summary(arrays: Mapping[Split, dict[str, np.ndarray]], passages: Mapping[Split, list[list[int]]]) -> dict:
    def outer_fraction(splits: Sequence[Split]) -> float | None:
        scored: list[int] = []
        for sp in splits:
            a = arrays[sp]
            for i, ps in enumerate(passages[sp]):
                if INSTRUCTIONS[a["instruction"][i]] == "any":
                    scored += ps[1:] if a["branch"][i, 0] != -1 else ps
        return sum(b == 0 for b in scored) / len(scored) if scored else None

    per_split: dict = {}
    combos: dict = {}
    tot_ticks = tot_pert = tot_win = tot_appr = 0
    for sp in SPLITS:
        a = arrays[sp]
        idx, appr, _ = _window_index(a["perturbed"], a["branch"], a["length"], DEFAULT_HORIZON, True)
        ticks = int(a["length"].sum())
        pert = int(a["perturbed"].sum())
        per_split[sp] = {
            "episodes": len(a["k"]), "ticks": ticks, "windows": len(idx),
            "perturbed_fraction": pert / ticks if ticks else None,
            "approaches_section_fraction": float(appr.mean()) if len(idx) else None,
            "any_outer_fraction": outer_fraction([sp]),
        }
        tot_ticks, tot_pert, tot_win, tot_appr = tot_ticks + ticks, tot_pert + pert, tot_win + len(idx), tot_appr + int(appr.sum())
        win_per_ep = np.bincount(idx[:, 0], minlength=len(a["k"]))
        combos[sp] = {}
        for ci, (ins, d) in enumerate(COMBINATIONS):
            m = (a["instruction"] == INSTRUCTIONS.index(ins)) & (a["direction"] == DIRECTIONS.index(d))
            combos[sp][f"{ins}/{d}"] = {"episodes": int(m.sum()), "windows": int(win_per_ep[m].sum())}
    return {
        "horizon": DEFAULT_HORIZON, "clean_future": True,
        "splits": per_split,
        "combinations": combos,
        "any_outer_fraction": outer_fraction(SPLITS),
        "perturbed_fraction": tot_pert / tot_ticks if tot_ticks else None,
        "approaches_section_fraction": tot_appr / tot_win if tot_win else None,
    }


def _manifest(episodes: Mapping[Split, int], length: int, arrays, passages) -> dict:
    commit, dirty = _git_info()
    return {
        "format_version": FORMAT_VERSION,
        "toycar_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": _source_hashes(),
        "episode_ticks": length,
        "episodes": {sp: int(episodes[sp]) for sp in SPLITS},
        "seed_blocks": dict(SEED_BLOCKS),
        "combinations": {"list": [list(c) for c in COMBINATIONS], "rule": "k mod 6"},
        "start_rule": "start_spec(instruction, direction)",
        "expert_params": _jsonable(ExpertParams()),
        "track_params": _jsonable(TrackParams()),
        "render_defaults": _jsonable(RenderParams()),
        "colour_jitter": {
            "fields": list(COLOUR_FIELDS), "brightness_range": list(BRIGHTNESS_RANGE),
            "channel_offset": CHANNEL_OFFSET, "stream": ["seed", 3],
        },
        "summary": _summary(arrays, passages),
    }


# -- recording the dataset (SPEC §5.2) ------------------------------------------------------
def record(
    out: str | Path,
    *,
    episodes: Mapping[Split, int] = DEFAULT_EPISODES,
    length: int = EPISODE_TICKS,
    workers: int | None = None,
    force: bool = False,
    progress: bool = False,
) -> Path:
    """Record every Split into `out` (atomically, via `out.tmp`). Returns `out`."""
    out = Path(out)
    unknown = set(episodes) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}")
    counts = {sp: int(episodes.get(sp, 0)) for sp in SPLITS}
    if any(n < 0 for n in counts.values()):
        raise ValueError(f"episode counts must be >= 0, got {counts}")
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass force=True (--force) to replace it")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)

    jobs = [(sp, k, length) for sp in SPLITS for k in range(counts[sp])]
    workers = workers or os.cpu_count() or 1
    t0 = time.time()
    try:
        tmp.mkdir(parents=True)
        results: dict[Split, list[dict]] = {sp: [] for sp in SPLITS}
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 and len(jobs) > 1 else None
        try:
            it = (pool.map(_record_episode, jobs, chunksize=max(1, len(jobs) // (workers * 16)))
                  if pool else map(_record_episode, jobs))
            done = {sp: 0 for sp in SPLITS}
            for res in it:
                sp = next(s for s in SPLITS if done[s] < counts[s])
                results[sp].append(res)
                done[sp] += 1
                if progress and done[sp] == counts[sp]:
                    print(f"  {sp:5s} {counts[sp]:5d} episodes  ({time.time() - t0:.1f} s)", flush=True)
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)

        arrays = {sp: _stack(results[sp], length) for sp in SPLITS}
        passages = {sp: [r["passages"] for r in results[sp]] for sp in SPLITS}
        for sp in SPLITS:
            _save_npz(tmp / f"{sp}.npz", arrays[sp])
        manifest = _manifest(counts, length, arrays, passages)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


# -- loading (SPEC §7) ----------------------------------------------------------------------
@dataclass(frozen=True)
class Episode:
    split: Split
    index: int
    k: int
    seed: int
    instruction: Instruction
    direction: Direction
    start_route: Branch | None
    length: int
    render_params: RenderParams
    states: np.ndarray
    clean: np.ndarray
    executed: np.ndarray
    perturbed: np.ndarray
    planned_branch: np.ndarray
    station_m: np.ndarray
    branch: np.ndarray
    frames_sha256: str


def _state(row: np.ndarray) -> State:
    return State(float(row[0]), float(row[1]), float(row[2]), float(row[3]))


class EpisodeSet:
    """One Split: the stored arrays, fully in RAM."""

    def __init__(self, name: Split, arrays: dict[str, np.ndarray], track_params: TrackParams) -> None:
        self.name: Split = name
        self.arrays = arrays
        self.track_params = track_params
        self._rps = [_rp_from_array(a) for a in arrays["render_params"]]

    @property
    def track(self) -> Track:
        return _track(self.track_params)

    def __len__(self) -> int:
        return len(self.arrays["k"])

    def _check(self, i: int) -> int:
        if not isinstance(i, (int, np.integer)) or not 0 <= i < len(self):
            raise IndexError(f"episode {i} out of range for split {self.name} ({len(self)} episodes)")
        return int(i)

    def render_params(self, i: int, probe: str | None = None) -> RenderParams:
        p = _colour_probe(probe)
        rp = self._rps[self._check(i)]
        return p.apply(rp) if p else rp

    def episode(self, i: int) -> Episode:
        i = self._check(i)
        a = self.arrays
        sr = int(a["start_route"][i])
        return Episode(
            split=self.name, index=i, k=int(a["k"][i]), seed=int(a["seed"][i]),
            instruction=INSTRUCTIONS[a["instruction"][i]], direction=DIRECTIONS[a["direction"][i]],
            start_route=None if sr < 0 else _BRANCHES[sr], length=int(a["length"][i]),
            render_params=self._rps[i],
            **{n: a[n][i] for n in _PER_TICK},
            frames_sha256=str(a["frames_sha256"][i]),
        )

    def _render(self, i: int, t: int, rp: RenderParams) -> np.ndarray:
        return render(_state(self.arrays["states"][i, t]), self.track, rp)

    def frames(self, i: int, ticks: Sequence[int] | None = None, probe: str | None = None) -> np.ndarray:
        """Re-rendered Frames of episode i: (n, 64, 64, 3) uint8, default all ticks 0 … length."""
        rp = self.render_params(i, probe)
        n = int(self.arrays["length"][i]) + 1
        ticks = range(n) if ticks is None else ticks
        out = np.empty((len(ticks), 64, 64, 3), np.uint8)
        for j, t in enumerate(ticks):
            if not 0 <= t < n:
                raise IndexError(f"tick {t} out of range [0, {n})")
            out[j] = self._render(i, t, rp)
        return out

    def frames_hash(self, i: int) -> str:
        rp = self.render_params(i)
        h = hashlib.sha256()
        for t in range(int(self.arrays["length"][i]) + 1):
            h.update(self._render(i, t, rp).tobytes())
        return h.hexdigest()

    def windows(
        self,
        *,
        horizon: int = DEFAULT_HORIZON,
        future_frames: Sequence[int] | None = None,
        clean_future: bool = True,
        probe: str | None = None,
    ) -> Windows:
        T = self.arrays["perturbed"].shape[1]
        if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= T:
            raise ValueError(f"horizon must be in [1, {T}], got {horizon}")
        horizon = int(horizon)
        if future_frames is None:
            ff = tuple(range(horizon))
        else:
            ff = tuple(int(k) for k in future_frames)
            if any(not 0 <= k < horizon for k in ff):
                raise ValueError(f"future_frames entries must be in [0, {horizon - 1}], got {ff}")
            if len(set(ff)) != len(ff):
                raise ValueError(f"future_frames entries must be distinct, got {ff}")
            ff = tuple(sorted(ff))
        _colour_probe(probe)
        a = self.arrays
        index, appr, in_sec = _window_index(a["perturbed"], a["branch"], a["length"], horizon, clean_future)
        return Windows(self, horizon, ff, clean_future, probe, index, appr, in_sec)

    def eval_subset(self, n: int = 2048, **window_kwargs) -> Windows:
        """Fixed subset: n spread evenly over episodes, drawn with stream [seed, 4] (SPEC §6.4)."""
        w = self.windows(**window_kwargs)
        E = len(self)
        if E == 0 or n <= 0:
            return w.subset(np.empty(0, np.int64))
        starts = np.searchsorted(w.index[:, 0], np.arange(E + 1))
        picks = []
        for i in range(E):
            lo, hi = int(starts[i]), int(starts[i + 1])
            quota = min(n // E + (1 if i < n % E else 0), hi - lo)
            if quota == 0:
                continue
            rng = np.random.default_rng([int(self.arrays["seed"][i]), 4])
            picks.append(lo + np.sort(rng.choice(hi - lo, size=quota, replace=False)))
        return w.subset(np.concatenate(picks) if picks else np.empty(0, np.int64))


class Windows:
    """Training Windows of one Split (SPEC §6). `w[i]` renders on demand."""

    def __init__(self, es: EpisodeSet, horizon: int, future_frames: tuple[int, ...], clean_future: bool,
                 probe: str | None, index: np.ndarray, approaches_section: np.ndarray, in_section: np.ndarray) -> None:
        self.episodes = es
        self.horizon = horizon
        self.future_frames = future_frames
        self.clean_future = clean_future
        self.probe = probe
        self.index = index
        self.approaches_section = approaches_section
        self.in_section = in_section

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        if not isinstance(i, (int, np.integer)):
            raise TypeError(f"Windows indices must be integers, got {type(i).__name__}")
        N = len(self)
        if not -N <= i < N:
            raise IndexError(f"window {i} out of range ({N} windows)")
        i = int(i) % N
        e, t = int(self.index[i, 0]), int(self.index[i, 1])
        es, H = self.episodes, self.horizon
        a = es.arrays
        rp = es.render_params(e, self.probe)
        s = _state(a["states"][e, t])
        ff = self.future_frames
        fut = np.empty((len(ff), 64, 64, 3), np.uint8)
        for j, k in enumerate(ff):
            fut[j] = es._render(e, t + 1 + k, rp)
        return {
            "obs_frame": render(s, es.track, rp),
            "state": s.features(),
            "state_raw": a["states"][e, t].copy(),
            "actions": a["clean"][e, t : t + H].copy(),
            "executed": a["executed"][e, t : t + H].copy(),
            "perturbed": a["perturbed"][e, t : t + H].copy(),
            "future_frames": fut,
            "future_index": np.array(ff, np.int64),
            "instruction": np.int64(a["instruction"][e]),
            "direction": np.int64(a["direction"][e]),
            "episode": np.int64(e),
            "t": np.int64(t),
            "seed": np.int64(a["seed"][e]),
            "approaches_section": np.bool_(self.approaches_section[i]),
            "in_section": np.bool_(self.in_section[i]),
        }

    def subset(self, mask_or_indices) -> Windows:
        sel = np.asarray(mask_or_indices)
        if sel.dtype == bool:
            if sel.shape != (len(self),):
                raise ValueError(f"mask must have shape ({len(self)},), got {sel.shape}")
            sel = np.nonzero(sel)[0]
        else:
            sel = sel.astype(np.int64)
            if sel.size and (sel.min() < 0 or sel.max() >= len(self)):
                raise IndexError("subset indices out of range")
        return Windows(self.episodes, self.horizon, self.future_frames, self.clean_future, self.probe,
                       self.index[sel], self.approaches_section[sel], self.in_section[sel])


class FrameItems:
    """Every Frame of every episode as one item, ordered by episode then tick (Step 4)."""

    def __init__(self, es: EpisodeSet, probe: str | None = None) -> None:
        _colour_probe(probe)
        self.episodes = es
        self.probe = probe
        self._offsets = np.concatenate([[0], np.cumsum(es.arrays["length"].astype(np.int64) + 1)])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, i: int) -> np.ndarray:
        N = len(self)
        if not isinstance(i, (int, np.integer)) or not -N <= i < N:
            raise IndexError(f"frame {i} out of range ({N} frames)")
        i = int(i) % N
        e = int(np.searchsorted(self._offsets, i, side="right")) - 1
        t = i - int(self._offsets[e])
        return self.episodes._render(e, t, self.episodes.render_params(e, self.probe))


def frame_dataset(es: EpisodeSet, probe: str | None = None) -> FrameItems:
    return FrameItems(es, probe)


class Dataset:
    def __init__(self, path: Path, manifest: dict, splits: dict[Split, EpisodeSet]) -> None:
        self.path = path
        self.manifest = manifest
        self._splits = splits

    def split(self, name: Split) -> EpisodeSet:
        if name not in self._splits:
            raise ValueError(f"unknown split {name!r}; known: {list(SPLITS)}")
        return self._splits[name]

    def _check_frames(self, sp: Split, episodes: Sequence[int]) -> None:
        es = self._splits[sp]
        for i in episodes:
            if es.frames_hash(i) != es.arrays["frames_sha256"][i]:
                raise ValueError(
                    f"{self.path}: Frame hash mismatch in split {sp}, episode {i}, seed {int(es.arrays['seed'][i])}; "
                    "re-record the dataset or check out the code version it was recorded with"
                )

    def verify(self) -> None:
        """Re-render every episode of every Split, and re-simulate one per Split; raises on mismatch."""
        for sid, sp in enumerate(SPLITS):
            es = self._splits[sp]
            self._check_frames(sp, range(len(es)))
            if len(es) == 0:
                continue
            i = int(np.random.default_rng([1, sid]).integers(len(es)))
            a = es.arrays
            res = _record_episode((sp, int(a["k"][i]), int(a["length"][i])))
            for n in _PER_TICK:
                if not np.array_equal(res[n], a[n][i]):
                    raise ValueError(f"{self.path}: re-simulated {n} differ in split {sp}, episode {i}, seed {int(a['seed'][i])}")


def open_dataset(path: str | Path, *, check: bool = True) -> Dataset:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format_version {manifest.get('format_version')!r} (expected {FORMAT_VERSION})")
    tp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in manifest["track_params"].items()}
    track_params = TrackParams(**tp)
    splits: dict[Split, EpisodeSet] = {}
    for sp in SPLITS:
        with np.load(path / f"{sp}.npz", allow_pickle=False) as z:
            arrays = {n: z[n] for n in z.files}
        splits[sp] = EpisodeSet(sp, arrays, track_params)
    ds = Dataset(path, manifest, splits)

    current = _source_hashes()
    changed = [n for n in SOURCE_FILES if manifest.get("source_sha256", {}).get(n) != current[n]]
    if changed:
        warnings.warn(f"{path}: source files changed since recording: {', '.join(changed)} (the Frame check is authoritative)",
                      stacklevel=2)
    if check:
        for sid, sp in enumerate(SPLITS):
            E = len(splits[sp])
            pick = np.random.default_rng([0, sid]).choice(E, size=min(OPEN_CHECK_EPISODES, E), replace=False)
            ds._check_frames(sp, [int(i) for i in pick])
    return ds


# -- CLI (SPEC §9) --------------------------------------------------------------------------
def _parse_episodes(text: str) -> dict[Split, int]:
    out: dict[Split, int] = {}
    for item in text.split(","):
        key, sep, value = item.partition("=")
        if not sep or key not in SPLITS:
            raise SystemExit(f"--episodes expects train=N,val=N,test=N, got {text!r}")
        out[cast(Split, key)] = int(value)
    return out


def cmd_record(out: Path, workers: int | None, force: bool, episodes: Mapping[Split, int], length: int) -> int:
    total = sum(episodes.values())
    print(f"recording {total} episodes × {length} ticks into {out}", flush=True)
    t0 = time.time()
    try:
        record(out, episodes=episodes, length=length, workers=workers, force=force, progress=True)
    except (FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    m = json.loads((out / "manifest.json").read_text())["summary"]
    wins = sum(s["windows"] for s in m["splits"].values())
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"{total} episodes, {total * length} ticks, {wins} usable Windows (H={DEFAULT_HORIZON}), "
          f"{size / 1e6:.1f} MB in {time.time() - t0:.1f} s")
    print(out)
    return 0


def cmd_verify(path: Path) -> int:
    try:
        ds = open_dataset(path, check=False)
        ds.verify()
    except (ValueError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS")
    return 0


def _pct(x: float | None) -> str:
    return "   -  " if x is None else f"{100 * x:5.1f}%"


def cmd_stats(path: Path) -> int:
    ds = open_dataset(path, check=False)
    summary = ds.manifest["summary"]
    print(f"{path}  (T = {ds.manifest['episode_ticks']}, H = {DEFAULT_HORIZON}, clean-future filter on)\n")
    print(f"{'split':6s} {'instr':6s} {'dir':4s} {'episodes':>9s} {'windows':>9s}")
    tot_e = tot_w = 0
    speeds = []
    for sp in SPLITS:
        es = ds.split(sp)
        a = es.arrays
        w = es.windows()
        per_ep = np.bincount(w.index[:, 0], minlength=len(es))
        for ins, d in COMBINATIONS:
            m = (a["instruction"] == INSTRUCTIONS.index(ins)) & (a["direction"] == DIRECTIONS.index(d))
            print(f"{sp:6s} {ins:6s} {d:4s} {int(m.sum()):9d} {int(per_ep[m].sum()):9d}")
        print(f"{sp:6s} {'all':6s} {'':4s} {len(es):9d} {len(w):9d}   perturbed {_pct(a['perturbed'].mean() if len(es) else None)}"
              f"  approaches_section {_pct(w.approaches_section.mean() if len(w) else None)}"
              f"  any outer {_pct(summary['splits'][sp]['any_outer_fraction'])}\n")
        tot_e, tot_w = tot_e + len(es), tot_w + len(w)
        speeds.append(a["states"][:, :, 3].ravel())
    print(f"total  {tot_e} episodes, {tot_w} usable Windows")
    print(f"any outer fraction {_pct(summary['any_outer_fraction'])}, perturbed {_pct(summary['perturbed_fraction'])}, "
          f"approaches_section {_pct(summary['approaches_section_fraction'])}\n")
    v = np.concatenate(speeds)
    hist, edges = np.histogram(v, bins=np.arange(11))
    print("speed histogram (all ticks)")
    peak = max(int(hist.max()), 1)
    for c, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"  {lo:2.0f}–{hi:<2.0f} m/s {int(c):9d} {100 * c / max(len(v), 1):5.1f}%  {'#' * round(40 * c / peak)}")
    return 0


def _zoom_box(xy: np.ndarray, zoom_m: float, resolution: int = 64) -> tuple[int, int, int, int]:
    """Pixel crop (r0, r1, c0, c1) of a square at least `zoom_m` wide, centred on the car path `xy` (n, 2)
    and grown to keep the whole path plus one car length in view; clamped to the Frame."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    side_m = max(zoom_m, float((hi - lo).max()) + 2 * CAR_LENGTH)
    side = min(resolution, math.ceil(side_m * resolution / WORLD_M))
    cx, cy = (lo + hi) / 2 * resolution / WORLD_M  # pixel coordinates: column = x, row = y
    c0 = min(max(round(cx - side / 2), 0), resolution - side)
    r0 = min(max(round(cy - side / 2), 0), resolution - side)
    return r0, r0 + side, c0, c0 + side


def cmd_plot(path: Path, split: Split, episode: int | None, t: int | None, seed: int, approaching: bool,
             probe: str | None, out: Path, zoom: float | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = open_dataset(path, check=False)
    es = ds.split(split)
    H = DEFAULT_HORIZON
    if episode is not None:
        w = es.windows(horizon=H, clean_future=False, probe=probe)
        rows = np.nonzero((w.index[:, 0] == episode) & (w.index[:, 1] == t))[0]
        if len(rows) == 0:
            T = int(es.arrays["length"][episode]) if 0 <= episode < len(es) else None
            raise SystemExit(f"no Window at episode {episode}, t {t} (episodes: {len(es)}, t in [0, {T and T - H}])")
        row = int(rows[0])
    else:
        w = es.windows(horizon=H, probe=probe)
        if approaching:
            w = w.subset(w.approaches_section)
        if len(w) == 0:
            raise SystemExit("no matching Windows")
        row = int(np.random.default_rng(seed).integers(len(w)))
    item = w[row]
    e, t0 = int(item["episode"]), int(item["t"])
    states = es.arrays["states"][e]
    steps = np.arange(H)

    fig = plt.figure(figsize=(1.1 * (H + 1), 6.4))
    gs = fig.add_gridspec(4, H + 1, height_ratios=[1.15, 1, 1, 1], wspace=0.0, hspace=0.25,
                          left=0.04, right=0.99, top=0.86, bottom=0.07)
    imgs = [("t", item["obs_frame"])] + [(f"k={k} (t+{k + 1})", f) for k, f in zip(item["future_index"], item["future_frames"])]
    if zoom is not None:
        # the same crop in every column, so the car moves across a fixed piece of track
        r0, r1, c0, c1 = _zoom_box(states[t0 : t0 + H + 1, :2], zoom)
        imgs = [(label, img[r0:r1, c0:c1]) for label, img in imgs]
    for c, (label, img) in enumerate(imgs):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=7)
    pert = item["perturbed"]
    series = [
        ("steer", item["actions"][:, 0], item["executed"][:, 0], (-1.05, 1.05)),
        ("throttle", item["actions"][:, 1], item["executed"][:, 1], (-1.05, 1.05)),
    ]
    xlim = (-1.5, H - 0.5)  # column 0 (the observed Frame) sits at x = −1, future Frame k at x = k
    for r, (name, cl, ex, ylim) in enumerate(series, start=1):
        ax = fig.add_subplot(gs[r, :])
        for k in steps[pert]:
            ax.axvspan(k - 0.5, k + 0.5, color="orange", alpha=0.25, lw=0)
        ax.plot(steps, cl, "o-", color="C0", label="clean", ms=3)
        ax.plot(steps, ex, "s--", color="C3", label="executed", ms=3)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(name)
        ax.set_xticks(np.arange(-1, H))
        ax.set_xticklabels([])
        ax.grid(axis="x", color="0.9")
        if r == 1:
            ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax = fig.add_subplot(gs[3, :])
    ax.plot(np.arange(-1, H), states[t0 : t0 + H + 1, 3], "o-", color="C2", ms=3)
    ax.set_xlim(*xlim)
    ax.set_ylabel("v (m/s)")
    ax.set_xticks(np.arange(-1, H))
    ax.set_xticklabels(["t"] + [str(k) for k in steps])
    ax.set_xlabel("k  (action k is applied at tick t+k and causes the Frame above it; v from states[t+1+k])")
    ax.grid(axis="x", color="0.9")
    fig.suptitle(
        f"{path.name} · {split} · episode {e} · seed {int(item['seed'])} · t {t0} · "
        f"{INSTRUCTIONS[int(item['instruction'])]} {DIRECTIONS[int(item['direction'])]} · "
        f"approaches_section {bool(item['approaches_section'])} · probe {probe or '-'}"
        + (f" · zoom {(r1 - r0) * WORLD_M / 64:.1f} m" if zoom is not None else ""),
        fontsize=10,
    )
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m toycar.dataset", description="Record expert episodes and inspect Windows.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="record the dataset")
    rec.add_argument("--out", type=Path, default=DEFAULT_DIR)
    rec.add_argument("--workers", type=int)
    rec.add_argument("--force", action="store_true")
    rec.add_argument("--episodes", type=_parse_episodes, default=None, help=argparse.SUPPRESS)
    rec.add_argument("--length", type=int, default=EPISODE_TICKS, help=argparse.SUPPRESS)
    ver = sub.add_parser("verify", help="re-render and re-simulate; PASS or the mismatch")
    ver.add_argument("dir", nargs="?", type=Path, default=DEFAULT_DIR)
    st = sub.add_parser("stats", help="counts, shares and speed histogram")
    st.add_argument("dir", nargs="?", type=Path, default=DEFAULT_DIR)
    pl = sub.add_parser("plot", help="PNG of one Window (needs --group viz)")
    pl.add_argument("dir", nargs="?", type=Path, default=DEFAULT_DIR)
    pl.add_argument("--split", choices=SPLITS, default="val")
    pl.add_argument("--episode", type=int)
    pl.add_argument("--t", type=int)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--approaching", action="store_true")
    pl.add_argument("--probe", choices=[n for n, p in PROBES.items() if p.track_params is None])
    pl.add_argument("--out", type=Path, default=Path("window.png"))
    pl.add_argument("--zoom", type=float, metavar="M",
                    help="crop every Frame to the same square of at least M metres around the car's path")
    args = ap.parse_args(argv)

    if args.cmd == "record":
        episodes = DEFAULT_EPISODES if args.episodes is None else {sp: args.episodes.get(sp, 0) for sp in SPLITS}
        return cmd_record(args.out, args.workers, args.force, episodes, args.length)
    if args.cmd == "verify":
        return cmd_verify(args.dir)
    if args.cmd == "stats":
        return cmd_stats(args.dir)
    if args.zoom is not None and not 0 < args.zoom <= WORLD_M:
        ap.error(f"--zoom must be in (0, {WORLD_M:g}] metres")
    if (args.episode is None) != (args.t is None):
        ap.error("--episode and --t go together")
    return cmd_plot(args.dir, args.split, args.episode, args.t, args.seed, args.approaching, args.probe, args.out, args.zoom)


if __name__ == "__main__":
    sys.exit(main())
