"""The generic closed loop (Step 9 SPEC §4, §8.1): Drivers, the replanning Scheme and `run_episodes`.

A Driver is anything that, once per Tick, turns the running episodes' Frames, States and Infos into one Action
each: a `wam.policy` Policy wrapped by a Scheme (`PolicyDriver`), the clean scripted Expert (`ExpertDriver`) or a
Saboteur (the clean expert with one defect). `run_episodes` plays a list of `EpisodeSpec`s in lockstep, applies the
Stress condition of each spec and returns one `EpisodeRecord` per episode.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Protocol

import numpy as np

from toycar.dataset import PROBES, sample_render_params
from toycar.env import CarEnv
from toycar.expert import EpisodeScore, Expert, ExpertParams, Instruction, score_episode, start_spec
from toycar.render import RenderParams
from toycar.state import Direction, Info, StartSpec, State
from toycar.track import Track

# -- Stress conditions (SPEC §8.1) ----------------------------------------------------------
COLOUR_CONDITIONS = ("brown_grass", "light_road", "blue_car", "no_kerb")
CONDITIONS = ("none", *COLOUR_CONDITIONS, "wide_fork", "offset_start", "push", "push_strong")
PUSH_FIRST, PUSH_EVERY, PUSH_TICKS = 60, 60, 8
PUSH_AMP = {"push": 0.4, "push_strong": 0.5}
OFFSET_LATERAL_M = (1.0, 1.5)
OFFSET_HEADING_DEG = (10.0, 20.0)
OFFSET_SPEED = (0.0, 6.0)
OFFSET_TRIES = 50
MAX_STEPS = 900
BRANCH_ID = {None: -1, "outer": 0, "inner": 1}


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    instruction: Instruction
    direction: Direction
    condition: str = "none"


@lru_cache(maxsize=4)
def condition_track(condition: str) -> Track:
    """The Track an episode under `condition` drives on (shared, read-only)."""
    if condition == "wide_fork":
        return Track(PROBES["wide_fork"].track_params)
    return Track.default()


def condition_render_params(spec: EpisodeSpec) -> RenderParams:
    rp = sample_render_params(spec.seed)
    return PROBES[spec.condition].apply(rp) if spec.condition in COLOUR_CONDITIONS else rp


def _check_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown Stress condition {condition!r}; known: {list(CONDITIONS)}")


def reset_env(spec: EpisodeSpec, max_steps: int = MAX_STEPS) -> tuple[CarEnv, State, Info]:
    """A fresh env for `spec`, reset with its Track, colours and start rule (SPEC §3.2, §8.1)."""
    _check_condition(spec.condition)
    env = CarEnv(track=condition_track(spec.condition), max_steps=max_steps)
    rp = condition_render_params(spec)
    base = start_spec(spec.instruction, spec.direction)
    if spec.condition != "offset_start":
        state, info = env.reset(spec.seed, base, render_params=rp)
        return env, state, info
    rng = np.random.default_rng([spec.seed, 5])
    for _ in range(OFFSET_TRIES):
        route = base.route if base.route is not None else ("outer", "inner")[int(rng.integers(2))]
        station = float(rng.uniform(0.0, env.track.length_m))
        lateral = float(rng.choice([-1, 1]) * rng.uniform(*OFFSET_LATERAL_M))
        heading = float(rng.choice([-1, 1]) * math.radians(rng.uniform(*OFFSET_HEADING_DEG)))
        speed = float(rng.uniform(*OFFSET_SPEED))
        sp = StartSpec(direction=spec.direction, route=route, station_m=station, lateral_m=lateral,
                       heading_offset=heading, speed=speed)
        try:
            state, info = env.reset(spec.seed, sp, render_params=rp)
        except ValueError:
            continue
        return env, state, info
    raise RuntimeError(f"no valid offset start for seed {spec.seed} after {OFFSET_TRIES} tries")


class Push:
    """Additive steer offsets on the executed Action (SPEC §8.1); a no-op for other conditions."""

    def __init__(self, spec: EpisodeSpec) -> None:
        self.amp = PUSH_AMP.get(spec.condition, 0.0)
        self._rng = np.random.default_rng([spec.seed, 4])
        self._sign = 0.0
        self.starts: list[int] = []

    def active(self, tick: int) -> bool:
        return self.amp > 0 and tick >= PUSH_FIRST and (tick - PUSH_FIRST) % PUSH_EVERY < PUSH_TICKS

    def __call__(self, tick: int, action: np.ndarray) -> np.ndarray:
        if not self.active(tick):
            return action
        if (tick - PUSH_FIRST) % PUSH_EVERY == 0:
            self._sign = float(self._rng.choice([-1, 1]))
            self.starts.append(tick)
        out = np.array(action, dtype=np.float64)
        out[0] = min(max(out[0] + self._sign * self.amp, -1.0), 1.0)
        return out


# -- Driver protocol (SPEC §4.1) ------------------------------------------------------------
class Driver(Protocol):
    name: str
    needs_frames: bool

    def reset(self, specs: list[EpisodeSpec], envs: list[CarEnv], states: list[State], infos: list[Info]) -> None: ...

    def act(self, rows: list[int], frames: np.ndarray | None, states: list[State], infos: list[Info]) -> np.ndarray:
        """rows: indices into the reset() lists of the running episodes, increasing; states/infos aligned with
        rows. Returns float (len(rows), 2), one Action per running episode."""
        ...


# -- the expert and the Saboteurs (SPEC §4.3) -----------------------------------------------
CLEAN = ExpertParams(burst_peak=0.0)
SABOTEURS = ("zero-steer", "averaged-fork", "slow", "wrong-branch", "always-outer")
OPPOSITE: dict[str, Instruction] = {"outer": "inner", "inner": "outer", "any": "any"}


class ExpertDriver:
    """The scripted Expert as a Driver; the clean one (`burst_peak = 0`) is the reference row."""

    needs_frames = False

    def __init__(self, params: ExpertParams = CLEAN, name: str = "expert") -> None:
        self.params = params
        self.name = name
        self.last_diag: dict[str, np.ndarray] = {}

    def _told(self, spec: EpisodeSpec) -> Instruction:
        return spec.instruction

    def reset(self, specs, envs, states, infos) -> None:
        self.specs = list(specs)
        self.experts = []
        for sp, env, s, i in zip(specs, envs, states, infos):
            ex = Expert(env.track, self._told(sp), self.params)
            ex.reset(sp.seed, s, i)
            self.experts.append(ex)

    def _override(self, j: int, state: State, info: Info, a: tuple[float, float]) -> tuple[float, float]:
        return a

    def act(self, rows, frames, states, infos) -> np.ndarray:
        out = np.empty((len(rows), 2))
        for r, (j, s, i) in enumerate(zip(rows, states, infos)):
            a = self.experts[j].act(s, i).executed  # every Tick: advances the coin and plan state
            out[r] = self._override(j, s, i, a)
        return out


class SaboteurDriver(ExpertDriver):
    """The clean expert with one defect; each must fail its gate(s) (SPEC §4.3, §6)."""

    def __init__(self, kind: str) -> None:
        if kind not in SABOTEURS:
            raise ValueError(f"unknown Saboteur {kind!r}; known: {list(SABOTEURS)}")
        params = {"slow": replace(CLEAN, v_cap=5.0), "always-outer": replace(CLEAN, p_outer=1.0)}.get(kind, CLEAN)
        super().__init__(params, name=f"saboteur:{kind}")
        self.kind = kind

    def _told(self, spec: EpisodeSpec) -> Instruction:
        return OPPOSITE[spec.instruction] if self.kind == "wrong-branch" else spec.instruction

    def _override(self, j, state, info, a):
        if self.kind == "zero-steer":
            return 0.0, a[1]
        if self.kind == "averaged-fork" and self.specs[j].instruction == "any":
            ex, d = self.experts[j], info.direction
            o, n = ex.control(state, "outer", d), ex.control(state, "inner", d)
            return (o[0] + n[0]) / 2, (o[1] + n[1]) / 2
        return a


# -- the replanning Scheme and PolicyDriver (SPEC §4.2) -------------------------------------
@dataclass(frozen=True)
class Scheme:
    execute_k: int = 1  # actions executed per replan, 1 ≤ execute_k ≤ H
    n_samples: int = 4
    select: str = "coherence"  # "coherence" | "first"
    rho: float = 0.9
    n_steps: int = 4
    cfg_action: float = 1.0
    cfg_video: float = 1.0
    seed_mode: str = "per_replan"  # "per_replan" | "per_episode"

    def __post_init__(self) -> None:
        if self.execute_k < 1:
            raise ValueError(f"execute_k must be ≥ 1, got {self.execute_k}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {self.n_samples}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be ≥ 1, got {self.n_steps}")
        if self.select not in ("coherence", "first"):
            raise ValueError(f"select must be 'coherence' or 'first', got {self.select!r}")
        if self.seed_mode not in ("per_replan", "per_episode"):
            raise ValueError(f"seed_mode must be 'per_replan' or 'per_episode', got {self.seed_mode!r}")


DEFAULT_SCHEME = Scheme()


def noise_seed(seed: int, sample: int, tick: int, mode: str) -> int:
    """The CPU generator seed of sample `sample` of an episode's replan at `tick` (SPEC §4.2 step 2)."""
    g = seed * 64 + sample
    return g * 100_003 + tick if mode == "per_replan" else g


def select_chunk(cands: np.ndarray, prev: np.ndarray | None, execute_k: int, select: str, rho: float) -> int:
    """Index of the candidate to execute (SPEC §4.2 step 4). cands (N, H, 2); prev the previously selected Chunk."""
    H = cands.shape[1]
    ov = H - execute_k
    if select == "first" or prev is None or ov <= 0 or len(cands) == 1:
        return 0
    w = rho ** np.arange(ov)
    cost = (np.linalg.norm(cands[:, :ov] - prev[None, execute_k:execute_k + ov], axis=-1) * w).sum(-1)
    return int(np.argmin(cost))  # first minimum → lowest index on ties


class PolicyDriver:
    """A Policy executing `execute_k` actions of a selected Chunk per replan; never averages Chunks."""

    needs_frames = True

    def __init__(self, policy, scheme: Scheme = DEFAULT_SCHEME, *, batch: int = 24, name: str | None = None,
                 keep_candidates: bool = False, imagine: bool = False) -> None:
        self.policy = policy
        self.horizon = int(policy.horizon)
        if scheme.execute_k > self.horizon:
            raise ValueError(f"execute_k {scheme.execute_k} exceeds the Policy's horizon {self.horizon}")
        self.regression = policy.config.objective == "regression"
        if self.regression:
            scheme = replace(scheme, n_samples=1, cfg_action=1.0, cfg_video=1.0)
        self.scheme = scheme
        self.batch = batch
        self.name = name or policy.name or (policy.source.name if policy.source is not None else "policy")
        self.keep_candidates = keep_candidates
        self.imagine = imagine
        self.last_diag: dict[str, np.ndarray] = {}
        self.last_candidates: dict[int, np.ndarray] = {}
        self.last_latents: dict[int, object] = {}

    def reset(self, specs, envs, states, infos) -> None:
        n = len(specs)
        self.specs = list(specs)
        self.queue: list[list[np.ndarray]] = [[] for _ in range(n)]
        self.prev: list[np.ndarray | None] = [None] * n

    def _noise(self, seeds: Sequence[int], tick: int):
        import torch

        from . import flow

        sc, xs_z, xs_a = self.scheme, [], []
        for s in seeds:
            for i in range(sc.n_samples):
                g = torch.Generator().manual_seed(noise_seed(int(s), i, tick, sc.seed_mode))
                z, a = flow.initial_noise(self.policy.config, 1, g)
                xs_z.append(z)
                xs_a.append(a)
        return (None if xs_z[0] is None else torch.cat(xs_z)), torch.cat(xs_a)

    def _sample(self, frames: np.ndarray, feats: np.ndarray, instr: list[str], seeds: list[int], tick: int):
        sc = self.scheme
        kw = dict(n_samples=sc.n_samples, n_steps=sc.n_steps, cfg_action=sc.cfg_action, cfg_video=sc.cfg_video,
                  return_latents=self.imagine)
        if self.regression:
            return self.policy.act(frames, feats, instr, seed=0, **kw)
        return self.policy.act(frames, feats, instr, noise=self._noise(seeds, tick), **kw)

    def act(self, rows, frames, states, infos) -> np.ndarray:
        sc, H, n = self.scheme, self.horizon, len(rows)
        replanned = np.zeros(n, bool)
        selected = np.full(n, -1, np.int64)
        chunk_steer = np.full((n, H), np.nan)
        sample_ms = np.zeros(n)
        need = [r for r, j in enumerate(rows) if not self.queue[j]]
        self.last_candidates, self.last_latents = {}, {}
        for b0 in range(0, len(need), max(1, self.batch)):
            sub = need[b0:b0 + max(1, self.batch)]
            js = [rows[r] for r in sub]
            tick = int(infos[sub[0]].steps)
            t0 = time.perf_counter()
            res = self._sample(frames[sub], np.stack([states[r].features() for r in sub]),
                               [self.specs[j].instruction for j in js], [self.specs[j].seed for j in js], tick)
            ms = 1000 * (time.perf_counter() - t0) / len(sub)
            for row, (r, j) in enumerate(zip(sub, js)):
                cands = res.actions[row]
                pick = select_chunk(cands, self.prev[j], sc.execute_k, sc.select, sc.rho)
                ch = cands[pick]
                self.prev[j] = ch
                self.queue[j] = [ch[t] for t in range(sc.execute_k)]
                replanned[r], selected[r], chunk_steer[r], sample_ms[r] = True, pick, ch[:, 0], ms
                if self.keep_candidates:
                    self.last_candidates[j] = cands
                if self.imagine and res.latents is not None:
                    self.last_latents[j] = res.latents[row, pick]
        out = np.stack([self.queue[j].pop(0) for j in rows]).astype(np.float64)
        self.last_diag = {"replanned": replanned, "selected": selected, "chunk_steer": chunk_steer,
                          "sample_ms": sample_ms}
        return out


# -- episodes (SPEC §4.4) -------------------------------------------------------------------
@dataclass
class EpisodeRecord:
    spec: EpisodeSpec
    score: EpisodeScore
    fractional_laps: float
    crash_tick: int | None
    crash_station_m: float | None
    crash_in_section: bool | None
    push_ticks: list[int]
    x: np.ndarray  # (steps + 1,) State after each step, start State at index 0
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    branch: np.ndarray  # (steps + 1,) int8: −1 / 0 outer / 1 inner
    station_m: np.ndarray  # (steps + 1,)
    action_driver: np.ndarray  # (steps, 2)
    action_exec: np.ndarray  # (steps, 2), after a Push
    dist_m: np.ndarray  # (steps,) distance to the centreline after the step
    expert_steer: np.ndarray  # (steps,) shadow expert's clean steer in the visited State (NaN without one)
    selected: np.ndarray  # (steps,) sample index of a replan, −1 when none
    dist0_m: float  # distance to the centreline of the start State
    replans: list[tuple[int, np.ndarray]] = field(default_factory=list)  # (tick, selected Chunk's steer (H,))
    wall_s: float = 0.0
    ms_per_replan: float | None = None

    @property
    def steps(self) -> int:
        return len(self.dist_m)

    @property
    def dist_state(self) -> np.ndarray:
        """(steps + 1,) distance to the centreline of the State at each Tick."""
        return np.concatenate([[self.dist0_m], self.dist_m])


def run_episodes(driver: Driver, specs: Sequence[EpisodeSpec], *, batch: int = 24, shadow_expert: bool = True,
                 progress: Callable[[int, int], None] | None = None, max_steps: int = MAX_STEPS) -> list[EpisodeRecord]:
    """Play `specs` in lockstep with `driver`; one record per spec, in order (SPEC §4.4)."""
    specs = list(specs)
    n = len(specs)
    if n == 0:
        return []
    if hasattr(driver, "batch"):
        driver.batch = batch
    envs, states, infos, pushes, shadows = [], [], [], [], []
    for sp in specs:
        env, s, i = reset_env(sp, max_steps)
        envs.append(env), states.append(s), infos.append([i]), pushes.append(Push(sp))
        if shadow_expert:
            ex = Expert(env.track, sp.instruction, CLEAN)
            ex.reset(sp.seed, s, i)
            shadows.append(ex)
    driver.reset(specs, envs, states, [i[-1] for i in infos])
    logs = [{"st": [states[j]], "ad": [], "ae": [], "dist": [], "ex": [], "sel": [], "replans": [], "ms": []}
            for j in range(n)]
    walls = [0.0] * n
    alive = list(range(n))
    done = 0
    while alive:
        t0 = time.perf_counter()
        frames = np.stack([envs[j].render() for j in alive]) if driver.needs_frames else None
        acts = np.asarray(driver.act(list(alive), frames, [states[j] for j in alive], [infos[j][-1] for j in alive]),
                          dtype=np.float64)
        if acts.shape != (len(alive), 2):
            raise ValueError(f"{driver.name}: act returned shape {acts.shape}, want ({len(alive)}, 2)")
        diag = getattr(driver, "last_diag", None) or {}
        finished = []
        for r, j in enumerate(alive):
            s, info = states[j], infos[j][-1]
            tick = info.steps
            lg = logs[j]
            lg["ex"].append(shadows[j].act(s, info).clean[0] if shadow_expert else math.nan)
            a_drv = acts[r]
            a_exe = pushes[j](tick, a_drv)
            res = envs[j].step((float(a_exe[0]), float(a_exe[1])))
            states[j] = res.state
            infos[j].append(res.info)
            lg["st"].append(res.state)
            lg["ad"].append(a_drv)
            lg["ae"].append(a_exe)
            lg["dist"].append(envs[j].track.distance_to_centerline(res.state.x, res.state.y))
            if "replanned" in diag and diag["replanned"][r]:
                lg["sel"].append(int(diag["selected"][r]))
                lg["replans"].append((tick, np.array(diag["chunk_steer"][r])))
                lg["ms"].append(float(diag["sample_ms"][r]))
            else:
                lg["sel"].append(-1)
            if res.terminated or res.truncated:
                finished.append(j)
        dt = time.perf_counter() - t0
        for j in alive:
            walls[j] += dt / len(alive)
        for j in finished:
            alive.remove(j)
            done += 1
            if progress is not None:
                progress(done, n)
    return [_record(specs[j], envs[j], infos[j], logs[j], pushes[j], walls[j]) for j in range(n)]


def _record(spec: EpisodeSpec, env: CarEnv, infos: list[Info], lg: dict, push: Push, wall: float) -> EpisodeRecord:
    track = env.track
    score = score_episode(track, spec.instruction, infos)
    last = infos[-1]
    crashed = score.crashed
    st = lg["st"]
    s0 = st[0]
    return EpisodeRecord(
        spec=spec, score=score, fractional_laps=last.progress_m / track.length_m,
        crash_tick=last.steps if crashed else None, crash_station_m=last.station_m if crashed else None,
        crash_in_section=(last.branch is not None) if crashed else None, push_ticks=list(push.starts),
        x=np.array([s.x for s in st]), y=np.array([s.y for s in st]), theta=np.array([s.theta for s in st]),
        v=np.array([s.v for s in st]), branch=np.array([BRANCH_ID[i.branch] for i in infos], np.int8),
        station_m=np.array([i.station_m for i in infos]),
        action_driver=np.array(lg["ad"]).reshape(-1, 2), action_exec=np.array(lg["ae"]).reshape(-1, 2),
        dist_m=np.array(lg["dist"]), expert_steer=np.array(lg["ex"]), selected=np.array(lg["sel"], np.int64),
        dist0_m=track.distance_to_centerline(s0.x, s0.y), replans=lg["replans"], wall_s=wall,
        ms_per_replan=float(np.mean(lg["ms"])) if lg["ms"] else None,
    )
