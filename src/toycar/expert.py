"""Scripted expert, Instructions and policy-agnostic episode scoring (Step 2 SPEC).

    uv run python -m toycar.expert show [--seed N] [--instruction outer|inner|any] [--direction cw|ccw]
                                        [--clean] [--param KEY=VALUE ...]
    uv run python -m toycar.expert eval [--n 1000] [--instruction ...] [--direction ...]
                                        [--clean] [--param KEY=VALUE ...] [--workers K]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from .env import CarEnv
from .physics import A_LAT_MAX, ACCEL, BRAKE, DELTA_MAX, DT, V_MAX, WHEELBASE, step_state, wrap
from .state import Branch, Direction, Info, StartSpec, State, Termination
from .track import Track, TrackParams

# -- Instructions (SPEC §2) -----------------------------------------------------------------
Instruction = Literal["outer", "inner", "any"]
INSTRUCTIONS: tuple[Instruction, ...] = ("outer", "inner", "any")  # index = integer id; append only

# -- constants (SPEC §3) --------------------------------------------------------------------
WS = 0.5  # 1/m, steering bandwidth: ke = WS², kθ = 2·WS (critically damped)
A_ACC_PLAN = 4.0  # m/s², profile forward pass (= ACCEL)
A_BRK_PLAN = 6.0  # m/s², profile backward pass (2 m/s² below BRAKE)
PROFILE_DS = 0.05  # m, profile grid spacing
SPEED_LOOKAHEAD_M = 0.5  # m, profile look-ahead beyond one tick
ROLLOUT_S = 3.0  # s, start-branch feasibility rollout
WRONG_BRANCH_M = 2.5  # m, off-Route threshold for the rollout and the safety rule
GRACE_S = 2.0  # s, no noise bursts after reset

_TAN_MAX = math.tan(DELTA_MAX)
_ROLLOUT_TICKS = round(ROLLOUT_S / DT)
_GRACE_TICKS = round(GRACE_S / DT)


@dataclass(frozen=True)
class ExpertParams:
    a_lat_plan: float = 5.0  # m/s², lateral accel used to plan curve speeds
    v_cap: float = 10.0  # m/s, speed-profile cap
    burst_peak: float = 0.4  # max steer amplitude of a noise burst; 0 = clean
    burst_gap_s: float = 3.0  # s, mean of the exponential gap between bursts
    p_outer: float = 0.5  # P(outer) per coin under "any"


@dataclass(frozen=True)
class ExpertAction:
    clean: tuple[float, float]  # expert label: control(state, planned_branch, direction)
    executed: tuple[float, float]  # clean + noise, clipped; pass this to env.step
    perturbed: bool  # True iff a burst is active on this tick


def _other(b: Branch) -> Branch:
    return "inner" if b == "outer" else "outer"


# -- speed profile (SPEC §4.1) --------------------------------------------------------------
_PROFILES: dict[tuple[TrackParams, Branch, int, float, float], tuple[list[float], float]] = {}


def _profile(track: Track, branch: Branch, d: int, a_lat_plan: float, v_cap: float) -> tuple[list[float], float]:
    """(speeds on the travel-order grid, route length). Cached; a pure function of the key."""
    key = (track.params, branch, d, a_lat_plan, v_cap)
    hit = _PROFILES.get(key)
    if hit is not None:
        return hit
    route = track.route(branch)
    length = route.length_m
    ds = PROFILE_DS
    n = math.ceil(length / ds)
    v = [min(v_cap, math.sqrt(a_lat_plan / max(abs(route.curvature_at((d * i * ds) % length)), 1e-9))) for i in range(n)]
    acc2, brk2 = 2.0 * A_ACC_PLAN * ds, 2.0 * A_BRK_PLAN * ds
    for _ in range(2):
        for i in range(1, 2 * n):
            j, p = i % n, (i - 1) % n
            v[j] = min(v[j], math.sqrt(v[p] * v[p] + acc2))
        for i in range(2 * n - 2, -1, -1):
            j, q = i % n, (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[q] * v[q] + brk2))
    _PROFILES[key] = (v, length)
    return v, length


def profile_speed(track: Track, branch: Branch, direction: Direction, s: float, params: ExpertParams = ExpertParams()) -> float:
    """Reference speed (m/s) of the speed profile at Station s."""
    d = 1 if direction == "cw" else -1
    v, length = _profile(track, branch, d, params.a_lat_plan, params.v_cap)
    return v[round(((d * s) % length) / PROFILE_DS) % len(v)]


# -- expert (SPEC §5) -----------------------------------------------------------------------
class Expert:
    def __init__(self, track: Track, instruction: Instruction, params: ExpertParams = ExpertParams()) -> None:
        if instruction not in INSTRUCTIONS:
            raise ValueError(f"unknown instruction {instruction!r}")
        p = params
        if not 0.0 < p.a_lat_plan < 8.0:
            raise ValueError(f"a_lat_plan must be in (0, 8), got {p.a_lat_plan}")
        if not 0.0 < p.v_cap <= 10.0:
            raise ValueError(f"v_cap must be in (0, 10], got {p.v_cap}")
        if not 0.0 <= p.burst_peak <= 1.0:
            raise ValueError(f"burst_peak must be in [0, 1], got {p.burst_peak}")
        if not p.burst_gap_s > 0.0:
            raise ValueError(f"burst_gap_s must be > 0, got {p.burst_gap_s}")
        if not 0.0 <= p.p_outer <= 1.0:
            raise ValueError(f"p_outer must be in [0, 1], got {p.p_outer}")
        self.track = track
        self.instruction: Instruction = instruction
        self.params = params
        self._ready = False
        self._planned: Branch = "outer"
        self._safety_switches = 0

    # -- properties -------------------------------------------------------------------------
    @property
    def planned_branch(self) -> Branch:
        return self._planned

    @property
    def safety_switches(self) -> int:
        return self._safety_switches

    # -- clean control law (SPEC §4); pure ---------------------------------------------------
    def control(self, state: State, branch: Branch, direction: Direction) -> tuple[float, float]:
        route = self.track.route(branch)
        d = 1 if direction == "cw" else -1
        x, y, theta, v = state.x, state.y, state.theta, state.v

        pr = route.project(x, y)
        s = pr.station_m
        e = d * pr.lateral_m
        _, _, th = route.pose_at(s)
        the = wrap(theta - th - (math.pi if d < 0 else 0.0))

        vref = profile_speed(self.track, branch, direction, s + d * (v * DT + SPEED_LOOKAHEAD_M), self.params)
        a = (vref - v) / DT
        throttle = min(1.0, a / ACCEL) if a >= 0 else max(-1.0, a / BRAKE)

        v1 = min(max(v + (ACCEL * throttle if throttle >= 0 else BRAKE * throttle) * DT, 0.0), V_MAX)

        dist = max(v1 * DT, 1e-3)
        _, _, th1 = route.pose_at(s + d * dist)
        kbar = wrap(th1 - th) / dist

        sinc = math.sin(the) / the if abs(the) > 1e-6 else 1.0
        k = kbar * math.cos(the) / max(1.0 - kbar * e, 0.2) - 2.0 * WS * the - WS * WS * e * sinc

        T = _TAN_MAX if v1 <= 0 else min(_TAN_MAX, A_LAT_MAX * WHEELBASE / (v1 * v1))
        k = min(max(k, -T / WHEELBASE), T / WHEELBASE)

        steer = min(max(math.atan(WHEELBASE * k) / DELTA_MAX, -1.0), 1.0)
        return steer, throttle

    # -- episode ----------------------------------------------------------------------------
    def _preferred(self) -> Branch:
        if self.instruction == "any":
            return "outer" if self._coin_rng.random() < self.params.p_outer else "inner"
        return self.instruction

    def _feasible(self, state: State, branch: Branch, direction: Direction) -> bool:
        route = self.track.route(branch)
        if route.project(state.x, state.y).distance_m > WRONG_BRANCH_M:
            return False
        st = state
        for _ in range(_ROLLOUT_TICKS):
            st = step_state(st, self.control(st, branch, direction))
            if self.track.is_crash(st) or route.project(st.x, st.y).distance_m > WRONG_BRANCH_M:
                return False
        return True

    def _draw_next(self) -> None:
        rng = self._noise_rng
        self._wait = round(rng.exponential(self.params.burst_gap_s) / DT)
        self._n = round(rng.uniform(0.5, 1.5) / DT)
        self._amp = self.params.burst_peak * rng.uniform(0.5, 1.0) * (1.0 if rng.random() < 0.5 else -1.0)
        self._k = 0

    def reset(self, seed: int, state: State, info: Info) -> None:
        """Call right after env.reset, with the same seed."""
        self._coin_rng = np.random.default_rng([seed, 1])
        self._noise_rng = np.random.default_rng([seed, 2])
        direction = info.direction
        pref = self._preferred()
        other = _other(pref)
        if not self._feasible(state, pref, direction) and self._feasible(state, other, direction):
            self._planned = other
        else:
            self._planned = pref
        self._prev_branch: Branch | None = info.branch
        self._safety_switches = 0
        self._wait = self._n = self._k = 0
        self._amp = 0.0
        if self.params.burst_peak > 0:
            self._draw_next()
            self._wait += _GRACE_TICKS
        self._ready = True

    def act(self, state: State, info: Info) -> ExpertAction:
        """Exactly once per tick."""
        if not self._ready:
            raise RuntimeError("call reset() first")
        # A: section exit → redraw the plan
        if self._prev_branch is not None and info.branch is None:
            self._planned = self._preferred()
        self._prev_branch = info.branch
        # B: safety rule
        if (
            info.branch is not None
            and info.branch != self._planned
            and self.track.route(self._planned).project(state.x, state.y).distance_m > WRONG_BRANCH_M
        ):
            self._planned = info.branch
            self._safety_switches += 1
        # C: clean control
        clean = self.control(state, self._planned, info.direction)
        # D: steer-only triangular bursts
        if self.params.burst_peak == 0 or self._wait > 0:
            if self._wait > 0:
                self._wait -= 1
            return ExpertAction(clean, clean, False)
        tri = 1.0 - abs(2.0 * (self._k + 0.5) / self._n - 1.0)
        executed = (min(max(clean[0] + self._amp * tri, -1.0), 1.0), clean[1])
        self._k += 1
        if self._k >= self._n:
            self._draw_next()
        return ExpertAction(clean, executed, True)


# -- episode scoring (SPEC §6) --------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeScore:
    instruction: Instruction
    direction: Direction
    steps: int
    termination: Termination
    crashed: bool
    laps: int
    progress_m: float
    wrong_way_ticks: int
    passages: tuple[Branch, ...]  # every completed Passage, in order
    scored: int
    correct: int
    instruction_followed: bool
    success: bool

    @property
    def scored_passages(self) -> tuple[Branch, ...]:
        """The passages that count: all but an exempt first one."""
        return self.passages[len(self.passages) - self.scored :]


def score_episode(track: Track, instruction: Instruction, infos: Sequence[Info]) -> EpisodeScore:
    """Score one episode from its Info stream: infos[0] from reset, infos[k] after step k.

    `track` is accepted for interface stability only and is not used.
    """
    if len(infos) == 0:
        raise ValueError("infos is empty")
    if instruction not in INSTRUCTIONS:
        raise ValueError(f"unknown instruction {instruction!r}")
    first, last = infos[0], infos[-1]
    passages: list[Branch] = []
    for prev, cur in zip(infos, infos[1:]):
        if cur.passages > prev.passages:
            assert cur.last_branch is not None
            passages.append(cur.last_branch)
    scored_list = passages[1:] if first.branch is not None else passages
    scored = len(scored_list)
    correct = scored if instruction == "any" else sum(b == instruction for b in scored_list)
    wrong_way_ticks = sum(i.wrong_way for i in infos[1:])
    crashed = last.termination == "crash"
    followed = correct == scored and wrong_way_ticks == 0
    return EpisodeScore(
        instruction=instruction,
        direction=first.direction,
        steps=last.steps,
        termination=last.termination,
        crashed=crashed,
        laps=last.laps,
        progress_m=last.progress_m,
        wrong_way_ticks=wrong_way_ticks,
        passages=tuple(passages),
        scored=scored,
        correct=correct,
        instruction_followed=followed,
        success=not crashed and followed,
    )


def summarize(scores: Sequence[EpisodeScore]) -> dict:
    n = len(scores)
    any_scored = [b for sc in scores if sc.instruction == "any" for b in sc.scored_passages]
    return {
        "episodes": n,
        "crash_rate": sum(sc.crashed for sc in scores) / n if n else 0.0,
        "laps_mean": sum(sc.laps for sc in scores) / n if n else 0.0,
        "laps_min": min((sc.laps for sc in scores), default=0),
        "instruction_followed_rate": sum(sc.instruction_followed for sc in scores) / n if n else 0.0,
        "success_rate": sum(sc.success for sc in scores) / n if n else 0.0,
        "scored_passages": sum(sc.scored for sc in scores),
        "wrong_passages": sum(sc.scored - sc.correct for sc in scores),
        "outer_fraction": (sum(b == "outer" for b in any_scored) / len(any_scored)) if any_scored else None,
    }


# -- running episodes -----------------------------------------------------------------------
def start_spec(instruction: Instruction, direction: Direction | None) -> StartSpec:
    """The §2 recording rule: outer/inner start on their own Route; `any` keeps it random."""
    return StartSpec(direction=direction, route=None if instruction == "any" else instruction)


def run_episode(
    seed: int,
    instruction: Instruction,
    direction: Direction | None,
    params: ExpertParams = ExpertParams(),
    env: CarEnv | None = None,
) -> tuple[EpisodeScore, int]:
    """Headless expert episode under the recording rule → (score, safety_switches)."""
    env = env if env is not None else CarEnv()
    state, info = env.reset(seed, start_spec(instruction, direction))
    expert = Expert(env.track, instruction, params)
    expert.reset(seed, state, info)
    infos = [info]
    while True:
        a = expert.act(state, info)
        r = env.step(a.executed)
        state, info = r.state, r.info
        infos.append(info)
        if r.terminated or r.truncated:
            break
    return score_episode(env.track, instruction, infos), expert.safety_switches


def _eval_job(job: tuple[int, Instruction, Direction, ExpertParams]) -> tuple[EpisodeScore, int]:
    return run_episode(*job)


# -- CLI (SPEC §7) --------------------------------------------------------------------------
def _parse_params(items: Sequence[str], clean: bool) -> ExpertParams:
    fields = {f.name for f in dataclasses.fields(ExpertParams)}
    kv: dict[str, float] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or key not in fields:
            raise SystemExit(f"--param expects KEY=VALUE with KEY in {sorted(fields)}, got {item!r}")
        kv[key] = float(value)
    if clean:
        kv["burst_peak"] = 0.0
    return dataclasses.replace(ExpertParams(), **kv)


def cmd_eval(n: int, instructions: Sequence[Instruction], directions: Sequence[Direction], params: ExpertParams, workers: int) -> int:
    combos = [(i, d) for i in instructions for d in directions]
    jobs = [(seed, i, d, params) for i, d in combos for seed in range(n)]
    t0 = time.time()
    if workers <= 1:
        results = [_eval_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_eval_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    print(f"{len(jobs)} episodes in {time.time() - t0:.1f} s  ({params})\n")

    print(f"{'instr':6s} {'dir':4s} {'eps':>5s} {'crash':>6s} {'laps/ep':>8s} {'min laps':>8s} "
          f"{'followed':>9s} {'scored':>7s} {'wrong':>6s} {'outer%':>7s} {'safety':>7s}")
    bad = 0
    crash_lines = []
    for c, (i, d) in enumerate(combos):
        chunk = results[c * n : (c + 1) * n]
        scores = [sc for sc, _ in chunk]
        sm = summarize(scores)
        crashes = sum(sc.crashed for sc in scores)
        safety = sum(sw > 0 for _, sw in chunk)
        outer = "" if sm["outer_fraction"] is None else f"{100 * sm['outer_fraction']:6.1f}%"
        print(f"{i:6s} {d:4s} {n:5d} {crashes:6d} {sm['laps_mean']:8.2f} {sm['laps_min']:8d} "
              f"{100 * sm['instruction_followed_rate']:8.1f}% {sm['scored_passages']:7d} {sm['wrong_passages']:6d} "
              f"{outer:>7s} {safety:7d}")
        bad += crashes + sm["wrong_passages"]
        crash_lines += [f"  crash: {i} {d} seed {seed}" for seed, sc in enumerate(scores) if sc.crashed]
    if crash_lines:
        print()
        print("\n".join(crash_lines))
    print("\nPASS" if bad == 0 else "\nFAIL")
    return 0 if bad == 0 else 1


def cmd_show(seed: int | None, instruction: Instruction, direction: Direction | None, params: ExpertParams) -> int:
    import pygame

    from .drive import CRASH_FREEZE_S, PANEL_W, VIEW, frame_surface
    from .render import render

    env = CarEnv(max_steps=10**9)
    seed = seed if seed is not None else int(time.time())

    class Run:
        pass

    run = Run()

    def start(sd: int) -> None:
        print(f"seed {sd}", flush=True)
        run.seed = sd
        run.state, run.info = env.reset(sd, start_spec(instruction, direction))
        run.expert = Expert(env.track, instruction, params)
        run.expert.reset(sd, run.state, run.info)
        run.action = None

    pygame.init()
    show_panel = True
    screen = pygame.display.set_mode((VIEW + PANEL_W, VIEW))
    pygame.display.set_caption("toycar expert")
    font = pygame.font.SysFont("menlo,monaco,couriernew,monospace", 15)
    banner_font = pygame.font.SysFont("helvetica,arial", 72, bold=True)
    clock = pygame.time.Clock()
    start(seed)
    paused = hires = False
    crash_until: float | None = None
    try:
        while True:
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        quit_requested = True
                    elif event.key == pygame.K_h:
                        hires = not hires
                    elif event.key == pygame.K_TAB:
                        show_panel = not show_panel
                        screen = pygame.display.set_mode((VIEW + (PANEL_W if show_panel else 0), VIEW))
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_r and crash_until is None:
                        start(run.seed if event.mod & pygame.KMOD_SHIFT else run.seed + 1)
            if quit_requested:
                break

            if crash_until is not None:
                if time.monotonic() >= crash_until:
                    crash_until = None
                    start(run.seed + 1)
            elif not paused:
                run.action = run.expert.act(run.state, run.info)
                r = env.step(run.action.executed)
                run.state, run.info = r.state, r.info
                if r.terminated:
                    print(f"CRASH seed {run.seed}", flush=True)
                    crash_until = time.monotonic() + CRASH_FREEZE_S

            frame = render(env.state, env.track, env.render_params, resolution=VIEW) if hires else env.render()
            screen.blit(frame_surface(frame), (0, 0))
            if crash_until is not None:
                text = banner_font.render("CRASH", True, (255, 255, 255))
                box = text.get_rect(center=(VIEW // 2, VIEW // 2)).inflate(40, 20)
                pygame.draw.rect(screen, (200, 20, 20), box)
                screen.blit(text, text.get_rect(center=box.center))
            if show_panel:
                s, i, a, ex = run.state, run.info, run.action, run.expert
                clean_s = f"{a.clean[0]:+6.2f}" if a else "   -  "
                exec_s = f"{a.executed[0]:+6.2f}" if a else "   -  "
                thr_s = f"{a.clean[1]:+6.2f}" if a else "   -  "
                lines = [
                    f"seed      {run.seed}",
                    f"instr     {instruction}",
                    f"direction {i.direction}",
                    f"planned   {ex.planned_branch}",
                    f"branch    {i.branch}",
                    f"last_br   {i.last_branch}",
                    f"passages  {i.passages}",
                    f"safety    {ex.safety_switches}",
                    f"v         {s.v:6.2f} m/s",
                    f"steer cln {clean_s}",
                    f"steer exe {exec_s}",
                    f"throttle  {thr_s}",
                    "BURST" if a and a.perturbed else "",
                    f"laps      {i.laps}",
                    f"progress  {i.progress_m:6.1f} m",
                    f"wrong_way {i.wrong_way}",
                    f"steps     {i.steps}",
                    ("PAUSED" if paused else "") + ("  HI-RES" if hires else ""),
                    "",
                    "H hi-res  Tab panel  P pause",
                    "R next seed  Shift+R same",
                    "Esc/Q quit",
                ]
                pygame.draw.rect(screen, (24, 24, 28), (VIEW, 0, PANEL_W, VIEW))
                for k, text in enumerate(lines):
                    red = text == "BURST" or text.startswith("wrong_way True")
                    screen.blit(font.render(text, True, (255, 170, 60) if red else (220, 220, 220)), (VIEW + 10, 10 + k * 20))
            pygame.display.flip()
            clock.tick(15)
    finally:
        pygame.quit()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m toycar.expert", description="Scripted expert: watch it or evaluate it.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--clean", action="store_true", help="no noise bursts (burst_peak = 0)")
    common.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="override an ExpertParams field")
    common.add_argument("--direction", choices=["cw", "ccw"])
    sh = sub.add_parser("show", parents=[common], help="watch the expert drive")
    sh.add_argument("--seed", type=int)
    sh.add_argument("--instruction", choices=INSTRUCTIONS, default="any")
    ev = sub.add_parser("eval", parents=[common], help="headless reliability sweep")
    ev.add_argument("--n", type=int, default=1000)
    ev.add_argument("--instruction", choices=INSTRUCTIONS)
    ev.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)

    params = _parse_params(args.param, args.clean)
    Expert(Track.default(), "any", params)  # validate params up front
    if args.cmd == "show":
        return cmd_show(args.seed, args.instruction, args.direction, params)
    instructions = [args.instruction] if args.instruction else list(INSTRUCTIONS)
    directions = cast(list[Direction], [args.direction] if args.direction else ["cw", "ccw"])
    return cmd_eval(args.n, instructions, directions, params, args.workers)


if __name__ == "__main__":
    sys.exit(main())
