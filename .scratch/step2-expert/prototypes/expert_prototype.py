"""PROTOTYPE — throwaway. Ticket 03 "Expert controller and speed profile".

Question: does curvature feed-forward + lateral/heading feedback, following the Planned branch's
Route, with a forward/backward speed profile at a_lat_plan = 5 m/s², drive every
Instruction × Direction from every valid start pose without Crashing, at a sensible pace and with
learnable (smooth) actions?

Run (from repo root):
  uv run --with matplotlib python .scratch/step2-expert/prototypes/expert_prototype.py sweep [--n 1000] [--noise]
  uv run --with matplotlib python .scratch/step2-expert/prototypes/expert_prototype.py plot
  uv run python .scratch/step2-expert/prototypes/expert_prototype.py show [--seed N] [--instruction any] [--direction cw] [--noise]
Out: .scratch/step2-expert/prototypes/out/
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from toycar import ACCEL, A_LAT_MAX, BRAKE, DELTA_MAX, DT, V_MAX, WHEELBASE as L, CarEnv, StartSpec, Track, wrap

OUT = Path(__file__).parent / "out"
TAN_MAX = math.tan(DELTA_MAX)

# ---- parameters under test -----------------------------------------------------------------
A_LAT_PLAN = 5.0  # m/s² used to plan curve speeds (clamp is 8)
A_BRK_PLAN = 6.0  # m/s² planned braking (limit 8)
A_ACC_PLAN = 4.0
GRACE_S = 2.0  # s without bursts after reset
WS = 0.5  # spatial bandwidth 1/m → ke = WS², kθ = 2·WS (critically damped)
SPEED_LOOKAHEAD_M = 0.5
ROLLOUT_S = 3.0  # at reset: keep the preferred branch unless a clean rollout of it crashes within this time
WRONG_BRANCH_M = 2.5  # switch plan to the branch we're in only when this far off the planned Route


# ---- speed profile ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def profile(route_name: str, d: int, a_lat: float = A_LAT_PLAN, v_cap: float = V_MAX, ds: float = 0.05) -> tuple[np.ndarray, float, float]:
    route = Track.default().route(route_name)
    n = int(math.ceil(route.length_m / ds))
    st = [(d * i * ds) % route.length_m for i in range(n)]
    kap = np.array([abs(route.curvature_at(s)) for s in st])
    v = np.minimum(v_cap, np.sqrt(a_lat / np.maximum(kap, 1e-9)))
    for _ in range(2):
        for i in range(1, 2 * n):
            j, p = i % n, (i - 1) % n
            v[j] = min(v[j], math.sqrt(v[p] ** 2 + 2 * A_ACC_PLAN * ds))
        for i in range(2 * n - 2, -1, -1):
            j, nx = i % n, (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[nx] ** 2 + 2 * A_BRK_PLAN * ds))
    return v, ds, route.length_m


def profile_at(route_name: str, d: int, s: float, a_lat: float = A_LAT_PLAN, v_cap: float = V_MAX) -> float:
    v, ds, length = profile(route_name, d, a_lat, v_cap)
    return float(v[int(round(((d * s) % length) / ds)) % len(v)])


# ---- expert ----------------------------------------------------------------------------------
class Expert:
    """Instruction-following expert. Stateful only in the Planned branch (ticket 02)."""

    def __init__(self, track: Track, instruction: str, direction: str, seed: int,
                 a_lat_plan: float = A_LAT_PLAN, v_cap: float = V_MAX, p_outer: float = 0.5) -> None:
        self.track, self.instruction = track, instruction
        self.d = 1 if direction == "cw" else -1
        self.rng = np.random.default_rng([seed, 1])  # coin stream (ticket 04 Q4)
        self.a_lat_plan, self.v_cap, self.p_outer = a_lat_plan, v_cap, p_outer
        self.planned: str = "outer"
        self.exempt = False
        self.switches = 0

    def _draw(self) -> str:
        if self.instruction == "any":
            return "outer" if self.rng.random() < self.p_outer else "inner"
        return self.instruction

    def feasible(self, state, info, branch: str) -> bool:
        """Clean rollout on `branch` for ROLLOUT_S: no Crash and never > WRONG_BRANCH_M from its Route?"""
        from toycar import step_state
        saved = self.planned
        self.planned = branch
        try:
            st = state
            if self.track.route(branch).project(st.x, st.y).distance_m > WRONG_BRANCH_M:
                return False
            for _ in range(round(ROLLOUT_S / DT)):
                st = step_state(st, self._control(st, branch))
                off = self.track.route(branch).project(st.x, st.y).distance_m > WRONG_BRANCH_M
                if self.track.is_crash(st) or off:
                    return False
            return True
        finally:
            self.planned = saved

    def reset(self, state, info) -> None:
        preferred = self._draw()  # Instruction, or the coin under `any` (always one draw)
        other = "inner" if preferred == "outer" else "outer"
        self.planned = other if not self.feasible(state, info, preferred) and self.feasible(state, info, other) else preferred
        self.exempt = self.planned != self.instruction and self.instruction != "any"

    def on_exit(self) -> None:
        """Car left the fork section (either end): redraw."""
        self.planned = self._draw()
        self.exempt = False

    def errors(self, state, route_name: str):
        route = self.track.route(route_name)
        pr = route.project(state.x, state.y)
        _, _, th = route.pose_at(pr.station_m)
        return pr, self.d * pr.lateral_m, wrap(state.theta - th - (math.pi if self.d < 0 else 0.0))

    def act(self, state, info) -> tuple[float, float]:
        # safety rule: in the section but clearly not on the planned branch → follow where we are
        if info.branch is not None and info.branch != self.planned:
            pr = self.track.route(self.planned).project(state.x, state.y)
            if pr.distance_m > WRONG_BRANCH_M:
                self.planned, self.exempt = info.branch, True
                self.switches += 1
        return self._control(state, self.planned)

    def _control(self, state, route_name: str) -> tuple[float, float]:
        route = self.track.route(route_name)
        pr, e, the = self.errors(state, route_name)
        s = pr.station_m

        vref = profile_at(route_name, self.d, s + self.d * (state.v * DT + SPEED_LOOKAHEAD_M), self.a_lat_plan, self.v_cap)
        a = (vref - state.v) / DT
        thr = min(1.0, a / ACCEL) if a >= 0 else max(-1.0, a / BRAKE)
        v1 = min(max(state.v + (ACCEL * thr if thr >= 0 else BRAKE * thr) * DT, 0.0), V_MAX)

        dist = max(v1 * DT, 1e-3)
        _, _, t0 = route.pose_at(s)
        _, _, t1 = route.pose_at(s + self.d * dist)
        kap = wrap(t1 - t0) / dist  # travel-signed mean curvature over the next tick
        sinc = math.sin(the) / the if abs(the) > 1e-6 else 1.0
        k = kap * math.cos(the) / max(1 - kap * e, 0.2) - 2 * WS * the - WS * WS * e * sinc
        T = TAN_MAX if v1 <= 0 else min(TAN_MAX, A_LAT_MAX * L / (v1 * v1))
        k = max(-T / L, min(T / L, k))
        steer = max(-1.0, min(1.0, math.atan(L * k) / DELTA_MAX))
        return steer, thr


class Bursts:
    """Steer-only triangular bursts on the *executed* action (ticket 04 Q2); noise stream [seed, 2]."""

    def __init__(self, seed: int, peak: float = 0.4, gap_s: float = 3.0) -> None:
        self.rng = np.random.default_rng([seed, 2])
        self.peak, self.gap_s = peak, gap_s
        self._next_gap()
        self.wait += round(GRACE_S / DT)  # no bursts while recovering from the random start pose

    def _next_gap(self) -> None:
        self.wait = int(round(self.rng.exponential(self.gap_s) / DT))
        self.n = int(round(self.rng.uniform(0.5, 1.5) / DT))
        self.amp = self.peak * self.rng.uniform(0.5, 1.0) * (1 if self.rng.random() < 0.5 else -1)
        self.k = 0

    def __call__(self, action):
        if self.peak <= 0:
            return action, False
        if self.wait > 0:
            self.wait -= 1
            return action, False
        tri = 1 - abs(2 * (self.k + 0.5) / self.n - 1)
        out = (max(-1.0, min(1.0, action[0] + self.amp * tri)), action[1])
        self.k += 1
        if self.k >= self.n:
            self._next_gap()
        return out, True


# ---- one episode -----------------------------------------------------------------------------
def episode(seed: int, instruction: str, direction: str, noise: bool = False, keep: bool = False, knobs: dict | None = None) -> dict:
    knobs = knobs or {}
    env = CarEnv()
    route = None if instruction == "any" else instruction
    state, info = env.reset(seed, StartSpec(direction=direction, route=route))
    ex = Expert(env.track, instruction, direction, seed, **{k: knobs[k] for k in ("a_lat_plan", "v_cap", "p_outer") if k in knobs})
    ex.reset(state, info)
    burst = Bursts(seed, knobs.get("burst_peak", 0.4), knobs.get("burst_gap_s", 3.0)) if noise else None
    perturbed_ticks = 0
    passages = []  # (branch taken, planned, exempt)
    trace = []
    max_e = 0.0
    mid_best, mid_branch = math.inf, None
    r = None
    for _ in range(env.max_steps):
        a = ex.act(state, info)
        execd, pert = burst(a) if burst else (a, False)
        perturbed_ticks += pert
        prev_branch, exempt = info.branch, ex.exempt
        r = env.step(execd)
        state, info = r.state, r.info
        _, e, _ = ex.errors(state, ex.planned)
        max_e = max(max_e, abs(e))
        if keep:
            trace.append((state.x, state.y, state.v, a[0], a[1], execd[0], e, pert))
        if info.branch is not None:
            dmid = abs(info.station_m - env.track.merge_station_m / 2)
            if dmid < mid_best:
                mid_best, mid_branch = dmid, info.branch
        if prev_branch is not None and info.branch is None:
            partial = mid_best > 1.0  # never reached the section midpoint → branch not scoreable
            passages.append((mid_branch, ex.planned, exempt or partial, prev_branch))
            mid_best, mid_branch = math.inf, None
            ex.on_exit()
        if r.terminated:
            break
    out = dict(
        seed=seed, instruction=instruction, direction=direction, crash=r.info.termination == "crash",
        laps=r.info.laps, progress=r.info.progress_m, passages=passages, max_e=max_e, switches=ex.switches,
        pert_frac=perturbed_ticks / max(1, r.info.steps),
    )
    if keep:
        out["trace"] = np.array(trace)
    return out


def _job(args):
    return episode(*args)


# ---- commands --------------------------------------------------------------------------------
COMBOS = [(i, d) for i in ("outer", "inner", "any") for d in ("cw", "ccw")]


def sweep(n: int, noise: bool) -> int:
    jobs = [(seed, i, d, noise) for i, d in COMBOS for seed in range(n)]
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        res = list(pool.map(_job, jobs, chunksize=20))
    print(f"{len(res)} episodes in {time.time() - t0:.0f} s  (noise={'bursts' if noise else 'off'})\n")
    hdr = f"{'instr':6s} {'dir':4s} {'crash':>6s} {'laps/ep':>8s} {'min laps':>8s} {'passages':>8s} {'wrong':>6s} {'exempt':>6s} {'outer%':>7s} {'max|e| p99':>10s} {'exit-label≠':>9s}"
    print(hdr)
    bad = 0
    for i, d in COMBOS:
        rs = [r for r in res if r["instruction"] == i and r["direction"] == d]
        crashes = sum(r["crash"] for r in rs)
        ps = [p for r in rs for p in r["passages"]]
        scored = [p for p in ps if not p[2]]
        wrong = sum(1 for taken, planned, *_ in scored if taken != planned or (i != "any" and taken != i))
        outer = sum(1 for taken, *_ in scored if taken == "outer") / max(1, len(scored))
        mislabel = sum(1 for taken, _, _, at_exit in ps if taken != at_exit)
        laps = np.array([r["laps"] for r in rs])
        p99 = np.percentile([r["max_e"] for r in rs], 99)
        print(f"{i:6s} {d:4s} {crashes:6d} {laps.mean():8.2f} {laps.min():8d} {len(ps):8d} {wrong:6d} "
              f"{len(ps) - len(scored):6d} {100 * outer:6.1f}% {p99:10.2f} {mislabel:9d}")
        bad += crashes + wrong
        for r in rs:
            if r["crash"]:
                print(f"   crash: seed {r['seed']}  progress {r['progress']:.1f} m")
    print("\nPASS" if bad == 0 else f"\nFAIL ({bad} crashes or wrong-branch passages)")
    return 0 if bad == 0 else 1


def edges(n: int) -> int:
    base = dict(a_lat_plan=5.0, v_cap=10.0, burst_peak=0.4, burst_gap_s=3.0, p_outer=0.5)
    variants = [("defaults", {})] + [(f"{k}={v}", {k: v}) for k, vs in [
        ("a_lat_plan", [4.0, 5.5, 6.0]), ("v_cap", [6.0, 8.0]), ("burst_peak", [0.0, 0.6, 0.8, 1.0]),
        ("burst_gap_s", [1.5, 1.0]), ("p_outer", [0.0, 0.7, 1.0])] for v in vs]
    variants.append(("worst: a_lat 5.5, peak 0.6, gap 1.5", dict(a_lat_plan=5.5, burst_peak=0.6, burst_gap_s=1.5)))
    print(f"{'variant':38s} {'crash':>6s} {'wrong':>6s} {'laps/ep':>8s} {'pert%':>6s} {'outer%':>7s}")
    bad = 0
    with ProcessPoolExecutor() as pool:
        for name, kv in variants:
            knobs = {**base, **kv}
            jobs = [(seed, i, d, True, False, knobs) for i, d in COMBOS for seed in range(n)]
            res = list(pool.map(_job, jobs, chunksize=20))
            crashes = sum(r["crash"] for r in res)
            wrong = 0
            for r in res:
                for taken, planned, exempt, _ in r["passages"]:
                    if not exempt and (taken != planned or (r["instruction"] != "any" and taken != r["instruction"])):
                        wrong += 1
            anyp = [p for r in res if r["instruction"] == "any" for p in r["passages"] if not p[2]]
            outer = sum(p[0] == "outer" for p in anyp) / max(1, len(anyp))
            print(f"{name:38s} {crashes:6d} {wrong:6d} {np.mean([r['progress'] for r in res]) / 88.056:8.2f} "
                  f"{100 * np.mean([r['pert_frac'] for r in res]):5.1f}% {100 * outer:6.1f}%", flush=True)
    return 0


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from toycar import RenderParams, State, render

    OUT.mkdir(exist_ok=True)
    track = Track.default()
    bg = render(State(-100, -100, 0, 0), track, RenderParams(), resolution=512)

    # 1. trajectories: 6 combos × 8 seeds over the track, clean and with bursts
    for noise in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, (i, d) in zip(axes.T.flat, COMBOS):
            ax.imshow(bg, extent=(0, 40, 40, 0))
            for seed in range(8):
                tr = episode(seed, i, d, noise=noise, keep=True)["trace"]
                ax.plot(tr[:, 0], tr[:, 1], lw=0.8)
                if noise:
                    p = tr[:, 7] > 0
                    ax.plot(tr[p, 0], tr[p, 1], ".", ms=1.5, color="yellow")
            ax.set_title(f"{i} / {d}" + ("  (bursts: yellow)" if noise else ""))
            ax.set_xlim(0, 40); ax.set_ylim(36, 0); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / f"trajectories{'_noise' if noise else ''}.png", dpi=80)
        plt.close(fig)

    # 2. action traces: one `any` episode per direction, clean and noisy
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax, (d, noise) in zip(axes, [("cw", False), ("ccw", False), ("cw", True), ("ccw", True)]):
        tr = episode(3, "any", d, noise=noise, keep=True)["trace"]
        t = np.arange(len(tr)) * DT
        ax.plot(t, tr[:, 3], label="steer (expert label)")
        if noise:
            ax.plot(t, tr[:, 5], lw=0.7, alpha=0.7, label="steer executed")
        ax.plot(t, tr[:, 4], label="throttle")
        ax.plot(t, tr[:, 2] / 10, label="v / 10")
        ax.plot(t, tr[:, 6], lw=0.7, label="lateral error e (m)")
        ax.set_ylim(-1.1, 1.1); ax.grid(alpha=0.3)
        ax.set_title(f"instruction any, {d}, seed 3, {'bursts' if noise else 'clean'}")
        ax.legend(loc="upper right", fontsize=7, ncol=5)
    axes[-1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(OUT / "action_traces.png", dpi=80)
    plt.close(fig)
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))} to {OUT}")


def show(seed: int, instruction: str, direction: str, noise: bool) -> None:
    import pygame

    from toycar.drive import frame_surface

    env = CarEnv(max_steps=10**9)
    pygame.init()
    screen = pygame.display.set_mode((512 + 220, 512))
    font = pygame.font.SysFont("menlo,monaco,monospace", 15)
    clock = pygame.time.Clock()

    def start(seed):
        route = None if instruction == "any" else instruction
        st, info = env.reset(seed, StartSpec(direction=direction, route=route))
        ex = Expert(env.track, instruction, direction, seed)
        ex.reset(st, info)
        return ex, info, (Bursts(seed) if noise else None)

    ex, info, burst = start(seed)
    a, pert = (0.0, 0.0), False
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                seed += 1; ex, info, burst = start(seed)
        a = ex.act(env.state, info)
        execd, pert = burst(a) if burst else (a, False)
        prev = info.branch
        r = env.step(execd)
        info = r.info
        if prev is not None and info.branch is None:
            ex.on_exit()
        screen.blit(frame_surface(env.render()), (0, 0))
        s = env.state
        lines = [f"seed {seed}", f"instr {instruction}", f"dir {direction}", f"planned {ex.planned}",
                 f"branch {info.branch}", f"last_br {info.last_branch}", f"v {s.v:5.2f}",
                 f"steer {a[0]:+.2f}", f"throttle {a[1]:+.2f}", f"burst {'ON' if pert else ''}",
                 f"laps {info.laps}", f"steps {info.steps}", "", "R next seed, Q quit"]
        pygame.draw.rect(screen, (24, 24, 28), (512, 0, 220, 512))
        for k, t in enumerate(lines):
            screen.blit(font.render(t, True, (220, 220, 220)), (522, 10 + 20 * k))
        pygame.display.flip()
        if r.terminated:
            print(f"CRASH seed {seed}"); pygame.time.wait(1500); seed += 1; ex, info, burst = start(seed)
        clock.tick(15)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "edges", "plot", "show"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default="any", choices=["outer", "inner", "any"])
    ap.add_argument("--direction", default="cw", choices=["cw", "ccw"])
    args = ap.parse_args()
    if args.cmd == "sweep":
        sys.exit(sweep(args.n, args.noise))
    elif args.cmd == "edges":
        sys.exit(edges(args.n))
    elif args.cmd == "plot":
        plot()
    else:
        show(args.seed, args.instruction, args.direction, args.noise)
