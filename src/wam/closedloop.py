"""Closed-loop evaluation (Step 9 SPEC, `.scratch/step9-closed-loop/SPEC.md`): Evaluation sets, metrics, gates,
the verdict for `wam-v1`, sweeps, the robustness report, Run directories and the CLI.

    caffeinate -i uv run python -m wam.closedloop eval <driver> [--name N] [--episodes 120] [scheme overrides]
                                       [--sweep replan|cfg|samples|steps] [--robustness [COND ...]]
                                       [--batch 24] [--device mps|cpu] [--json PATH]
    uv run python -m wam.closedloop show  <driver> [--set gate|dev|robustness] [--k K] [--instruction I] [--direction D] [--condition C] [--imagine]
    uv run python -m wam.closedloop video <run_dir> --episode K [--point P] [--out DIR] [--hires]

`<driver>`: a Checkpoint name or Run directory (a Policy), `expert` (the clean expert) or `saboteur:<name>`.
Scheme overrides: --execute-k --n-samples --n-steps --cfg-action --cfg-video --select --rho --seed-mode.
Exit codes: 0 = PASS or no verdict, 1 = FAIL (only a `wam-v1` default-Scheme full gate run), 2 = usage/load error.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import re
import struct
import sys
import time
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from toycar.dataset import CLOSED_LOOP_SEED_BLOCK, COMBINATIONS
from toycar.expert import summarize

from .drive import (CONDITIONS, DEFAULT_SCHEME, MAX_STEPS, PUSH_TICKS, SABOTEURS, EpisodeRecord, EpisodeSpec,
                    ExpertDriver, PolicyDriver, SaboteurDriver, Scheme, condition_render_params, condition_track,
                    run_episodes)
from .runs import CHECKPOINTS, _git_info, _new_run_dir, _now, _rel, _sha256, _Tee, data_path

RUNS_DIR = Path("runs/drive")
DEFAULT_DATA = "data/step3-v1"
GATED_POLICY = "wam-v1"
FPS = 15

# -- Evaluation sets (SPEC §3.1) ------------------------------------------------------------
SETS: dict[str, tuple[int, int]] = {  # name → (first seed, count)
    "dev": (CLOSED_LOOP_SEED_BLOCK + 10_000, 90_000),
    "gate": (CLOSED_LOOP_SEED_BLOCK + 100_000, 120),
    "robustness": (CLOSED_LOOP_SEED_BLOCK + 200_000, 60),
}
GATE_EPISODES = SETS["gate"][1]
SWEEP_EPISODES = 60
ROBUSTNESS_EPISODES = SETS["robustness"][1]


def episode_specs(set_name: str, n: int, condition: str = "none") -> list[EpisodeSpec]:
    """Episodes k = 0 … n−1 of an Evaluation set; episode k plays COMBINATIONS[k % 6]."""
    if set_name not in SETS:
        raise ValueError(f"unknown Evaluation set {set_name!r}; known: {list(SETS)}")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown Stress condition {condition!r}; known: {list(CONDITIONS)}")
    first, count = SETS[set_name]
    if n % len(COMBINATIONS) != 0:
        raise ValueError(f"the number of episodes must be a multiple of {len(COMBINATIONS)}, got {n}")
    if not 0 < n <= count:
        raise ValueError(f"set {set_name!r} has {count} episodes, asked for {n}")
    return [EpisodeSpec(first + k, *COMBINATIONS[k % len(COMBINATIONS)], condition=condition) for k in range(n)]


# -- metrics (SPEC §5) ----------------------------------------------------------------------
Z = 1.96
DIST_BINS = (0.0, 0.3, 0.6, 1.0, math.inf)
STATION_BIN_M = 2.0
RECOVERED_M = 0.3
PUSH_WINDOW = 45  # Ticks (3 s) after a Push start
COMBO_KEYS = [f"{i}/{d}" for i, d in COMBINATIONS]


def wilson(x: int, n: int, z: float = Z) -> tuple[float, float]:
    """95 % Wilson interval of x successes in n, clipped to [0, 1]."""
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _entries(branch: np.ndarray) -> list[int]:
    """Ticks te at which the State enters the fork section (branch[te] ≠ −1, branch[te−1] = −1)."""
    b = np.asarray(branch)
    return [int(t) for t in np.nonzero((b[1:] != -1) & (b[:-1] == -1))[0] + 1]


def fork_flips(records: Sequence[EpisodeRecord], ref: np.ndarray) -> dict:
    """Per approach to the Fork: classes of the selected Chunks of replans 1 … min(MAX_ENTRY, H) Ticks before
    entry; it flips if they disagree (SPEC §5, `replan_variants.fork_flips`)."""
    from .evaluate import COMMIT_R, MAX_ENTRY, classify

    out = {"approaches": 0, "flips": 0, "any_approaches": 0, "any_flips": 0, "classified": 0, "uncommitted": 0}
    for r in records:
        if not r.replans:
            continue
        H = len(r.replans[0][1])
        emax = min(MAX_ENTRY, H, ref.shape[0] - 1)
        sign = 1.0 if r.spec.direction == "cw" else -1.0
        for te in _entries(r.branch):
            cls = []
            for t, steer in r.replans:
                e = te - t
                if not 1 <= e <= emax:
                    continue
                r0, r1 = ref[e, 0, :H], ref[e, 1, :H]
                if np.isnan(r0).any() or np.isnan(r1).any():
                    continue
                c, rr = classify(np.asarray(steer) * sign, r0, r1)
                cls.append(int(c))
                out["classified"] += 1
                out["uncommitted"] += int(rr >= COMMIT_R)
            if len(cls) < 2:
                continue
            flip = len(set(cls)) > 1
            out["approaches"] += 1
            out["flips"] += int(flip)
            if r.spec.instruction == "any":
                out["any_approaches"] += 1
                out["any_flips"] += int(flip)
    out["fraction_uncommitted"] = out["uncommitted"] / out["classified"] if out["classified"] else None
    return out


def action_error(records: Sequence[EpisodeRecord]) -> dict:
    """RMSE of the Driver's steer vs the shadow expert's, binned by the State's distance off the line (SPEC §5).
    In `any` episodes, Ticks in the fork section or within MAX_ENTRY Ticks before an entry are left out."""
    from .evaluate import MAX_ENTRY

    errs, dists = [], []
    for r in records:
        n = r.steps
        if n == 0:
            continue
        keep = np.isfinite(r.expert_steer)
        if r.spec.instruction == "any":
            keep &= r.branch[:n] == -1
            for te in _entries(r.branch):
                keep[max(0, te - MAX_ENTRY):min(n, te)] = False
        errs.append((r.action_driver[:, 0] - r.expert_steer)[keep])
        dists.append(r.dist_state[:n][keep])
    e = np.concatenate(errs) if errs else np.zeros(0)
    d = np.concatenate(dists) if dists else np.zeros(0)
    out = {"all": {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e ** 2))) if len(e) else None}}
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        m = (d >= lo) & (d < hi)
        key = f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">{lo:g}"
        out[key] = {"n": int(m.sum()), "rmse": float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else None}
    return out


def push_recovery(records: Sequence[EpisodeRecord]) -> dict:
    """Per Push: a Crash within 45 Ticks of its start, the peak distance off the line in those Ticks and the Ticks
    from its end until the State is back within 0.3 m (censored at the next Push or the episode end)."""
    n_push = crashed = 0
    peaks, rec_ticks, censored = [], [], 0
    for r in records:
        dist = r.dist_state
        T = r.steps
        for m, p in enumerate(r.push_ticks):
            n_push += 1
            if r.crash_tick is not None and p < r.crash_tick <= p + PUSH_WINDOW:
                crashed += 1
            peaks.append(float(dist[p + 1:min(T, p + PUSH_WINDOW) + 1].max()) if p + 1 <= T else math.nan)
            end = p + PUSH_TICKS
            stop = min(r.push_ticks[m + 1] if m + 1 < len(r.push_ticks) else T, T)
            found = None
            for t in range(end, stop + 1):
                if dist[t] < RECOVERED_M:
                    found = t - end
                    break
            if found is None:
                censored += 1
            else:
                rec_ticks.append(found)
    return {"pushes": n_push, "crash_within_45": crashed,
            "crash_within_45_rate": crashed / n_push if n_push else None,
            "peak_dist_m_mean": float(np.nanmean(peaks)) if peaks else None,
            "recovered_fraction": len(rec_ticks) / n_push if n_push else None,
            "recovery_ticks_median": float(np.median(rec_ticks)) if rec_ticks else None}


def offset_recovery(records: Sequence[EpisodeRecord], reference: Sequence[EpisodeRecord] | None) -> dict:
    """offset_start: crash rate on episodes the clean expert survives; Ticks until first within 0.3 m."""
    ticks = []
    for r in records:
        hit = np.nonzero(r.dist_state < RECOVERED_M)[0]
        if len(hit):
            ticks.append(int(hit[0]))
    out = {"recovered_fraction": len(ticks) / len(records) if records else None,
           "ticks_to_line_median": float(np.median(ticks)) if ticks else None}
    if reference is not None:
        alive = {x.spec.seed for x in reference if not x.score.crashed}
        sel = [r for r in records if r.spec.seed in alive]
        out["expert_survived"] = len(sel)
        out["crash_rate_where_expert_survives"] = (sum(r.score.crashed for r in sel) / len(sel)) if sel else None
    return out


def group_metrics(records: Sequence[EpisodeRecord]) -> dict:
    """The §5 metrics of one group of episodes (the gate inputs with Wilson intervals, outcome, line holding,
    action error, fork, smoothness, cost)."""
    n = len(records)
    scores = [r.score for r in records]
    sm = summarize(scores)
    crashes = sum(s.crashed for s in scores)
    followed = sum(s.instruction_followed for s in scores)
    any_scored = [b for s in scores if s.instruction == "any" for b in s.scored_passages]
    n_outer = sum(b == "outer" for b in any_scored)
    frac = np.array([r.fractional_laps for r in records]) if n else np.zeros(0)
    ok = [r for r in records if not r.score.crashed]
    dist = np.concatenate([r.dist_m for r in records]) if n else np.zeros(0)
    dsteer = [np.abs(np.diff(r.action_driver[:, 0])) for r in records if r.steps > 1]
    dsteer = np.concatenate(dsteer) if dsteer else np.zeros(0)
    ms = [r.ms_per_replan for r in records if r.ms_per_replan is not None]
    passages = {"outer": sum(b == "outer" for s in scores for b in s.passages),
                "inner": sum(b == "inner" for s in scores for b in s.passages)}
    return {
        "episodes": n,
        "crash_rate": sm["crash_rate"], "crash_rate_ci": wilson(crashes, n),
        "frac_laps_mean": float(frac.mean()) if n else 0.0,
        "instruction_followed_rate": sm["instruction_followed_rate"],
        "instruction_followed_ci": wilson(followed, n),
        "outer_fraction": sm["outer_fraction"],
        "outer_fraction_ci": wilson(n_outer, len(any_scored)) if any_scored else None,
        "any_scored_passages": len(any_scored),
        "success_rate": sm["success_rate"], "success_ci": wilson(sum(s.success for s in scores), n),
        "laps_mean": sm["laps_mean"], "laps_min": sm["laps_min"],
        "frac_laps_noncrashed": float(np.mean([r.fractional_laps for r in ok])) if ok else None,
        "lap_time_s": float(np.mean([r.steps / FPS / r.fractional_laps for r in ok if r.fractional_laps > 0]))
        if any(r.fractional_laps > 0 for r in ok) else None,
        "crashes": crashes,
        "island_crashes": sum(bool(r.crash_in_section) for r in records),
        "dist_mean_m": float(dist.mean()) if len(dist) else None,
        "dist_p95_m": float(np.percentile(dist, 95)) if len(dist) else None,
        "dist_max_m": float(dist.max()) if len(dist) else None,
        "action_error": action_error(records),
        "scored_passages": sm["scored_passages"], "wrong_passages": sm["wrong_passages"], "passages": passages,
        "steer_change_mean": float(dsteer.mean()) if len(dsteer) else None,
        "wrong_way_ticks": sum(s.wrong_way_ticks for s in scores),
        "ms_per_replan": float(np.mean(ms)) if ms else None,
        "wall_s": float(sum(r.wall_s for r in records)),
    }


def metrics(records: Sequence[EpisodeRecord], *, ref: np.ndarray | None = None,
            reference: Sequence[EpisodeRecord] | None = None) -> dict:
    """§5 metrics in total and per combination, plus the per-episode Crash list, Station histogram, Fork flips
    (when `ref`, the Fork references, is given) and the Push / offset-start recovery numbers."""
    out = {"total": group_metrics(records), "combinations": {}}
    for (ins, d), key in zip(COMBINATIONS, COMBO_KEYS):
        sel = [r for r in records if r.spec.instruction == ins and r.spec.direction == d]
        if sel:
            out["combinations"][key] = group_metrics(sel)
    out["crash_list"] = [{"seed": r.spec.seed, "combination": f"{r.spec.instruction}/{r.spec.direction}",
                          "tick": r.crash_tick, "station_m": r.crash_station_m, "in_section": r.crash_in_section}
                         for r in records if r.crash_tick is not None]
    if records:
        L = condition_track(records[0].spec.condition).length_m
        edges = np.arange(0.0, L + STATION_BIN_M, STATION_BIN_M)
        h, _ = np.histogram([c["station_m"] for c in out["crash_list"]], bins=edges)
        out["crash_station_hist"] = {"bin_m": STATION_BIN_M, "counts": h.tolist()}
    has_replans = any(r.replans for r in records)
    no_flips = records and records[0].spec.condition == "wide_fork"
    out["fork_flips"] = fork_flips(records, ref) if (ref is not None and has_replans and not no_flips) else None
    if any(r.push_ticks for r in records):
        out["push"] = push_recovery(records)
    if records and records[0].spec.condition == "offset_start":
        out["offset_start"] = offset_recovery(records, reference)
    return out


# -- gates and verdict (SPEC §6) ------------------------------------------------------------
GATES = {
    "crashes": {"metric": "crash_rate", "ci": "crash_rate_ci", "max": 0.05},
    "laps": {"metric": "frac_laps_mean", "ci": None, "min": 3.5},
    "followed": {"metric": "instruction_followed_rate", "ci": "instruction_followed_ci", "min": 0.95},
    "balance": {"metric": "outer_fraction", "ci": "outer_fraction_ci", "min": 0.30, "max": 0.70},
}


def gates(total: Mapping) -> dict:
    """The four gates on a group's metrics; thresholds inclusive, `outer_fraction = None` fails balance."""
    out = {}
    for name, g in GATES.items():
        v = total.get(g["metric"])
        ok = v is not None and v >= g.get("min", -math.inf) and v <= g.get("max", math.inf)
        out[name] = {"value": v, "ci": total.get(g["ci"]) if g["ci"] else None, "pass": bool(ok),
                     **{k: g[k] for k in ("min", "max") if k in g}}
    return out


def verdict(gate_table: Mapping, *, driver: str, is_policy: bool, scheme: Scheme | None, set_name: str, n: int,
            condition: str = "none", mode: str = "gate") -> tuple[str | None, str | None]:
    """(PASS | FAIL, None) for a full default-Scheme gate run of `wam-v1`, else (None, reason)."""
    if not (is_policy and driver == GATED_POLICY):
        return None, "not gated"
    if scheme != DEFAULT_SCHEME:
        return None, "non-default scheme"
    if mode != "gate" or set_name != "gate" or n != GATE_EPISODES or condition != "none":
        return None, "not a gate run"
    return ("PASS" if all(g["pass"] for g in gate_table.values()) else "FAIL"), None


# -- sweeps (SPEC §7) -----------------------------------------------------------------------
SWEEPS: dict[str, tuple[str, tuple]] = {
    "replan": ("execute_k", (1, 2, 4, 8, 16)),
    "cfg": ("cfg_action", (1.0, 1.5, 2.0, 3.0)),
    "samples": ("n_samples", (1, 4, 8)),
    "steps": ("n_steps", (2, 4, 10)),
}


def sweep_points(name: str, base: Scheme, horizon: int) -> list[tuple[str, Scheme]]:
    field_, grid = SWEEPS[name]
    pts = []
    for v in grid:
        if field_ == "execute_k" and v > horizon:
            continue
        pts.append((f"{field_}={v:g}", replace(base, **{field_: v})))
    return pts


# -- Drivers from the command line ----------------------------------------------------------
def make_driver(target: str, scheme: Scheme, *, device: str | None = None, root: Path = CHECKPOINTS,
                batch: int = 24, **kw):
    """`expert`, `saboteur:<name>` or a Checkpoint name / Run directory → (Driver, policy or None)."""
    if target == "expert":
        return ExpertDriver(), None
    if target.startswith("saboteur:"):
        return SaboteurDriver(target.split(":", 1)[1]), None
    from . import policy as P

    pol = P.load(target, device, root=root)
    return PolicyDriver(pol, scheme, batch=batch, name=pol.name or Path(target).name, **kw), pol


def default_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


# -- Run directory (SPEC §9) ----------------------------------------------------------------
TICK_FIELDS = ("action_driver", "action_exec", "dist_m", "expert_steer", "selected")
STATE_FIELDS = ("x", "y", "theta", "v", "branch", "station_m")


def episode_line(r: EpisodeRecord, *, index: int, driver: str, role: str, point: str, k: int) -> dict:
    sc = asdict(r.score)
    sc["passages"] = list(sc["passages"])
    line = {"index": index, "driver": driver, "role": role, "point": point, "k": k, "spec": asdict(r.spec),
            "score": sc, "fractional_laps": r.fractional_laps, "crash_tick": r.crash_tick,
            "crash_station_m": r.crash_station_m, "crash_in_section": r.crash_in_section,
            "push_ticks": r.push_ticks, "steps": r.steps, "wall_s": r.wall_s, "ms_per_replan": r.ms_per_replan}
    if r.push_ticks:
        line["recovery"] = push_recovery([r])
    if r.spec.condition == "offset_start":
        line["recovery"] = offset_recovery([r], None)
    return line


def save_ticks(path: Path, records: Sequence[EpisodeRecord]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for f in TICK_FIELDS + STATE_FIELDS:
        parts = [getattr(r, f) for r in records]
        arrays[f] = np.concatenate(parts) if parts else np.zeros(0)
    arrays["offsets"] = np.cumsum([0] + [r.steps for r in records])[:-1].astype(np.int64)
    arrays["offsets_state"] = np.cumsum([0] + [r.steps + 1 for r in records])[:-1].astype(np.int64)
    H = max((len(r.replans[0][1]) for r in records if r.replans), default=0)
    reps = [(t, s) for r in records for t, s in r.replans]
    arrays["replan_tick"] = np.array([t for t, _ in reps], np.int64)
    arrays["replan_steer"] = np.array([s for _, s in reps], np.float64).reshape(len(reps), H)
    arrays["offsets_replan"] = np.cumsum([0] + [len(r.replans) for r in records])[:-1].astype(np.int64)
    arrays["dist0_m"] = np.array([r.dist0_m for r in records])
    np.savez_compressed(path, **arrays)


def _jsonable(obj):
    from .evaluate import jsonable

    return jsonable(obj)


# -- report ---------------------------------------------------------------------------------
def _f(v, fmt="{:.2f}") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return fmt.format(v)


def _pct(v) -> str:
    return _f(None if v is None else 100 * v, "{:.1f} %")


def _ci(ci) -> str:
    return "" if ci is None else f"{100 * ci[0]:.1f}–{100 * ci[1]:.1f} %"


def gate_table_md(row: Mapping, ref_row: Mapping | None) -> list[str]:
    g, rg = row["gates"], (ref_row or {}).get("gates")
    lines = ["| Gate | Value | 95 % CI | Threshold | Pass | Expert | Δ vs expert |", "|---|---|---|---|---|---|---|"]
    for name, x in g.items():
        is_rate = name != "laps"
        fmt = _pct if is_rate else _f
        thr = (f"≤ {fmt(x['max'])}" if "min" not in x else f"≥ {fmt(x['min'])}" if "max" not in x
               else f"{fmt(x['min'])} … {fmt(x['max'])}")
        e = rg[name]["value"] if rg else None
        d = (x["value"] - e) if (x["value"] is not None and e is not None) else None
        dd = _f(None if d is None else (100 * d if is_rate else d), "{:+.1f} pp" if is_rate else "{:+.2f}")
        lines.append(f"| {name} | {fmt(x['value'])} | {_ci(x['ci'])} | {thr} | {'PASS' if x['pass'] else 'FAIL'} "
                     f"| {fmt(e)} | {dd} |")
    return lines


def combo_table_md(m: Mapping) -> list[str]:
    lines = ["| Combination | Episodes | Crash | Frac. laps | Followed | Outer | Wrong passages | Island crashes "
             "| Dist mean / p95 (m) |", "|---|---|---|---|---|---|---|---|---|"]
    for key, c in list(m["combinations"].items()) + [("total", m["total"])]:
        lines.append(f"| {key} | {c['episodes']} | {_pct(c['crash_rate'])} | {_f(c['frac_laps_mean'])} "
                     f"| {_pct(c['instruction_followed_rate'])} | {_f(c['outer_fraction'])} | {c['wrong_passages']} "
                     f"| {c['island_crashes']} | {_f(c['dist_mean_m'])} / {_f(c['dist_p95_m'])} |")
    return lines


def detail_md(m: Mapping) -> list[str]:
    t = m["total"]
    ae = t["action_error"]
    lines = [f"- Success {_pct(t['success_rate'])}, integer laps mean {_f(t['laps_mean'])} / min {t['laps_min']}, "
             f"Fractional laps of non-crashed {_f(t['frac_laps_noncrashed'])}, lap time {_f(t['lap_time_s'], '{:.1f} s')}",
             f"- Line holding: dist mean {_f(t['dist_mean_m'])} m, p95 {_f(t['dist_p95_m'])} m, "
             f"max {_f(t['dist_max_m'])} m",
             "- Steer error vs shadow expert (RMSE): " + ", ".join(
                 f"{k} m {_f(v['rmse'], '{:.3f}')} (n {v['n']})" for k, v in ae.items()),
             f"- Smoothness: mean |Δ steer| {_f(t['steer_change_mean'], '{:.3f}')} per Tick, "
             f"wrong-way Ticks {t['wrong_way_ticks']}",
             f"- Passages: outer {t['passages']['outer']}, inner {t['passages']['inner']}, wrong {t['wrong_passages']}; "
             f"Island crashes {t['island_crashes']}"]
    ff = m.get("fork_flips")
    if ff:
        lines.append(f"- Fork flips: {ff['flips']} / {ff['approaches']} approaches (`any`: {ff['any_flips']} / "
                     f"{ff['any_approaches']}), uncommitted replans {_pct(ff['fraction_uncommitted'])}")
    if m.get("crash_list"):
        lines.append("- Crashes (seed, combination, Tick, Station, in fork section): " + "; ".join(
            f"{c['seed']} {c['combination']} t{c['tick']} s{c['station_m']:.1f}{' fork' if c['in_section'] else ''}"
            for c in m["crash_list"][:40]) + (" …" if len(m["crash_list"]) > 40 else ""))
    if m.get("push"):
        p = m["push"]
        lines.append(f"- Pushes: {p['pushes']}, crash within 3 s {_pct(p['crash_within_45_rate'])}, peak dist "
                     f"{_f(p['peak_dist_m_mean'])} m, recovered {_pct(p['recovered_fraction'])} "
                     f"(median {_f(p['recovery_ticks_median'], '{:.0f}')} Ticks)")
    if m.get("offset_start"):
        o = m["offset_start"]
        lines.append(f"- offset_start: crash rate where the expert survives "
                     f"{_pct(o.get('crash_rate_where_expert_survives'))} (n {o.get('expert_survived')}), "
                     f"reached the line {_pct(o['recovered_fraction'])} "
                     f"(median {_f(o['ticks_to_line_median'], '{:.0f}')} Ticks)")
    t_cost = f"- Cost: {_f(t['ms_per_replan'], '{:.1f}')} ms per replan, wall {t['wall_s']:.0f} s"
    return lines + [t_cost]


def print_console(row: Mapping, ref_row: Mapping | None) -> None:
    print(f"\n== {row['driver']} [{row['point']}] ==")
    print("\n".join(gate_table_md(row, ref_row)))
    print()
    print("\n".join(combo_table_md(row["metrics"])))
    if row.get("verdict"):
        print(f"\nverdict: {row['verdict']}")
    else:
        print(f"\nno verdict: {row['no_verdict_reason']}")


def report_md(cfg: Mapping, rows: Sequence[Mapping]) -> str:
    out = [f"# Closed-loop evaluation: {cfg['driver']} ({cfg['mode']})", "",
           f"Set `{cfg['set']}`, {cfg['episodes']} episodes per point, conditions {', '.join(cfg['conditions'])}; "
           + (f"Scheme `{cfg['scheme']}`" if cfg["scheme"] else "no Scheme (not a Policy)") + f"; created {cfg['created']}.",
           ""]
    refs = {(r["point"], r["condition"]): r for r in rows if r["role"] == "reference"}
    mains = [r for r in rows if r["role"] == "driver"]
    base = next((r for r in mains if r["condition"] == "none"), None)
    if cfg["mode"].startswith("sweep") or cfg["mode"] == "robustness":
        out += ["## Overview", "", "| Point | Crash | Frac. laps | Followed | Outer | Fork flips | ms/replan |"
                + (" Δ crash vs none | Δ frac. laps vs none |" if cfg["mode"] == "robustness" else ""),
                "|---|---|---|---|---|---|---|" + ("---|---|" if cfg["mode"] == "robustness" else "")]
        for r in mains + [x for x in rows if x["role"] == "reference"]:
            t, ff = r["metrics"]["total"], r["metrics"].get("fork_flips")
            line = (f"| {r['driver'] if r['role'] == 'reference' else ''} {r['point']} | {_pct(t['crash_rate'])} "
                    f"| {_f(t['frac_laps_mean'])} | {_pct(t['instruction_followed_rate'])} | {_f(t['outer_fraction'])} "
                    f"| {'n/a' if not ff else str(ff['flips']) + '/' + str(ff['approaches'])} "
                    f"| {_f(t['ms_per_replan'], '{:.1f}')} |")
            if cfg["mode"] == "robustness":
                b = base if r["role"] == "driver" else refs.get(("none", "none"))
                if b is not None:
                    bt = b["metrics"]["total"]
                    line += (f" {_f(100 * (t['crash_rate'] - bt['crash_rate']), '{:+.1f} pp')} "
                             f"| {_f(t['frac_laps_mean'] - bt['frac_laps_mean'], '{:+.2f}')} |")
                else:
                    line += " | |"
            out.append(line)
        out.append("")
    for r in rows:
        ref = refs.get((r["point"], r["condition"])) if r["role"] == "driver" else None
        if ref is None and r["role"] == "driver":
            ref = refs.get(("reference", r["condition"]))
        head = f"## {r['driver']} — {r['point']}" + (" (reference)" if r["role"] == "reference" else "")
        out += [head, ""]
        if r.get("verdict"):
            out += [f"**Verdict: {r['verdict']}**", ""]
        else:
            out += [f"No verdict: {r['no_verdict_reason']}" + (" (gates report-only)" if cfg["mode"] == "robustness"
                                                                 else ""), ""]
        out += gate_table_md(r, ref) + [""] + combo_table_md(r["metrics"]) + [""] + detail_md(r["metrics"]) + [""]
    return "\n".join(out)


# -- eval -----------------------------------------------------------------------------------
def _scheme_from(a) -> Scheme:
    kw = {}
    for f in dataclasses.fields(Scheme):
        v = getattr(a, f.name, None)
        if v is not None:
            kw[f.name] = v
    return replace(DEFAULT_SCHEME, **kw)


def _fork_ref(data: str, horizon: int) -> np.ndarray | None:
    from toycar.dataset import open_dataset

    from .evaluate import fork_references

    p = data_path(data)
    if not p.exists():
        print(f"note: {data} not found; Fork flips are not computed")
        return None
    return fork_references(open_dataset(p, check=False), horizon=horizon)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "driver"


def _progress(label: str):
    t0 = time.perf_counter()

    def cb(done: int, n: int) -> None:
        if done == n or done % max(1, n // 10) == 0:
            print(f"  {label}: {done}/{n} episodes, {time.perf_counter() - t0:.0f} s", flush=True)

    return cb


def cmd_eval(a) -> int:
    try:
        scheme = _scheme_from(a)
    except ValueError as exc:
        print(f"invalid Scheme: {exc}", file=sys.stderr)
        return 2
    if a.sweep and a.robustness is not None:
        print("--sweep and --robustness are mutually exclusive", file=sys.stderr)
        return 2
    conds = list(CONDITIONS) if a.robustness == [] else (a.robustness or ["none"])
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        print(f"unknown Stress condition(s) {bad}; known: {list(CONDITIONS)}", file=sys.stderr)
        return 2
    if a.sweep:
        mode, set_name, n = f"sweep:{a.sweep}", "dev", SWEEP_EPISODES
    elif a.robustness is not None:
        mode, set_name, n = "robustness", "robustness", ROBUSTNESS_EPISODES
    else:
        mode, set_name, n = "gate", "gate", a.episodes
    if a.episodes_override is not None:  # hidden: tests and smoke runs shorten sweeps / robustness
        n = a.episodes_override
    try:
        episode_specs(set_name, n)
    except ValueError as exc:
        print(f"--episodes: {exc}", file=sys.stderr)
        return 2
    device = a.device or default_device()
    try:
        driver, pol = make_driver(a.driver, scheme, device=device, root=a.root, batch=a.batch)
    except (ValueError, FileNotFoundError) as exc:
        print(f"cannot load driver {a.driver!r}: {exc}", file=sys.stderr)
        return 2
    is_policy = pol is not None
    if a.sweep and not is_policy:
        print("--sweep needs a Policy driver", file=sys.stderr)
        return 2
    if is_policy:
        scheme = driver.scheme
    if a.sweep:
        try:
            points = [(p, s, "none") for p, s in sweep_points(a.sweep, scheme, driver.horizon)]
        except ValueError as exc:
            print(f"invalid sweep: {exc}", file=sys.stderr)
            return 2
    elif mode == "robustness":
        points = [(c, scheme, c) for c in conds]
    else:
        points = [("default", scheme, "none")]

    dname = driver.name if is_policy else a.driver
    name = a.name or _safe(dname) + (f"-sweep-{a.sweep}" if a.sweep else "-robustness" if mode == "robustness" else "")
    run = _new_run_dir(a.runs, name)
    commit, dirty = _git_info()
    cfg = {"driver": dname, "driver_kind": "policy" if is_policy else ("saboteur" if a.driver.startswith("saboteur:")
                                                                         else "expert"),
           "target": a.driver if not is_policy else (pol.name or _rel(a.driver)),
           "checkpoint": pol.name if is_policy else None,
           "card_sha256": _sha256(Path(pol.source) / "card.json") if is_policy and pol.card is not None else None,
           "scheme": asdict(scheme) if is_policy else None, "mode": mode, "set": set_name, "episodes": n,
           "seeds": [s.seed for s in episode_specs(set_name, n)], "conditions": [p[2] for p in points],
           "points": [p[0] for p in points], "batch": a.batch, "device": device if is_policy else "cpu",
           "max_steps": a.max_steps, "git_commit": commit, "git_dirty": dirty, "created": _now(),
           "argv": [_rel(x) if (x.startswith("/") and Path(x).exists()) else x for x in a.argv]}
    if is_policy and a.max_steps != MAX_STEPS:
        cfg["note"] = "max_steps shortened"
    (run / "config.json").write_text(json.dumps(_jsonable(cfg), indent=2) + "\n")
    with open(run / "stdout.txt", "a") as so, contextlib.redirect_stdout(_Tee(sys.stdout, so)):
        print(f"closed-loop eval of {dname} ({mode}) → {_rel(run)}", flush=True)
        ref = _fork_ref(a.data, driver.horizon) if is_policy else None
        rows, all_records, lines = [], [], []

        def add(records, role, drv_name, point, condition, sch, verdict_ok):
            m = metrics(records, ref=ref if role == "driver" else None,
                        reference=next((rr for (pp, cc, rr) in ref_cache if cc == condition), None)
                        if role == "driver" else None)
            g = gates(m["total"])
            if verdict_ok:
                v, reason = verdict(g, driver=drv_name, is_policy=is_policy, scheme=sch, set_name=set_name, n=n,
                                    condition=condition, mode=mode)
            else:
                v, reason = None, "not gated"
            row = {"driver": drv_name, "role": role, "point": point, "condition": condition,
                   "scheme": asdict(sch) if sch is not None else None, "metrics": m, "gates": g,
                   **({"verdict": v} if v else {"no_verdict_reason": reason})}
            rows.append(row)
            for k, r in enumerate(records):
                lines.append(episode_line(r, index=len(all_records), driver=drv_name, role=role, point=point, k=k))
                all_records.append(r)
            return row

        ref_cache: list = []  # (point, condition, reference records)
        expert_driver = driver if a.driver == "expert" else ExpertDriver()
        for point, sch, condition in points:
            specs = episode_specs(set_name, n, condition)
            if not any(cc == condition for (_, cc, _) in ref_cache):
                rpoint = "reference" if a.sweep else point
                recs = run_episodes(expert_driver, specs, shadow_expert=False, max_steps=a.max_steps)
                ref_cache.append((rpoint, condition, recs))
            if is_policy:
                driver = PolicyDriver(pol, sch, batch=a.batch, name=dname)
                print(f"{point}: {n} episodes, Scheme {sch}", flush=True)
                recs = run_episodes(driver, specs, batch=a.batch, max_steps=a.max_steps, progress=_progress(point))
            elif a.driver == "expert":
                recs = next(rr for (_, cc, rr) in ref_cache if cc == condition)
            else:
                recs = run_episodes(driver, specs, max_steps=a.max_steps)
            row = add(recs, "driver", dname, point, condition, sch if is_policy else None, True)
            rref = next(r for (_, cc, r) in ref_cache if cc == condition)
            rp = next(p for (p, cc, _) in ref_cache if cc == condition)
            if not any(x["role"] == "reference" and x["condition"] == condition for x in rows):
                ref_row = add(rref, "reference", "expert", rp, condition, None, False)
            else:
                ref_row = next(x for x in rows if x["role"] == "reference" and x["condition"] == condition)
            print_console(row, ref_row)
        # expert reference rows after their driver rows; keep the file order: driver rows then references
        with open(run / "episodes.jsonl", "w") as f:
            for ln in lines:
                f.write(json.dumps(_jsonable(ln)) + "\n")
        save_ticks(run / "ticks.npz", all_records)
        out = {"driver": dname, "mode": mode, "rows": rows,
               "verdict": next((r["verdict"] for r in rows if r.get("verdict")), None)}
        if out["verdict"] is None:
            out["no_verdict_reason"] = next(r["no_verdict_reason"] for r in rows if r["role"] == "driver")
        (run / "metrics.json").write_text(json.dumps(_jsonable(out), indent=1, allow_nan=False) + "\n")
        (run / "report.md").write_text(report_md(cfg, rows))
        if a.json:
            Path(a.json).write_text(json.dumps(_jsonable(out), indent=1, allow_nan=False) + "\n")
        print(f"\nwrote {_rel(run)}")
    return 1 if out["verdict"] == "FAIL" else 0


# -- video (SPEC §10) -----------------------------------------------------------------------
def write_png(path: Path, img: np.ndarray) -> None:
    """Minimal RGB PNG writer (no extra dependency)."""
    img = np.ascontiguousarray(img, np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def load_episode(run_dir: Path, episode: int, point: str | None = None, role: str = "driver") -> tuple[dict, dict]:
    """(episodes.jsonl line, per-Tick arrays) of episode k = `episode` of the Driver row at `point`."""
    lines = [json.loads(x) for x in (Path(run_dir) / "episodes.jsonl").read_text().splitlines() if x.strip()]
    rows = [ln for ln in lines if ln["role"] == role]
    if point is None and rows:
        point = rows[0]["point"]
    hit = [ln for ln in rows if ln["point"] == point and ln["k"] == episode]
    if not hit:
        points = sorted({ln["point"] for ln in rows})
        raise ValueError(f"no episode {episode} at point {point!r}; points: {points}")
    ln = hit[0]
    z = np.load(Path(run_dir) / "ticks.npz")
    i = ln["index"]
    so, to = int(z["offsets_state"][i]), int(z["offsets"][i])
    T = ln["steps"]
    arr = {f: z[f][so:so + T + 1] for f in STATE_FIELDS}
    arr.update({f: z[f][to:to + T] for f in TICK_FIELDS})
    return ln, arr


def rerender(run_dir: Path, episode: int, point: str | None = None, resolution: int = 64) -> np.ndarray:
    """The Frames of a recorded episode, re-rendered from its States: uint8 (steps + 1, res, res, 3)."""
    from toycar.render import render
    from toycar.state import State

    ln, arr = load_episode(run_dir, episode, point)
    spec = EpisodeSpec(**ln["spec"])
    track, rp = condition_track(spec.condition), condition_render_params(spec)
    return np.stack([render(State(float(x), float(y), float(t), float(v)), track, rp, resolution)
                     for x, y, t, v in zip(arr["x"], arr["y"], arr["theta"], arr["v"])])


def trajectory_png(path: Path, ln: Mapping, arr: Mapping) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    spec = EpisodeSpec(**ln["spec"])
    track = condition_track(spec.condition)
    fig, ax = plt.subplots(figsize=(6, 6))
    for b in ("outer", "inner"):
        route = track.route(b)
        p = np.array([route.pose_at(s)[:2] for s in np.linspace(0, route.length_m, 800)])
        ax.plot(p[:, 0], p[:, 1], color="0.85", lw=14, solid_capstyle="round")
    ax.plot(arr["x"], arr["y"], lw=1.2, color="tab:blue")
    ax.plot(arr["x"][0], arr["y"][0], "o", color="tab:green")
    if ln["crash_tick"] is not None:
        ax.plot(arr["x"][-1], arr["y"][-1], "x", color="tab:red", ms=12, mew=3)
    ax.set_xlim(0, 40), ax.set_ylim(40, 0), ax.set_aspect("equal")
    ax.set_title(f"{ln['driver']} {ln['point']} k={ln['k']}: seed {spec.seed} {spec.instruction}/{spec.direction} "
                 f"{spec.condition}\nfrac. laps {ln['fractional_laps']:.2f}"
                 + (f", crash at Tick {ln['crash_tick']}" if ln["crash_tick"] is not None else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return True


def cmd_video(a) -> int:
    run = Path(a.run_dir)
    if not (run / "episodes.jsonl").exists():
        print(f"{run} is not a closed-loop Run directory", file=sys.stderr)
        return 2
    try:
        ln, arr = load_episode(run, a.episode, a.point)
        frames = rerender(run, a.episode, a.point, 512 if a.hires else 64)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else run / "video" / str(a.episode)
    out.mkdir(parents=True, exist_ok=True)
    for t, f in enumerate(frames):
        write_png(out / f"frame_{t:05d}.png", f)
    if not trajectory_png(out / "trajectory.png", ln, arr):
        print("note: trajectory.png needs matplotlib (uv run --group viz …)")
    print(f"{len(frames)} Frames → {_rel(out)}")
    return 0


# -- show (SPEC §10) ------------------------------------------------------------------------
def cmd_show(a) -> int:  # pragma: no cover - interactive
    import os

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    from toycar.drive import PANEL_W, VIEW, frame_surface
    from toycar.render import render

    from .drive import Push, reset_env

    try:
        scheme = _scheme_from(a)
        driver, pol = make_driver(a.driver, scheme, device=a.device or default_device(), root=a.root, batch=1,
                                  **({} if a.driver == "expert" or a.driver.startswith("saboteur:")
                                     else {"keep_candidates": True, "imagine": a.imagine}))
    except (ValueError, FileNotFoundError) as exc:
        print(f"cannot load driver {a.driver!r}: {exc}", file=sys.stderr)
        return 2
    first, count = SETS[a.set]
    strip = 64 if (a.imagine and pol is not None and pol.config.video_stream) else 0
    pygame.init()
    W, Hh = VIEW + PANEL_W + 40, VIEW + strip
    screen = pygame.display.set_mode((W, Hh))
    pygame.display.set_caption(f"closed loop: {driver.name}")
    font = pygame.font.SysFont("menlo,monaco,couriernew,monospace", 14)
    clock = pygame.time.Clock()
    def matches(k: int) -> bool:
        ins, d = COMBINATIONS[k % 6]
        return (a.instruction is None or ins == a.instruction) and (a.direction is None or d == a.direction)

    def next_k(k: int) -> int:  # the first episode ≥ k that plays the chosen Instruction / Direction
        while not matches(k % count):
            k += 1
        return k % count

    k = next_k(a.k)

    def start(k):
        spec = EpisodeSpec(first + k, *COMBINATIONS[k % 6], condition=a.condition)
        env, s, i = reset_env(spec)
        driver.reset([spec], [env], [s], [i])
        return spec, env, s, i, Push(spec)

    spec, env, s, i, push = start(k)
    paused, ended = False, False
    imagined = None
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return 0
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                    k = next_k(k + 1)
                    spec, env, s, i, push = start(k)
                    ended, imagined = False, None
            if not paused and not ended:
                frame = env.render()[None] if driver.needs_frames else None
                act = driver.act([0], frame, [s], [i])[0]
                res = env.step(tuple(float(v) for v in push(i.steps, act)))
                s, i = res.state, res.info
                ended = res.terminated or res.truncated
                lat = getattr(driver, "last_latents", {}).get(0)
                if lat is not None:
                    imagined = pol.compressor.decode(lat[None])[0].cpu().numpy()
            screen.fill((24, 24, 28))
            screen.blit(frame_surface(render(s, env.track, env.render_params, VIEW)), (0, 0))
            lines = [f"seed  {spec.seed}  k {k}", f"instr {spec.instruction} {spec.direction}",
                     f"cond  {spec.condition}", f"tick  {i.steps}", f"laps  {i.progress_m / env.track.length_m:.2f}",
                     f"off   {env.track.distance_to_centerline(s.x, s.y):.2f} m", f"v     {s.v:.2f} m/s",
                     "CRASH" if i.termination == "crash" else ("END" if ended else ""),
                     "PAUSED" if paused else "", "", "Space pause  N next", "Esc/Q quit"]
            for n_, txt in enumerate(lines):
                screen.blit(font.render(txt, True, (220, 220, 220)), (VIEW + 10, 10 + 18 * n_))
            cands = getattr(driver, "last_candidates", {}).get(0)
            diag = getattr(driver, "last_diag", {})
            box = pygame.Rect(VIEW + 10, 250, PANEL_W + 20, 200)
            pygame.draw.rect(screen, (40, 40, 46), box)
            pygame.draw.line(screen, (90, 90, 90), (box.left, box.centery), (box.right, box.centery))
            if cands is not None:
                H = cands.shape[1]
                pts = lambda st: [(float(box.left + box.width * t / (H - 1)), float(box.centery - st[t] * box.height / 2))  # noqa: E731
                                  for t in range(H)]
                sel = int(diag["selected"][0]) if "selected" in diag else 0
                for c_i, c in enumerate(cands):
                    if c_i != sel:
                        pygame.draw.lines(screen, (120, 120, 120), False, pts(c[:, 0]), 1)
                pygame.draw.lines(screen, (255, 200, 60), False, pts(cands[sel, :, 0]), 2)
                screen.blit(font.render("Chunk steer (selected / others)", True, (200, 200, 200)),
                            (box.left, box.bottom + 4))
            if strip and imagined is not None:
                for f_i, f in enumerate(imagined[: VIEW // 64]):
                    screen.blit(frame_surface(f) if f.shape[0] == VIEW else pygame.surfarray.make_surface(
                        f.swapaxes(0, 1)), (f_i * 64, VIEW))
            pygame.display.flip()
            clock.tick(a.speed * FPS)
    finally:
        pygame.quit()


# -- CLI ------------------------------------------------------------------------------------
def _scheme_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--execute-k", type=int)
    p.add_argument("--n-samples", type=int)
    p.add_argument("--n-steps", type=int)
    p.add_argument("--cfg-action", type=float)
    p.add_argument("--cfg-video", type=float)
    p.add_argument("--select", choices=["first", "coherence"])
    p.add_argument("--rho", type=float)
    p.add_argument("--seed-mode", choices=["per_replan", "per_episode"])


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m wam.closedloop", description="Closed-loop evaluation (Step 9).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    hide = argparse.SUPPRESS

    ev = sub.add_parser("eval", help="run a Driver and the expert reference; gate table and verdict")
    ev.add_argument("driver", help="Checkpoint name, Run directory, `expert` or `saboteur:<name>`")
    ev.add_argument("--name")
    ev.add_argument("--episodes", type=int, default=GATE_EPISODES)
    _scheme_args(ev)
    g = ev.add_mutually_exclusive_group()
    g.add_argument("--sweep", choices=list(SWEEPS))
    g.add_argument("--robustness", nargs="*", metavar="COND")
    ev.add_argument("--batch", type=int, default=24)
    ev.add_argument("--device", choices=["mps", "cpu"])
    ev.add_argument("--json", type=Path, metavar="PATH")
    ev.add_argument("--data", default=DEFAULT_DATA, help=hide)
    ev.add_argument("--root", type=Path, default=CHECKPOINTS, help=hide)
    ev.add_argument("--runs", type=Path, default=RUNS_DIR, help=hide)
    ev.add_argument("--max-steps", type=int, default=MAX_STEPS, help=hide)
    ev.add_argument("--episodes-override", type=int, help=hide)

    sh = sub.add_parser("show", help="watch one live episode (pygame)")
    sh.add_argument("driver")
    sh.add_argument("--set", choices=list(SETS), default="gate")
    sh.add_argument("--k", type=int, default=0)
    sh.add_argument("--instruction", choices=["outer", "inner", "any"], help="only episodes with this Instruction")
    sh.add_argument("--direction", choices=["cw", "ccw"], help="only episodes with this Direction")
    sh.add_argument("--condition", choices=list(CONDITIONS), default="none")
    sh.add_argument("--imagine", action="store_true")
    sh.add_argument("--speed", type=float, default=1.0)
    sh.add_argument("--device", choices=["mps", "cpu"])
    sh.add_argument("--root", type=Path, default=CHECKPOINTS, help=hide)
    _scheme_args(sh)

    vi = sub.add_parser("video", help="re-render a recorded episode to PNG Frames and a trajectory plot")
    vi.add_argument("run_dir", type=Path)
    vi.add_argument("--episode", type=int, required=True)
    vi.add_argument("--point")
    vi.add_argument("--out", type=Path)
    vi.add_argument("--hires", action="store_true")

    try:
        a = ap.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code not in (0, None) else 0
    a.argv = argv
    if a.cmd == "eval" and a.driver.startswith("saboteur:") and a.driver.split(":", 1)[1] not in SABOTEURS:
        print(f"unknown Saboteur {a.driver!r}; known: {[f'saboteur:{s}' for s in SABOTEURS]}", file=sys.stderr)
        return 2
    return {"eval": cmd_eval, "show": cmd_show, "video": cmd_video}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
