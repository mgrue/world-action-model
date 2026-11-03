"""Step 2 SPEC §9: scripted expert."""

import dataclasses
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from toycar import CarEnv, Expert, ExpertParams, Track
from toycar.expert import (
    A_ACC_PLAN,
    A_BRK_PLAN,
    INSTRUCTIONS,
    PROFILE_DS,
    _profile,
    main,
    profile_speed,
    run_episode,
    start_spec,
)

COMBOS = [(i, d) for i in INSTRUCTIONS for d in ("cw", "ccw")]
CLEAN = ExpertParams(burst_peak=0.0)
NOISY = ExpertParams()


@pytest.fixture(scope="module")
def pool():
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as p:
        yield p


def _job(job):
    return run_episode(*job)


def run_many(pool, jobs):
    return list(pool.map(_job, jobs, chunksize=8))


def episode_actions(seed, instruction, direction, params):
    env = CarEnv()
    state, info = env.reset(seed, start_spec(instruction, direction))
    ex = Expert(env.track, instruction, params)
    ex.reset(seed, state, info)
    actions, states = [], [state]
    while True:
        a = ex.act(state, info)
        actions.append(a)
        r = env.step(a.executed)
        state, info = r.state, r.info
        states.append(state)
        if r.terminated or r.truncated:
            return actions, states, ex


# -- reliability --------------------------------------------------------------------------------
@pytest.mark.parametrize("params", [CLEAN, NOISY], ids=["clean", "noise"])
def test_reliability(pool, params):
    jobs = [(seed, i, d, params) for i, d in COMBOS for seed in range(100)]
    for (seed, i, d, _), (sc, switches) in zip(jobs, run_many(pool, jobs)):
        where = f"seed {seed} {i} {d}"
        assert not sc.crashed, where
        assert sc.correct == sc.scored, where
        assert sc.instruction_followed, where
        assert sc.laps >= 4, where
        assert switches == 0, where


REGRESSION = [(838, "any", "cw"), (285, "any", "ccw"), (3, "outer", "cw"), (3, "outer", "ccw"),
              (26, "inner", "ccw"), (81, "any", "cw"), (938, "any", "cw")]


@pytest.mark.parametrize("params", [CLEAN, NOISY], ids=["clean", "noise"])
@pytest.mark.parametrize("seed,instruction,direction", REGRESSION)
def test_regression_starts(seed, instruction, direction, params):
    sc, _ = run_episode(seed, instruction, direction, params)
    assert not sc.crashed
    assert sc.instruction_followed


# -- determinism and random streams -------------------------------------------------------------
def test_determinism(pool):
    for params in (CLEAN, NOISY):
        a1, s1, _ = episode_actions(7, "any", "ccw", params)
        a2, s2, _ = episode_actions(7, "any", "ccw", params)
        assert a1 == a2
        assert s1 == s2
    jobs_clean = [(seed, "any", d, CLEAN) for seed in range(20) for d in ("cw",)]
    jobs_noisy = [(seed, "any", d, NOISY) for seed in range(20) for d in ("cw",)]
    for (c, _), (n, _) in zip(run_many(pool, jobs_clean), run_many(pool, jobs_noisy)):
        assert c.passages == n.passages


def test_branch_split(pool):
    def outer_fraction(params):
        jobs = [(seed, "any", d, params) for seed in range(300) for d in ("cw", "ccw")]
        branches = [b for sc, _ in run_many(pool, jobs) for b in sc.scored_passages]
        return sum(b == "outer" for b in branches) / len(branches)

    assert 0.45 <= outer_fraction(CLEAN) <= 0.55
    assert 0.65 <= outer_fraction(dataclasses.replace(CLEAN, p_outer=0.7)) <= 0.75
    assert outer_fraction(dataclasses.replace(CLEAN, p_outer=1.0)) == 1.0


# -- noise ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_noise_shape(seed):
    actions, _, _ = episode_actions(seed, "any", "cw", NOISY)
    pert = [a.perturbed for a in actions]
    assert not any(pert[:30])
    assert any(pert)
    for a in actions:
        assert a.executed[1] == a.clean[1]
        if not a.perturbed:
            assert a.executed == a.clean
    runs, k = [], 0
    for p in pert + [False]:
        if p:
            k += 1
        elif k:
            runs.append(k)
            k = 0
    if pert[-1]:
        runs.pop()  # the last run may be cut by the episode end
    assert min(runs) >= 8


def test_no_noise_draws_when_clean():
    seed = 4
    actions, _, ex = episode_actions(seed, "any", "cw", CLEAN)
    assert not any(a.perturbed for a in actions)
    assert all(a.executed == a.clean for a in actions)
    fresh = np.random.default_rng([seed, 2])
    assert ex._noise_rng.bit_generator.state == fresh.bit_generator.state


# -- API -------------------------------------------------------------------------------------------
def test_control_pure(track):
    env = CarEnv(track)
    state, info = env.reset(11, start_spec("any", "cw"))
    ex = Expert(track, "any", NOISY)
    ex.reset(11, state, info)
    for _ in range(40):
        a = ex.act(state, info)
        r = env.step(a.executed)
        state, info = r.state, r.info
    snap = (ex.planned_branch, ex.safety_switches, ex._coin_rng.bit_generator.state, ex._noise_rng.bit_generator.state)
    outs = [ex.control(state, b, d) for _ in range(3) for b in ("outer", "inner") for d in ("cw", "ccw")]
    assert outs[:4] == outs[4:8] == outs[8:]
    assert snap == (ex.planned_branch, ex.safety_switches, ex._coin_rng.bit_generator.state, ex._noise_rng.bit_generator.state)


def test_act_before_reset(track):
    env = CarEnv(track)
    state, info = env.reset(0)
    with pytest.raises(RuntimeError):
        Expert(track, "outer").act(state, info)


@pytest.mark.parametrize("bad", [dict(a_lat_plan=0.0), dict(a_lat_plan=8.0), dict(v_cap=10.5), dict(v_cap=0.0),
                                 dict(burst_peak=-0.1), dict(burst_peak=1.1), dict(burst_gap_s=0.0),
                                 dict(p_outer=1.5)])
def test_params_validated(track, bad):
    with pytest.raises(ValueError):
        Expert(track, "any", ExpertParams(**bad))


def test_profile(track):
    merge = track.merge_station_m
    rho = 1.0 / abs(track.outer.curvature_at(merge / 2))
    end_mid = merge + 7.5 * math.pi / 2
    for b in ("outer", "inner"):
        for d in ("cw", "ccw"):
            assert profile_speed(track, b, d, merge / 2) == pytest.approx(math.sqrt(5.0 * rho), abs=1e-9)
            assert profile_speed(track, b, d, end_mid) == pytest.approx(math.sqrt(5.0 * 7.5), abs=1e-9)
            v, _ = _profile(track, b, 1 if d == "cw" else -1, 5.0, 10.0)
            assert max(v) <= 10.0
            for j in range(len(v)):
                p = v[j - 1]
                assert v[j] ** 2 <= p ** 2 + 2 * A_ACC_PLAN * PROFILE_DS + 1e-9
                assert p ** 2 <= v[j] ** 2 + 2 * A_BRK_PLAN * PROFILE_DS + 1e-9
    assert math.sqrt(5.0 * rho) == pytest.approx(4.94, abs=0.01)


def test_cli_eval(capsys):
    assert main(["eval", "--n", "3"]) == 0
    assert "PASS" in capsys.readouterr().out
