import hashlib
import math

import numpy as np
import pytest
from conftest import pursuit_action

from toycar import CarEnv, StartSpec, State


def test_env_determinism():
    actions = np.random.default_rng(7).uniform(-1, 1, (500, 2))
    runs = []
    for _ in range(2):
        env = CarEnv(max_steps=10**6)
        s, _ = env.reset(123)
        seq = [(s, hashlib.sha256(env.render().tobytes()).hexdigest())]
        for a in actions:
            r = env.step(a)
            seq.append((r.state, hashlib.sha256(env.render().tobytes()).hexdigest()))
            if r.terminated:
                break
        runs.append(seq)
    assert runs[0] == runs[1]


def test_start_pose_validity():
    env = CarEnv()
    track = env.track
    for seed in range(10000):
        s, info = env.reset(seed)
        assert not track.is_crash(s)
        assert 0.0 <= s.v <= 6.0
        assert info.steps == 0 and info.progress_m == 0.0 and info.termination is None
    spec = StartSpec(direction="ccw", route="inner", station_m=10.0, lateral_m=0.0, heading_offset=0.0, speed=3.0)
    s, info = env.reset(5, spec)
    x, y, th = track.inner.pose_at(10.0)
    assert (s.x, s.y, s.v) == pytest.approx((x, y, 3.0))
    assert math.cos(s.theta - th) == pytest.approx(-1.0)
    assert info.direction == "ccw" and info.branch == "inner"
    # overriding speed doesn't change the sampled station
    s1, i1 = env.reset(42)
    s2, i2 = env.reset(42, StartSpec(speed=0.0))
    if not track.is_crash(State(s1.x, s1.y, s1.theta, 0.0)):
        assert (s1.x, s1.y, s1.theta) == (s2.x, s2.y, s2.theta) and s2.v == 0.0


def _drive(env, route, direction, distance):
    L = env.track.length_m
    info = None
    while True:
        r = env.step(pursuit_action(env, route, direction))
        assert not r.terminated, r.info
        info = r.info
        assert not info.wrong_way
        if info.progress_m >= distance * L:
            return info


@pytest.mark.parametrize("route,direction", [("outer", "cw"), ("inner", "ccw"), ("inner", "cw"), ("outer", "ccw")])
def test_progress_and_laps(route, direction):
    env = CarEnv(max_steps=10**6)
    env.reset(0, StartSpec(direction=direction, route=route, station_m=50.0, lateral_m=0.0,
                           heading_offset=0.0, speed=5.0))
    info = _drive(env, route, direction, 2.5)
    assert info.laps == 2
    assert info.last_branch == route
    assert info.direction == direction


def test_wrong_way():
    env = CarEnv(max_steps=10**6)
    env.reset(0, StartSpec(direction="cw", route="outer", station_m=50.0, lateral_m=0.0,
                           heading_offset=math.pi, speed=3.0))
    seen = False
    for _ in range(200):
        r = env.step(pursuit_action(env, "outer", "ccw", speed=3.0))
        assert not r.terminated
        if r.info.progress_m < -5.0:
            seen = r.info.wrong_way
            break
    assert seen


def test_wrong_way_fork_passage_sets_last_branch():
    # cw episode, car pointed ccw just past the Merge: it passes the inner branch the wrong way
    env = CarEnv(max_steps=10**6)
    _, info = env.reset(0, StartSpec(direction="cw", route="inner", station_m=30.0, lateral_m=0.0,
                                     heading_offset=math.pi, speed=5.0))
    assert info.branch is None and info.last_branch is None
    branches = set()
    for _ in range(150):
        r = env.step(pursuit_action(env, "inner", "ccw"))
        assert not r.terminated
        branches.add(r.info.branch)
        if r.info.station_m > 60.0:  # wrapped past the Fork
            break
    assert "inner" in branches and r.info.branch is None
    assert r.info.last_branch == "inner" and r.info.wrong_way


def test_episode_end():
    env = CarEnv(max_steps=5)
    env.reset(0, StartSpec(speed=0.0))
    for k in range(1, 6):
        r = env.step((0.0, 0.0))
        assert r.truncated == (k == 5) and not r.terminated
    assert r.info.termination == "max_steps" and r.info.steps == 5
    with pytest.raises(RuntimeError):
        env.step((0.0, 0.0))

    env = CarEnv()
    env.reset(0, StartSpec(direction="cw", route="outer", station_m=60.0, lateral_m=0.0,
                           heading_offset=0.0, speed=10.0))
    for _ in range(200):
        r = env.step((1.0, 1.0))
        if r.terminated:
            break
    assert r.terminated and not r.truncated and r.info.termination == "crash"
    assert env.track.is_crash(r.state)
    with pytest.raises(RuntimeError):
        env.step((0.0, 0.0))
