import math
from pathlib import Path

import numpy as np

from toycar import A_LAT_MAX, R_MIN, State, step_state

GOLDEN = Path(__file__).parent / "golden" / "physics_trajectory.npz"


def test_physics_golden():
    z = np.load(GOLDEN)
    actions, expected = z["actions"], z["states"]
    assert actions.shape == (300, 2) and expected.shape == (301, 4)
    s = State(20.0, 27.5, math.pi, 5.0)
    got = [[s.x, s.y, s.theta, s.v]]
    for a in actions:
        s = step_state(s, a)
        got.append([s.x, s.y, s.theta, s.v])
    np.testing.assert_allclose(np.array(got), expected, rtol=0, atol=1e-9)


def _turn_radius(v: float, steer: float) -> float:
    s = State(0.0, 0.0, 0.0, v)
    pts = []
    for _ in range(40):
        s = step_state(s, (steer, 0.0))
        pts.append((s.x, s.y))
    (x1, y1), (x2, y2), (x3, y3) = pts[0], pts[10], pts[20]
    # circumradius of three points on the path
    a, b, c = math.dist((x1, y1), (x2, y2)), math.dist((x2, y2), (x3, y3)), math.dist((x1, y1), (x3, y3))
    area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
    return a * b * c / (4 * area)


def test_physics_properties():
    rng = np.random.default_rng(1)
    s = State(0.0, 0.0, 0.0, 0.0)
    for a in rng.uniform(-1.5, 1.5, (2000, 2)):
        s = step_state(s, a)
        assert 0.0 <= s.v <= 10.0
        assert -math.pi < s.theta <= math.pi

    # throttle 0 keeps speed; steer 0 is a straight line
    s0 = State(1.0, 2.0, 0.3, 4.0)
    s = s0
    for _ in range(50):
        s = step_state(s, (0.0, 0.0))
    assert s.v == 4.0 and s.theta == s0.theta
    assert math.isclose(s.x - s0.x, 50 * 4.0 / 15 * math.cos(0.3), rel_tol=1e-12)
    assert math.isclose(s.y - s0.y, 50 * 4.0 / 15 * math.sin(0.3), rel_tol=1e-12)

    # lateral-acceleration clamp at 10 m/s; R_MIN at low speed
    assert math.isclose(_turn_radius(10.0, 1.0), 10.0**2 / A_LAT_MAX, rel_tol=1e-9)
    assert math.isclose(_turn_radius(10.0, -1.0), 12.5, rel_tol=1e-9)
    assert math.isclose(_turn_radius(2.0, 1.0), R_MIN, rel_tol=1e-9)

    # braking never reverses
    s = State(0.0, 0.0, 0.0, 1.0)
    for _ in range(20):
        s = step_state(s, (0.0, -1.0))
        assert s.v >= 0.0
    assert s.v == 0.0
    x = s.x
    s = step_state(s, (0.5, -1.0))
    assert s.x == x and s.v == 0.0
