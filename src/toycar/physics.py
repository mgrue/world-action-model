"""Kinematic bicycle physics (SPEC §4). Python float + math only — never numpy — for determinism."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .state import State

DT = 1 / 15  # s, 15 Hz; one Tick = one Action, one State update, one Frame
WHEELBASE = 2.7  # m
DELTA_MAX = math.radians(35)
R_MIN = WHEELBASE / math.tan(DELTA_MAX)  # ≈ 3.856 m, minimum turning radius
A_LAT_MAX = 8.0  # m/s²
ACCEL = 4.0  # m/s² at throttle +1
BRAKE = 8.0  # m/s² at throttle −1
V_MAX = 10.0  # m/s
CAR_LENGTH, CAR_WIDTH = 4.4, 2.5  # m

_TWO_PI = 2.0 * math.pi
_TAN_DELTA_MAX = math.tan(DELTA_MAX)


def wrap(a: float) -> float:
    """Wrap an angle into (−π, π]."""
    return a - _TWO_PI * math.ceil((a - math.pi) / _TWO_PI)


def step_state(state: State, action: Sequence[float]) -> State:
    """Advance one Tick. Pure; the operation order is part of the determinism contract."""
    steer = min(max(float(action[0]), -1.0), 1.0)
    throttle = min(max(float(action[1]), -1.0), 1.0)

    a = ACCEL * throttle if throttle >= 0 else BRAKE * throttle
    v1 = min(max(state.v + a * DT, 0.0), V_MAX)

    t = math.tan(steer * DELTA_MAX)
    if v1 > 0:
        T = min(_TAN_DELTA_MAX, A_LAT_MAX * WHEELBASE / (v1 * v1))
        t = min(max(t, -T), T)

    k = t / WHEELBASE
    theta = state.theta
    th1 = theta + v1 * k * DT
    if abs(k) > 1e-9:
        x1 = state.x + (math.sin(th1) - math.sin(theta)) / k
        y1 = state.y - (math.cos(th1) - math.cos(theta)) / k
    else:
        x1 = state.x + v1 * DT * math.cos(theta)
        y1 = state.y + v1 * DT * math.sin(theta)

    return State(x1, y1, wrap(th1), v1)
