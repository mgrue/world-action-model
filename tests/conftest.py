import math

import pytest

from toycar import DELTA_MAX, WHEELBASE, CarEnv, Track, wrap


@pytest.fixture(scope="session")
def track() -> Track:
    return Track.default()


def pursuit_action(env: CarEnv, route: str, direction: str, speed: float = 5.0, lookahead: float = 4.0):
    """Simple pure-pursuit controller along a Route, for tests only."""
    s = env.state
    r = env.track.route(route)
    st = r.project(s.x, s.y).station_m
    tx, ty, _ = r.pose_at(st + (lookahead if direction == "cw" else -lookahead))
    alpha = wrap(math.atan2(ty - s.y, tx - s.x) - s.theta)
    kappa = 2.0 * math.sin(alpha) / lookahead
    steer = max(-1.0, min(1.0, math.atan(kappa * WHEELBASE) / DELTA_MAX))
    throttle = max(-1.0, min(1.0, (speed - s.v) * 2.0))
    return steer, throttle
