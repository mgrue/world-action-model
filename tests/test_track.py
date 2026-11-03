import math

import numpy as np
import pytest

from toycar import CAR_LENGTH, State, Track, TrackParams


def test_track_geometry(track):
    assert math.isclose(track.length_m, 88.0560, abs_tol=1e-4)
    assert track.outer.length_m == pytest.approx(track.inner.length_m, abs=1e-9)
    assert track.merge_station_m == pytest.approx(22.9321, abs=1e-4)
    assert track.fork_station_m == 0.0

    for route in (track.outer, track.inner):
        x0, y0, th0 = route.pose_at(0.0)
        x1, y1, th1 = route.pose_at(route.length_m)
        assert (x0, y0, th0) == pytest.approx((11.0, 12.5, 0.0), abs=1e-12)
        assert (x1, y1) == pytest.approx((x0, y0), abs=1e-9)
        for s in np.linspace(0.0, route.length_m, 997, endpoint=False):
            x, y, _ = route.pose_at(s)
            p = route.project(x, y)
            assert p.station_m == pytest.approx(s, abs=1e-9)
            assert p.lateral_m == pytest.approx(0.0, abs=1e-9)
            assert p.distance_m == pytest.approx(0.0, abs=1e-9)

        # lateral sign: + is right of the cw tangent
        x, y, th = route.pose_at(50.0)
        p = route.project(x - 1.5 * math.sin(th), y + 1.5 * math.cos(th))
        assert p.lateral_m == pytest.approx(1.5) and p.distance_m == pytest.approx(1.5)

    rho = 4.875
    m = track.merge_station_m
    q = m / 4
    assert track.outer.curvature_at(q / 2) == pytest.approx(-1 / rho)
    assert track.outer.curvature_at(m / 2) == pytest.approx(1 / rho)
    assert track.outer.curvature_at(m - q / 2) == pytest.approx(-1 / rho)
    assert track.inner.curvature_at(q / 2) == pytest.approx(1 / rho)
    assert track.inner.curvature_at(m / 2) == pytest.approx(-1 / rho)
    end = 7.5 * math.pi
    for route in (track.outer, track.inner):
        assert route.curvature_at(m + end / 2) == pytest.approx(1 / 7.5)
        assert route.curvature_at(m + end + 9.0) == 0.0
        assert route.curvature_at(m + end + 18.0 + end / 2) == pytest.approx(1 / 7.5)
        assert route.pose_at(m + end + 9.0)[1] == pytest.approx(27.5)

    assert track.outer.pose_at(m / 2)[:2] == pytest.approx((20.0, 6.5))
    assert track.inner.pose_at(m / 2)[:2] == pytest.approx((20.0, 18.5))

    with pytest.raises(ValueError):
        Track(TrackParams(bulge_m=12.0))
    with pytest.raises(ValueError):
        Track(TrackParams(fork_length_m=16.0))


def test_locate(track):
    s, b = track.locate(20.0, 6.5)
    assert b == "outer" and s == pytest.approx(track.merge_station_m / 2)
    s, b = track.locate(20.0, 18.5)
    assert b == "inner" and s == pytest.approx(track.merge_station_m / 2)
    s, b = track.locate(20.0, 27.5)
    assert b is None
    assert track.distance_to_centerline(20.0, 6.5) == pytest.approx(0.0, abs=1e-12)


def test_crash_predicate(track):
    for route in (track.outer, track.inner):
        for s in np.linspace(0, track.length_m, 400, endpoint=False):
            x, y, th = route.pose_at(s)
            assert not track.is_crash(State(x, y, th, 0.0)), (route.name, s)
    # bottom straight, heading π (cw); right normal points to −y... use explicit offsets both ways
    assert track.is_crash(State(20.0, 27.5 + 1.6, math.pi, 0.0))
    assert track.is_crash(State(20.0, 27.5 - 1.6, math.pi, 0.0))
    assert not track.is_crash(State(20.0, 27.5 + 1.5, math.pi, 0.0))
    assert track.is_crash(State(20.0, 12.5, 0.0, 0.0))  # Island centre
    assert CAR_LENGTH == 4.4
