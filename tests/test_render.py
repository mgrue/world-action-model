import math
import time

import numpy as np

from toycar import RenderParams, State, render


def _far_state():
    return State(-100.0, -100.0, 0.0, 0.0)  # off-frame: background only


def test_render(track):
    p = RenderParams()
    s = State(20.0, 27.5, 0.0, 0.0)
    f = render(s, track)
    assert f.shape == (64, 64, 3) and f.dtype == np.uint8 and f.flags.c_contiguous

    bg = render(_far_state(), track)
    # Island centre (20, 12.5) is grass
    assert tuple(bg[int(12.5 / 40 * 64), int(20 / 40 * 64)]) == p.grass

    # the car only changes pixels near the car
    changed = np.argwhere(np.any(f != bg, axis=2))
    assert len(changed) > 0
    cy, cx = (changed + 0.5).mean(axis=0) * 40 / 64
    assert math.hypot(cx - 20.0, cy - 27.5) < 0.7
    radius = 0.5 * math.hypot(4.4, 2.5) + 40 / 64 * 1.5
    for i, j in changed:
        assert math.hypot((j + 0.5) * 40 / 64 - 20.0, (i + 0.5) * 40 / 64 - 27.5) < radius

    # a 0.25 m move changes at least one pixel
    assert np.any(render(State(20.25, 27.5, 0.0, 0.0), track) != f)

    # nose and body readable at 64 px
    palette = np.array([p.grass, p.road, p.kerb, p.body, p.nose], dtype=float)
    nearest = np.argmin(((f[:, :, None, :].astype(float) - palette) ** 2).sum(-1), axis=2)
    assert (nearest == 4).sum() >= 6
    assert (nearest == 3).sum() >= 8
    # nose is ahead (+x) of the body
    assert np.argwhere(nearest == 4)[:, 1].mean() > np.argwhere(nearest == 3)[:, 1].mean()

    # pure and deterministic
    assert np.array_equal(render(s, track), f)


def test_render_speed(track):
    render(State(20.0, 27.5, 0.3, 0.0), track)
    n = 500
    t0 = time.perf_counter()
    for k in range(n):
        render(State(20.0 + k * 0.01, 27.5, 0.3, 0.0), track)
    assert (time.perf_counter() - t0) / n < 2e-3


def test_render_hires(track):
    f = render(State(20.0, 27.5, 0.0, 0.0), track, resolution=128)
    assert f.shape == (128, 128, 3)


def _reference_render(state, track, p, resolution):
    """Cache-free reference: colour every sub-pixel, then block-average (Step 1 SPEC §5 steps 1–4)."""
    from toycar import CAR_LENGTH, CAR_WIDTH
    from toycar.render import NOSE_FRACTION

    ss = p.supersample
    n = resolution * ss
    c = (np.arange(n, dtype=np.float64) + 0.5) * (40.0 / n)
    px, py = np.meshgrid(c, c)
    d = track.distance_grid(px, py)
    hw = track.params.road_half_width_m
    sub = np.empty((n, n, 3), dtype=np.float64)
    sub[...] = p.grass
    sub[d < hw + track.params.kerb_width_m] = p.kerb
    sub[d < hw] = p.road
    fx, fy = math.cos(state.theta), math.sin(state.theta)
    fwd = (px - state.x) * fx + (py - state.y) * fy
    lat = (px - state.x) * -fy + (py - state.y) * fx
    car = (np.abs(fwd) < CAR_LENGTH / 2) & (np.abs(lat) < CAR_WIDTH / 2)
    sub[car] = p.body
    sub[car & (fwd > CAR_LENGTH / 2 - NOSE_FRACTION * CAR_LENGTH)] = p.nose
    avg = sub.reshape(resolution, ss, resolution, ss, 3).mean(axis=(1, 3), dtype=np.float64)
    return np.round(avg).astype(np.uint8)


def test_render_matches_reference_any_colours(track):
    """Recolourable background cache (Step 3 ticket 03) is bit-identical to a full redraw."""
    rng = np.random.default_rng(0)
    for k in range(60):
        cols = [tuple(int(v) for v in rng.integers(0, 256, 3)) for _ in range(5)]
        p = RenderParams(*cols, supersample=int(rng.choice([1, 2, 4])))
        res = int(rng.choice([32, 64, 96]))
        s = State(float(rng.uniform(-2, 42)), float(rng.uniform(-2, 42)), float(rng.uniform(-math.pi, math.pi)), 0.0)
        assert np.array_equal(render(s, track, p, res), _reference_render(s, track, p, res)), (k, p, res, s)


def test_render_speed_new_colours(track):
    """A frame with never-seen colours costs about as much as a cached one (no background redraw)."""
    render(State(20.0, 27.5, 0.3, 0.0), track)
    n = 300
    t0 = time.perf_counter()
    for k in range(n):
        render(State(20.0, 27.5, 0.3, 0.0), track, RenderParams(grass=(46, 110, k % 256), road=(k % 256, 70, 76)))
    assert (time.perf_counter() - t0) / n < 2e-3
