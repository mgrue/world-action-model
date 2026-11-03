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
