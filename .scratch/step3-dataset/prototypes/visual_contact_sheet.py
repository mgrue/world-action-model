"""PROTOTYPE — throwaway. Ticket 03: contact sheet of jittered and probe visuals at true 64×64.

  uv run python .scratch/step3-dataset/prototypes/visual_contact_sheet.py
Out: .scratch/step3-dataset/prototypes/out/visual_contact_sheet.png
"""

import math
from pathlib import Path

import numpy as np
import pygame

from toycar import RenderParams, State, Track, TrackParams, render

OUT = Path(__file__).parent / "out"
D = RenderParams()
NAMES = ("grass", "road", "kerb", "body", "nose")


def jitter_colour(rgb, rng):
    b = rng.uniform(0.85, 1.15)
    return tuple(int(np.clip(round(c * b + rng.uniform(-10, 10)), 0, 255)) for c in rgb)


def sample_params(seed):
    rng = np.random.default_rng([seed, 3])
    return RenderParams(**{n: jitter_colour(getattr(D, n), rng) for n in NAMES})


def extreme(b, off):
    return RenderParams(**{n: tuple(int(np.clip(round(c * b + off), 0, 255)) for c in getattr(D, n)) for n in NAMES})


PROBES = {
    "brown_grass": (RenderParams(grass=(120, 100, 50)), TrackParams()),
    "light_road": (RenderParams(road=(140, 140, 145)), TrackParams()),
    "blue_car": (RenderParams(body=(40, 80, 220)), TrackParams()),
    "no_kerb": (RenderParams(kerb=D.road), TrackParams()),
    "wide_fork": (RenderParams(), TrackParams(bulge_m=7.0)),
}


def pose(track, k):
    """Car on the track at a few readable spots: bottom straight, fork branch, end arc."""
    route = track.route("outer" if k % 2 else "inner")
    s = [track.length_m * f for f in (0.08, 0.15, 0.45, 0.62, 0.8, 0.95)][k % 6]
    x, y, th = route.pose_at(s)
    return State(x, y, th + (math.pi if k % 3 == 0 else 0.0), 5.0)


def tile(frame, scale=3):
    return np.kron(frame, np.ones((scale, scale, 1), np.uint8))


def main():
    OUT.mkdir(exist_ok=True)
    track = Track.default()
    rows = []
    # rows 1–3: 18 sampled episodes
    for r in range(3):
        rows.append([tile(render(pose(track, 6 * r + c), track, sample_params(1_000_000 + 6 * r + c))) for c in range(6)])
    # row 4: default + extremes of the jitter range
    ext = [D, extreme(0.85, -10), extreme(0.85, 10), extreme(1.15, -10), extreme(1.15, 10), D]
    rows.append([tile(render(pose(track, c), track, p)) for c, p in enumerate(ext)])
    # row 5: probes + blank
    prow = []
    for c, (name, (rp, tp)) in enumerate(PROBES.items()):
        t = Track(tp)
        prow.append(tile(render(pose(t, c), t, rp)))
    prow.append(np.full_like(prow[0], 255))
    rows.append(prow)
    gap = 6
    h, w = rows[0][0].shape[:2]
    sheet = np.full((len(rows) * (h + gap) + gap, 6 * (w + gap) + gap, 3), 255, np.uint8)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            y, x = gap + i * (h + gap), gap + j * (w + gap)
            sheet[y : y + h, x : x + w] = im
    pygame.image.save(pygame.surfarray.make_surface(sheet.swapaxes(0, 1)), str(OUT / "visual_contact_sheet.png"))
    # car readability: nose vs body vs road contrast at the extremes
    for name, p in [("default", D), ("dark extreme", extreme(0.85, -10)), ("bright extreme", extreme(1.15, 10))]:
        dist = lambda a, b: math.dist(a, b)
        print(f"{name:15s} body-road {dist(p.body, p.road):6.1f}  nose-body {dist(p.nose, p.body):6.1f}  "
              f"road-grass {dist(p.road, p.grass):6.1f}  nose-kerb {dist(p.nose, p.kerb):6.1f}")
    print("worst case over 10000 sampled episodes:")
    worst = {k: 1e9 for k in ("body-road", "nose-body", "road-grass", "nose-kerb", "body-grass")}
    for s in range(10000):
        p = sample_params(s)
        for k in worst:
            a, b = k.split("-")
            worst[k] = min(worst[k], math.dist(getattr(p, a), getattr(p, b)))
    print({k: round(v, 1) for k, v in worst.items()})


if __name__ == "__main__":
    main()
