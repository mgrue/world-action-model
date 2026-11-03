"""PROTOTYPE — throwaway. Answers ticket "Track geometry, car size and look at 64x64".

Three structurally different track/car layouts, rendered at native 64x64 and upscaled,
so a human can pick one. Not production code: no tests, no API, no reuse intended.

Run:  uv run --with numpy --with pillow python .scratch/step1-simulator/prototypes/track_look_prototype.py
Out:  .scratch/step1-simulator/prototypes/out/variant_{A,B,C}.png
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RES = 64          # native frame size (world units == native pixels)
SS = 4            # supersampling factor for anti-aliasing
UP = 6            # upscale factor for viewing
OUT = Path(__file__).parent / "out"

GRASS = np.array([46, 110, 52])
ROAD = np.array([70, 70, 76])
KERB = np.array([200, 200, 200])
DIVIDER = np.array([225, 225, 120])
ISLAND = np.array([30, 80, 36])
CAR_BODY = np.array([220, 40, 40])
CAR_NOSE = np.array([255, 235, 60])


# ---------------------------------------------------------------- geometry

def stadium(cx, cy, straight, r, n=600):
    """Closed stadium centerline, counterclockwise in image coords? (y down) — doesn't matter here."""
    per = 2 * straight + 2 * np.pi * r
    s = np.linspace(0, per, n, endpoint=False)
    pts = []
    for d in s:
        if d < straight:                                   # bottom straight, left->right
            pts.append((cx - straight / 2 + d, cy + r))
        elif d < straight + np.pi * r:                     # right semicircle
            a = (d - straight) / r
            pts.append((cx + straight / 2 + r * np.sin(a), cy + r * np.cos(a)))
        elif d < 2 * straight + np.pi * r:                 # top straight, right->left
            pts.append((cx + straight / 2 - (d - straight - np.pi * r), cy - r))
        else:                                              # left semicircle
            a = (d - 2 * straight - np.pi * r) / r
            pts.append((cx - straight / 2 - r * np.sin(a), cy - r * np.cos(a)))
    return np.array(pts)


def catmull_rom_closed(ctrl, n=800):
    ctrl = np.asarray(ctrl, float)
    k = len(ctrl)
    out = []
    per = n // k
    for i in range(k):
        p0, p1, p2, p3 = ctrl[(i - 1) % k], ctrl[i], ctrl[(i + 1) % k], ctrl[(i + 2) % k]
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def seg_distance(px, poly, closed=True):
    """Distance from points px (N,2) to polyline; also arc length of nearest point."""
    a = poly
    b = np.roll(poly, -1, axis=0) if closed else poly[1:]
    if not closed:
        a = poly[:-1]
    ab = b - a
    seglen = np.linalg.norm(ab, axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)[:-1]])
    best_d = np.full(len(px), np.inf)
    best_s = np.zeros(len(px))
    for i in range(0, len(a), 64):  # chunk to bound memory
        A, AB, L, C = a[i:i + 64], ab[i:i + 64], seglen[i:i + 64], cum[i:i + 64]
        ap = px[:, None, :] - A[None]
        t = np.clip((ap * AB[None]).sum(-1) / np.maximum(L ** 2, 1e-9), 0, 1)
        proj = A[None] + t[..., None] * AB[None]
        d = np.linalg.norm(px[:, None, :] - proj, axis=-1)
        j = d.argmin(1)
        dj = d[np.arange(len(px)), j]
        better = dj < best_d
        best_d[better] = dj[better]
        best_s[better] = (C[j] + t[np.arange(len(px)), j] * L[j])[better]
    return best_d, best_s


def tangent_pose(poly, frac, lateral=0.0, reverse=False):
    i = int(frac * len(poly)) % len(poly)
    p, q = poly[i], poly[(i + 3) % len(poly)]
    d = (q - p) / np.linalg.norm(q - p)
    if reverse:
        d = -d
    normal = np.array([-d[1], d[0]])
    pos = p + lateral * normal
    return pos[0], pos[1], np.arctan2(d[1], d[0])


# ---------------------------------------------------------------- variants

@dataclass
class Variant:
    key: str
    name: str
    paths: list            # list of (polyline, closed)
    road_w: float
    car_len: float
    car_w: float
    lanes: int             # 1 or 2 (2 => dashed divider on centerline)
    poses: list = field(default_factory=list)
    island: object = None  # optional (cx, cy, rx, ry) ellipse painted inside fork


def variant_a():
    cl = stadium(32, 32, straight=24, r=13)
    lane = 3.0  # lane centre offset from centerline
    return Variant("A", "Stadium oval, 2 lanes, road 12px, car 6x3.5", [(cl, True)], 12, 6, 3.5, 2, poses=[
        tangent_pose(cl, 0.10, +lane), tangent_pose(cl, 0.35, -lane),
        tangent_pose(cl, 0.62, +lane, reverse=True), tangent_pose(cl, 0.88, -lane, reverse=True)])


def variant_b():
    ctrl = [(10, 12), (32, 8), (54, 12), (56, 30), (44, 36), (54, 52), (30, 56), (12, 50), (20, 32)]
    cl = catmull_rom_closed(ctrl)
    lane = 2.5
    return Variant("B", "Twisty spline, 2 lanes, road 10px, car 5x3", [(cl, True)], 10, 5, 3, 2, poses=[
        tangent_pose(cl, 0.05, +lane), tangent_pose(cl, 0.40, -lane),
        tangent_pose(cl, 0.55, +lane, reverse=True), tangent_pose(cl, 0.85, -lane)])


def variant_c():
    # Rounded-rectangle loop; on the top straight the road splits around an island.
    main = stadium(32, 34, straight=26, r=14)
    # fork branch: leaves the top straight, bulges up, rejoins
    xs = np.linspace(14, 50, 120)
    bulge = 15 * np.sin(np.pi * (xs - 14) / 36) ** 2
    upper = np.stack([xs, 20 - bulge], axis=1)
    lower_xs = np.linspace(14, 50, 120)
    lower = np.stack([lower_xs, 20 + 0 * lower_xs], axis=1)
    return Variant("C", "Loop with a fork (grass between branches), 1 lane, road 8px, car 7x4",
                   [(main, True), (upper, False), (lower, False)], 8, 7, 4, 1,
                   island=None,
                   poses=[(24, 20 - 15 * np.sin(np.pi * 10 / 36) ** 2, -1.0),   # taking upper branch
                          (40, 20.0, np.pi),                                         # lower branch, other way
                          tangent_pose(main, 0.20), tangent_pose(main, 0.45, reverse=True)])


# ---------------------------------------------------------------- rendering

def render(v: Variant, pose):
    n = RES * SS
    ys, xs = np.mgrid[0:n, 0:n]
    px = np.stack([(xs + 0.5) / SS, (ys + 0.5) / SS], -1).reshape(-1, 2)
    img = np.tile(GRASS, (len(px), 1)).astype(float)

    # road (union of paths), kerb edge, divider
    dmin = np.full(len(px), np.inf)
    s_near = np.zeros(len(px))
    for poly, closed in v.paths:
        d, s = seg_distance(px, poly, closed)
        upd = d < dmin
        dmin[upd], s_near[upd] = d[upd], s[upd]
    half = v.road_w / 2
    img[dmin < half + 0.5] = KERB
    img[dmin < half] = ROAD
    if v.lanes == 2:
        dash = (s_near % 4.0) < 2.0
        img[(dmin < 0.35) & dash] = DIVIDER
    if v.island is not None:
        cx, cy, rx, ry = v.island
        inside = ((px[:, 0] - cx) / rx) ** 2 + ((px[:, 1] - cy) / ry) ** 2 < 1
        img[inside] = ISLAND

    # car: rotated rectangle; front 35% is the nose colour
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    rel = px - np.array([x, y])
    fwd = rel @ np.array([c, s])
    lat = rel @ np.array([-s, c])
    body = (np.abs(fwd) < v.car_len / 2) & (np.abs(lat) < v.car_w / 2)
    img[body] = CAR_BODY
    img[body & (fwd > v.car_len / 2 * 0.3)] = CAR_NOSE

    img = img.reshape(n, n, 3).reshape(RES, SS, RES, SS, 3).mean((1, 3))   # box-filter downsample
    return img.round().astype(np.uint8)


def up(arr, k=UP):
    return np.kron(arr, np.ones((k, k, 1), dtype=np.uint8))


def grid_overlay(arr, cell=8, k=UP):
    big = up(arr, k).copy()
    big[::cell * k, :] = (255, 255, 255)
    big[:, ::cell * k] = (255, 255, 255)
    return big


def crop_around(frame, x, y, half=8):
    pad = np.pad(frame, ((half, half), (half, half), (0, 0)), constant_values=0)
    xi, yi = int(round(x)) + half, int(round(y)) + half
    return pad[yi - half:yi + half, xi - half:xi + half]


def sheet(v: Variant):
    frames = [render(v, p) for p in v.poses]
    W = 4 * RES * UP + 5 * 10
    rows = []

    # row 1: four poses, upscaled
    rows.append(("Four poses, x6 nearest-neighbour", [up(f) for f in frames]))
    # row 2: native size (x1 and x2) + latent grid overlay
    rows.append(("Native 64x64 (x1, x2) | pose 1 with 8x8 latent grid (8px cells)",
                 [f for f in frames[:2]] + [up(frames[0], 2), grid_overlay(frames[1])]))
    # row 3: heading wheel — car crop at 12 headings
    x0, y0, _ = v.poses[0]
    wheel = [crop_around(render(v, (x0, y0, a)), x0, y0) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    rows.append(("Heading every 30 deg (16x16 crop, x6)", [up(w) for w in wheel[:6]]))
    rows.append(("", [up(w) for w in wheel[6:]]))
    # row 4: sub-pixel motion — 0.25px steps along heading
    x, y, th = v.poses[0]
    steps = [crop_around(render(v, (x + k * 0.25 * np.cos(th), y + k * 0.25 * np.sin(th), th)), x, y)
             for k in range(6)]
    rows.append(("Sub-pixel motion: 0.25px steps (fixed crop, x6) — should visibly shift", [up(w) for w in steps]))

    heights = [max(im.shape[0] for im in ims) + 30 for _, ims in rows]
    canvas = Image.new("RGB", (W, sum(heights) + 40), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"Variant {v.key}: {v.name}", fill=(255, 255, 255))
    yy = 40
    for (label, ims), h in zip(rows, heights):
        draw.text((10, yy), label, fill=(190, 190, 190))
        xx = 10
        for im in ims:
            canvas.paste(Image.fromarray(im), (xx, yy + 18))
            xx += im.shape[1] + 10
        yy += h

    car_px = v.car_len * v.car_w
    draw.text((10, yy), f"car area ~{car_px:.0f} px^2 at native res; road {v.road_w}px; "
                        f"{v.lanes} lane(s)", fill=(190, 190, 190))
    return canvas


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    import sys
    want = sys.argv[1:] or ['A', 'B', 'C']
    for make in (variant_a, variant_b, variant_c):
        if make.__name__[-1].upper() not in want:
            continue
        v = make()
        path = OUT / f"variant_{v.key}.png"
        sheet(v).save(path)
        print(f"wrote {path}  —  {v.name}")
