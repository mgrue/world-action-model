"""PROTOTYPE — throwaway. Candidate final layout for ticket "Track geometry, car size and look at 64x64".

Symmetric fork, two Routes of lines + arcs, metres, 40 m frame, decided palette and AA.

Run:  uv run --python 3.14 --with numpy --with pillow python .scratch/step1-simulator/prototypes/track_final_prototype.py
Out:  .scratch/step1-simulator/prototypes/out/final_C.png
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WORLD = 40.0                      # metres covered by the frame
ROAD_W, KERB_W = 5.0, 0.3
CAR_L, CAR_W, NOSE_FRAC = 4.4, 2.5, 0.35
SS = 4
PAL = dict(grass=(46, 110, 52), road=(70, 70, 76), kerb=(200, 200, 200),
           body=(220, 40, 40), nose=(255, 235, 60))

# ---- geometry (metres; x right, y down; heading 0 = +x, positive = clockwise on screen)
CX, CY, STRAIGHT, R_END = 20.0, 20.0, 18.0, 7.5
FORK_LEN, BULGE_H = STRAIGHT, 6.0
ALPHA = 2 * np.arctan(2 * BULGE_H / FORK_LEN)        # tan(a/2) = 2h/L for the a,-2a,a arc chain
RHO = FORK_LEN / (4 * np.sin(ALPHA))


def build_route(sign):
    """sign=-1: outer branch (bulges up, toward frame edge); +1: inner branch (into infield)."""
    x0, y0 = CX - STRAIGHT / 2, CY - R_END                # Fork point, start of top straight
    prims = [("arc", RHO, sign * ALPHA), ("arc", RHO, -sign * 2 * ALPHA), ("arc", RHO, sign * ALPHA),
             ("arc", R_END, np.pi), ("line", STRAIGHT), ("arc", R_END, np.pi)]
    pts, x, y, th = [], x0, y0, 0.0
    for p in prims:
        if p[0] == "line":
            n = int(p[1] / 0.02)
            for _ in range(n):
                pts.append((x, y)); x += 0.02 * np.cos(th); y += 0.02 * np.sin(th)
        else:
            _, r, turn = p
            n = int(abs(turn) * r / 0.02)
            dth = turn / n
            step = 2 * r * np.sin(abs(dth) / 2)
            for _ in range(n):
                pts.append((x, y))
                th_mid = th + dth / 2
                x += step * np.cos(th_mid); y += step * np.sin(th_mid); th += dth
    assert np.hypot(x - x0, y - y0) < 1e-6 and abs((th % (2 * np.pi))) < 1e-6, "route not closed"
    return np.array(pts)


OUTER, INNER = build_route(-1), build_route(+1)


def dist_field(n, pts, reach):
    """Distance (m) from each of n x n pixel centres to the nearest route point, valid up to `reach`."""
    cell = WORLD / n
    d = np.full((n, n), np.inf)
    w = int(np.ceil(reach / cell)) + 1
    for x, y in pts[::2]:                                   # ~0.04 m spacing is plenty
        i0, j0 = int(y / cell), int(x / cell)
        ys = slice(max(i0 - w, 0), min(i0 + w + 1, n)); xs = slice(max(j0 - w, 0), min(j0 + w + 1, n))
        yy, xx = np.mgrid[ys, xs]
        dd = np.hypot((xx + 0.5) * cell - x, (yy + 0.5) * cell - y)
        d[ys, xs] = np.minimum(d[ys, xs], dd)
    return d.reshape(-1)


_bg_cache = {}


def background(res):
    if res not in _bg_cache:
        n = res * SS
        ys, xs = np.mgrid[0:n, 0:n]
        px = np.stack([(xs + 0.5), (ys + 0.5)], -1).reshape(-1, 2) * (WORLD / n)
        reach = ROAD_W / 2 + KERB_W + 0.1
        d = np.minimum(dist_field(n, OUTER, reach), dist_field(n, INNER, reach))
        img = np.tile(np.array(PAL["grass"], float), (len(px), 1))
        img[d < ROAD_W / 2 + KERB_W] = PAL["kerb"]
        img[d < ROAD_W / 2] = PAL["road"]
        _bg_cache[res] = (img.reshape(n, n, 3), px.reshape(n, n, 2))
    return _bg_cache[res]


def render(pose, res=64):
    img, px = background(res)
    img = img.copy()
    x, y, th = pose
    rel = px - np.array([x, y])
    fwd = rel @ np.array([np.cos(th), np.sin(th)])
    lat = rel @ np.array([-np.sin(th), np.cos(th)])
    body = (np.abs(fwd) < CAR_L / 2) & (np.abs(lat) < CAR_W / 2)
    img[body] = PAL["body"]
    img[body & (fwd > CAR_L / 2 - NOSE_FRAC * CAR_L)] = PAL["nose"]
    return img.reshape(res, SS, res, SS, 3).mean((1, 3)).round().astype(np.uint8)


def pose_on(route, frac, reverse=False):
    i = int(frac * len(route)) % len(route)
    p, q = route[i], route[(i + 5) % len(route)]
    d = q - p
    th = np.arctan2(d[1], d[0]) + (np.pi if reverse else 0)
    return p[0], p[1], th


def up(a, k):
    return np.kron(a, np.ones((k, k, 1), dtype=np.uint8))


if __name__ == "__main__":
    poses = [pose_on(OUTER, 0.06), pose_on(INNER, 0.06), pose_on(INNER, 0.10, reverse=True),
             pose_on(OUTER, 0.45), pose_on(INNER, 0.70, reverse=True), pose_on(OUTER, 0.02)]
    labels = ["outer branch, cw", "inner branch, cw", "inner branch, ccw", "right end arc", "bottom, ccw", "at the Fork"]
    frames = [render(p) for p in poses]
    big = render(poses[0], res=256)

    canvas = Image.new("RGB", (6 * 266 + 10, 266 + 30 + 300 + 64 + 120), (24, 24, 28))
    dr = ImageDraw.Draw(canvas)
    for i, (f, lab) in enumerate(zip(frames, labels)):
        canvas.paste(Image.fromarray(up(f, 4)), (10 + i * 266, 30))
        canvas.paste(Image.fromarray(f), (10 + i * 266, 30 + 266))
        dr.text((10 + i * 266, 12), lab, fill=(200, 200, 200))
    dr.text((10, 380), "same pose rendered with resolution=256 (shown 1:1)", fill=(200, 200, 200))
    canvas.paste(Image.fromarray(big), (10, 400))

    gap = 2 * BULGE_H - ROAD_W
    clear = (CY + R_END) - (CY - R_END + BULGE_H) - ROAD_W
    info = [f"alpha={np.degrees(ALPHA):.1f} deg, branch arc radius={RHO:.2f} m, end arc radius={R_END} m",
            f"route length: {np.sum(np.linalg.norm(np.diff(OUTER, axis=0), axis=1)):.1f} m (both routes equal)",
            f"island width at apex: {gap:.1f} m; inner branch road edge to bottom road edge: {clear:.1f} m",
            f"x extent of road incl. kerb: {CX - STRAIGHT/2 - R_END - ROAD_W/2 - KERB_W:.1f} .. {CX + STRAIGHT/2 + R_END + ROAD_W/2 + KERB_W:.1f} m; "
            f"outer apex top edge y={CY - R_END - BULGE_H - ROAD_W/2 - KERB_W:.1f} m"]
    for j, t in enumerate(info):
        dr.text((280, 420 + j * 20), t, fill=(200, 200, 200))
    out = Path(__file__).parent / "out" / "final_C.png"
    canvas.save(out)
    print("\n".join(info)); print("wrote", out)
