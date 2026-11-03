# Track geometry, car size and look at 64×64

Labels: wayfinder:prototype
Status: closed
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

What track layout, road and lane width, car size, nose marker and colour palette make a 64×64 overhead frame of the **whole track** readable — car position and heading clearly visible, two lanes distinguishable, both driving directions possible — while the car still occupies enough pixels to survive the 8×8 latent grid of Step 4 (8 px per latent cell)? Includes: track shape family (oval / rounded rectangle / spline), centerline representation, world-to-pixel scale, and whether to anti-alias/supersample so sub-pixel motion is visible.

Prototype: render a few candidate layouts at 64×64 (and upscaled ×8) with the car at several poses, for the human to react to.

Check: the human picks a candidate and can read the car's heading at native 64×64.

## Comments

Prototype: prototypes/track_look_prototype.py (sheets in prototypes/out/variant_{A,B,C}.png).

- Q1 track structure → **Variant C**: single-lane loop with one fork around a grass island. Instructions = branch (upper/lower) × direction (cw/ccw). Reason: only a fork makes regression "average into the wall" (Step 10); lanes average onto the divider, directions average to a stall.
- Q2 resolution → model frames are **64×64**; the world is defined in resolution-independent track units and the renderer takes `resolution` as a parameter (default 64; debug/keyboard views may render larger). Reason: 256 costs ~16× dataset size and VAE compute (or a token count that doesn't fit 16 GB) and adds no information past the 8×8×16 latent.
- Q3 fork shape → **mirror-symmetric fork**: island centred on the top straight; one branch bulges outward, one inward into the infield; equal length and curvature so neither branch is the default.
- Q4 representation → **Track = two closed Routes (outer, inner) built from straight segments and circular arcs; road surface = union of points within half road width of either Route.** Exact arc length / closest point / curvature for expert, progress and off-road tests. Terms in repo-root CONTEXT.md.
- Q5 units & scale → **track units are metres**; the frame covers a fixed 40 m × 40 m square (0.625 m per native px at 64). Car 4.4 m × 2.5 m (≈7×4 px), road 5.0 m wide (8 px). Origin top-left of frame, x right, y down; heading 0 = +x, positive heading = clockwise on screen.
- Q6 look → palette grass (46,110,52), road (70,70,76), kerb (200,200,200) 0.3 m on both road edges, car body (220,40,40), nose = front 35% of car length in (255,235,60); island/infield plain grass. Anti-aliasing: render 4× supersampled at target resolution, box-average 4×4. Draw order grass → kerb → road → car. No start line, divider or other markings. Colours are named parameters with these defaults (randomisation ranges stay fog).
- Q7 exact geometry → accepted as rendered in prototypes/out/final_C.png (source: prototypes/track_final_prototype.py).

**Resolution (2026-11-03):**
- **Structure:** single-lane loop with one mirror-symmetric fork (Variant C). Instructions will be branch (outer/inner) × direction (cw/ccw).
- **Representation:** Track = two closed Routes (outer, inner) of straight segments and circular arcs; road surface = points within 2.5 m of either Route centerline; kerb band 2.5–2.8 m; island = grass between branches.
- **Units/frame:** metres; frame = fixed 40 m × 40 m; origin top-left, x right, y down; heading 0 = +x, positive = clockwise on screen.
- **Geometry:** stadium centred (20, 20), straights 18 m, end arcs r = 7.5 m (top straight y = 12.5, bottom y = 27.5). Fork at (11, 12.5), Merge at (29, 12.5) = whole top straight. Each branch = arcs turning (∓α, ±2α, ∓α) with α = 2·atan(2h/L) = 67.4°, ρ = L / (4 sin α) = 4.88 m, L = 18 m, h = 6 m; outer branch bulges up (−y), inner down (+y). Both routes 88.0 m. Island 7 m at apex. Margins: 0.7 m left/right, 3.7 m top.
- **Car:** 4.4 m × 2.5 m rectangle; front 35 % of length is the nose colour.
- **Look:** grass (46,110,52), road (70,70,76), kerb (200,200,200), body (220,40,40), nose (255,235,60); colours are named parameters. No other markings. Draw order grass → kerb → road → car.
- **Rendering:** model frames 64×64 uint8 RGB; `resolution` is a parameter; 4× supersample + 4×4 box average. Prototype technique for speed: cache the static background per resolution; only the car is drawn per frame.
- **Constraint passed on:** tightest curve is the 4.88 m branch arc → the car's minimum turning radius must be comfortably below that (Physics ticket).
