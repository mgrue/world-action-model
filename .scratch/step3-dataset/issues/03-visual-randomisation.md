# Minor-visual randomisation and held-out world changes

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

PLAN says "randomise … minor visuals", and Step 10 later tests "small world changes: change track color or shape and see what generalizes". What varies in training, and what is held back for Step 10?

- Which `RenderParams` vary (grass, road, kerb, body, nose colours), by how much (e.g. small HSV jitter around the defaults vs wide ranges), and per episode or per Frame?
- Constraints: the car and its nose marker must stay clearly distinguishable from road and grass at 64×64 (Step 1 look decision); the Step 4 check is that reconstructions keep position and heading visible.
- Pixel-level augmentation (noise, brightness) at load time vs baked into recorded Frames?
- Track shape (`TrackParams`) stays fixed for the training set? Which colour/shape changes are reserved as held-out probes for Step 10, so they never appear in training?
- Does the expert or physics change with visuals? (They shouldn't — confirm.)

Check: a table of randomised parameters with ranges and sampling rule, the list of held-out probes reserved for Step 10, and a contact sheet (prototype) of sampled Frames showing the car stays readable.

## Comments
- Q1 what varies → **per episode** (never per Frame: all 17 Frames of a Window share one look), **small jitter on all five colours** (grass, road, kerb, body, nose), each independently: brightness factor U[0.85, 1.15] × default RGB, plus a per-channel offset U[−10, +10], rounded and clipped to [0, 255]. Car colours jitter too, so the model reads the car by shape and nose/body contrast, not exact RGB. Drawn from a separate stream `default_rng([seed, 3])` (expert coin `[seed, 1]`, noise `[seed, 2]`), so visuals never change driving. The drawn `RenderParams` is stored per episode. val/test use the same jitter as train; larger changes are reserved for Step 10 probes.
- Q2 render cost of per-episode colours → **make the renderer recolourable** (Step 1 change, output bit-identical): cache the colour-independent layout per (track, resolution, supersample), apply colours per call. A new colour set costs 0.12 ms instead of 16 ms. Rejected: a fixed set of about 32 colour schemes (fits the old cache, but caps visual variety and ties the dataset to a cache size). Step 1 ticket 02 reopened and re-resolved with code, SPEC §5 and tests (`test_render_matches_reference_any_colours`, `test_render_speed_new_colours`); `runs/` recordings still replay identically.
- Q3 pixel augmentation → **none in the dataset**: the loader returns exactly the render for the episode's `RenderParams`. Future Frames are prediction targets (no baked noise), Frames must stay reproducible against stored hashes, and the colour jitter already covers minor visuals. Pixel augmentation (e.g. on the observed Frame only) stays an optional model-side training transform for Steps 4/6.
- Q4 held-out probes → **track shape fixed** (`TrackParams()` defaults; CONTEXT: exactly one Track). Five named probes reserved in the dataset module (`PROBES`), never used in training: `brown_grass` (grass (120, 100, 50)), `light_road` (road (140, 140, 145)), `blue_car` (body (40, 80, 220), nose unchanged), `no_kerb` (kerb = road colour), `wide_fork` (`bulge_m` 6 → 7, branch arc 4.88 → 4.64 m > R_MIN 3.86 m). Colour probes re-render the stored test episodes (no re-recording): same situations, only the look changes. `wide_fork` needs new driving, so it's evaluated closed-loop in Steps 9/10, with an optional small recording on seed block `4_000_000 + k`, after re-checking the expert with `eval` on that Track.
- Q5 (answered from code) visuals never affect driving: the expert reads State/Info only, physics never sees `RenderParams`, and colours come from their own stream `[seed, 3]`.
- Q6 check → contact sheet `prototypes/out/visual_contact_sheet.png` (source `prototypes/visual_contact_sheet.py`): 18 jittered episodes, the four jitter extremes and the five probes at true 64×64. The car stays readable everywhere. Worst colour distances over 10,000 sampled episodes: nose–body 140 (default 199), body–road 107 (157), nose–kerb 116 (154), body–grass 139. The weakest pair is road–grass at 26.6 (default 52.5; slight purple or green road tints), but the kerb always outlines the road. **Ranges accepted as shown.**

**Resolution (2026-11-03):**

| Parameter | Sampling (per episode, stream `default_rng([seed, 3])`) |
|---|---|
| `grass`, `road`, `kerb`, `body`, `nose` | each independently, in this order: `b = U[0.85, 1.15]`, then per channel `clip(round(c·b + U[−10, 10]), 0, 255)`, channel order R, G, B |
| `supersample` | fixed 4 |
| `TrackParams` | fixed defaults |

- Same jitter for train, val and test; drawn `RenderParams` stored per episode; visuals never affect driving.
- No pixel augmentation in the dataset (model-side option for Steps 4/6).
- **Held-out probes** (`PROBES`, never in training): `brown_grass`, `light_road`, `blue_car`, `no_kerb` (re-render the test split), `wide_fork` (`bulge_m = 7`; closed-loop and optional recording on seed block `4_000_000 + k`, after `eval` on that Track).
- **Step 1 change made:** recolourable render cache, bit-identical output; Step 1 look ticket reopened and re-resolved (commit 55a6ef4).
