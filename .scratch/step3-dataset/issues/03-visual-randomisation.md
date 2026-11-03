# Minor-visual randomisation and held-out world changes

Labels: wayfinder:grilling
Status: open
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
