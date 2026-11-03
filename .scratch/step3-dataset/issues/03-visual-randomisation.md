# Minor-visual randomisation and held-out world changes

Labels: wayfinder:grilling
Status: open
Assignee:
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
