# Training recipe and freezing

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 03-architecture-and-objective.md
Parent: ../MAP.md

## Question

How is the compressor trained to the frozen version, within a weekend budget on this Mac?

- Data: all train Frames, or a subsample (consecutive Frames are near-duplicates)? Random-Frame sampling via `frame_dataset(train)`; val Frames for monitoring; Probes for reporting only.
- Optimiser, learning rate and schedule, batch size, number of steps / wall-clock budget, EMA weights or not, mixed precision on MPS (from ticket 01).
- Stopping rule and what "frozen" means: the checkpoint that passes the ticket 04 thresholds on val; the latent normalisation statistics computed afterwards on train.
- Reproducibility: seed, determinism expectations from ticket 01.
- What gets logged per run (losses, ticket 04 metrics on val, reconstruction images) — consistent with ticket 02.

Check: a full recipe table, an expected wall-clock time, and the freeze procedure.
