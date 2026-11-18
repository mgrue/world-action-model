# A JEPA objective that keeps the car

Labels: wayfinder:prototype
Status: open
Assignee:
Blocked by:
Parent: ../MAP.md

## Question

Which self-supervised change, beyond the I-JEPA recipe of [Architecture and objective](03-architecture-and-objective.md), makes `jepa-v1` keep the car, measured by the Pose readout (MLP pos/head p95 near `vae-v1`'s 0.24 m / 1.7°) and car contrast staying well above 1?

The failure to beat: on a fixed Track, Track position + episode colour predicts every background target, so the encoder (and its EMA target) drops the car within ~2k steps. The car is the only unpredictable part, and dropping it from the targets is the cheapest way to lower the loss. Candidates, each staying free of pose labels (the State may still steer masking or cropping, as the car bias did):

- **Frozen random target encoder** (no EMA): the random encoder at step 0 shows the car (contrast 2.3, ridge 2.7 m); a fixed target cannot be redefined to drop it.
- **4 px patches** (256 tokens, pooled to 8 × 8 at the output): the car spans more tokens, so part of it can predict the rest; ≈ 4–8× slower.
- **Car-centred crops as extra views**: upsampled crops where the car fills a large share, so predicting it dominates a share of the loss.
- **SIGReg (LeJEPA)** on context tokens, no EMA.

Same budget rule as ticket 03 (8k screens, then one 30k confirmation of the winner), scored with `prototypes/jepa_proto.py` diagnostics and `readout_ckpt.py`.

Check: one objective with screen and 30k numbers that keeps the car (or a documented failure of all candidates, which reopens the destination).

## Comments
