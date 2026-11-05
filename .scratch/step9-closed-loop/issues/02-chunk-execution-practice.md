# How chunked policies are executed in practice

Labels: wayfinder:research
Status: open
Assignee:
Blocked by: —
Parent: ../MAP.md

## Question

How do action-chunking policies decide how much of a predicted Chunk to execute and how to replan, and what are the known failure modes at multi-modal decisions?

- Diffusion Policy (execute 8 of 16, receding horizon), ACT (temporal ensembling of overlapping Chunks), bidirectional decoding / real-time chunking (Black et al.), π0/π0.5 and similar flow-matching VLAs; FLUX 3 Action and the local reference `/Users/grue/code/flux-action` (inference/sampling code) if it shows an execution rule.
- Mode switching between consecutive Chunks at a two-answer decision (our Fork with `any`): is it reported, and what fixes are used (ensembling, warm-starting noise, conditioning on the previous Chunk, longer execution)?
- Sample selection when n_samples > 1, and typical replan intervals relative to Chunk length.

Check: `research/02-chunk-execution-practice.md` with sources, a short table of practices (execute-k, ensembling, consistency tricks) and their fit for our toy (no retraining).
