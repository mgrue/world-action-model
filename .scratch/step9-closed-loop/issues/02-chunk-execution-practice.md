# How chunked policies are executed in practice

Labels: wayfinder:research
Status: closed
Assignee: claude
Blocked by: —
Parent: ../MAP.md

## Question

How do action-chunking policies decide how much of a predicted Chunk to execute and how to replan, and what are the known failure modes at multi-modal decisions?

- Diffusion Policy (execute 8 of 16, receding horizon), ACT (temporal ensembling of overlapping Chunks), bidirectional decoding / real-time chunking (Black et al.), π0/π0.5 and similar flow-matching VLAs; FLUX 3 Action and the local reference `/Users/grue/code/flux-action` (inference/sampling code) if it shows an execution rule.
- Mode switching between consecutive Chunks at a two-answer decision (our Fork with `any`): is it reported, and what fixes are used (ensembling, warm-starting noise, conditioning on the previous Chunk, longer execution)?
- Sample selection when n_samples > 1, and typical replan intervals relative to Chunk length.

Check: `research/02-chunk-execution-practice.md` with sources, a short table of practices (execute-k, ensembling, consistency tricks) and their fit for our toy (no retraining).

## Resolution

- Common practice: execute 25–50 % of a Chunk, then replan. Diffusion Policy runs 8 of 16. π0 runs 16 or 25 of 50. FLUX `games` runs 8 of 32 at 15 Hz. ACT queries every Tick and ensembles.
- Averaging Chunks fails at multimodal decisions: temporal ensembling and EMA produce the in-between action. RTC, BID and π0 all report this or dropped averaging because of it. For our Fork the in-between action means steering into the Island.
- Mode switching between consecutive Chunks is a named, known problem. The fixes that need no retraining are:
  - picking among samples by coherence with the previous Chunk (BID);
  - inpainting the overlap with the previous Chunk (RTC);
  - choosing the initial noise so the ODE reproduces the previous Chunk (PAINT).
- Our sim waits for the model (inference delay d = 0), so RTC's latency machinery doesn't apply to us.
- What ticket 03 should try: an execute-k sweep, the same noise every replan within an episode, and best-of-N selection by backward coherence. Ensembling is the negative control, and hard-prefix inpainting is deferred.
- Details, sources and the table of practices: [research/02-chunk-execution-practice.md](../research/02-chunk-execution-practice.md).
