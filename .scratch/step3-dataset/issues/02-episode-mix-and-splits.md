# Episode mix, dataset size and splits

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by:
Parent: ../MAP.md

## Question

What exactly gets recorded, how much of it, and how is it split?

- Size: how many episodes ("a few thousand"), episode length (900 ticks, or shorter episodes for more start diversity per tick)?
- Mix: balance across Instruction × Direction (6 combinations) — equal counts, or weighted (e.g. more `any` for the multimodality experiment)? Does Direction stay random from the seed or is it assigned?
- Start randomisation: are Step 1's `reset` ranges (any Station, ±1 m, ±10°, 0–6 m/s) enough, or does Step 3 widen any of them? What does "randomise speed" mean beyond the start speed, given the expert's speed profile is fixed?
- Expert style knobs: confirm `a_lat_plan`/`v_cap` fixed (Step 2 hidden-variable caveat); noise at Step 2 defaults?
- Splits: train/val/test by episode (never by window), proportions, and **disjoint seed ranges** from each other and from Step 9 closed-loop evaluation seeds and Step 2 `eval` seeds 0–999.
- Crashes: the expert doesn't crash, but should the recorder keep or reject a crashed episode if one ever happens?

Check: a table of the recorded set (Instruction × Direction × count, seed ranges per split, episode length, expert params) and the total size in episodes, ticks and windows.
