# Compressor API, CLI, tests and acceptance

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 02-package-layout-and-checkpoints.md, 04-car-visibility-check.md, 05-training-recipe.md, 06-latents-for-windows.md
Parent: ../MAP.md

## Question

What module, API and commands deliver the compressor, and what proves it correct?

- API: model class, `load(name)` of the frozen checkpoint, `encode(frames_uint8) -> latents` (normalised), `decode(latents) -> frames_uint8`, batch/device handling.
- CLI: `train`, `eval` (ticket 04 metrics on val and each Probe, PASS/FAIL against the thresholds), `show` (contact sheet PNG), and `cache` if ticket 06 caches.
- Automated tests (fast, CPU, no full training): shapes and dtypes, encode/decode round trip on a tiny untrained model, checkpoint save/load, normalisation applied/undone, the pose estimator on original Frames, `toycar` still imports without torch.
- Manual acceptance: train the frozen model, `eval` passes, contact sheet shows the car's position and heading in reconstructions (PLAN check).

Check: an API and CLI sketch with signatures, the list of automated tests, and the manual acceptance steps.

## Comments
- API → as decided in ticket 06 (`load`, `Compressor.encode` / `decode` / `encode_windows`, attributes) plus ticket 02 (`promote`, checkpoint files).
- Q1 modules and CLI → **`wam/compressor.py`**: VAE network, `Compressor`, `load`, `promote`, training loop, CLI; its sha256 is the card's code hash. **`wam/carcheck.py`**: the ticket 04 check (`estimate`, `readable`, `eval_frames`, `check(originals, reconstructions, …)` → summary, `contact_sheet`), independent of the compressor so Steps 8 and 10 can score predicted / Probe Frames. CLI in the style of `toycar.dataset`, `main(argv) -> int`:
  ```
  python -m wam.compressor train   [--name NAME] [--steps 12000] [--seed 0] [--data data/step3-v1] [--cpu]
  python -m wam.compressor train   --resume runs/vae/<run>
  python -m wam.compressor eval    <run-dir | checkpoint-name> [--eval-frames 4096] [--json OUT]
  python -m wam.compressor show    <run-dir | checkpoint-name> [--out contact.png]
  python -m wam.compressor promote runs/vae/<run> --name vae-v1 [--force]
  ```
  `train` runs the ticket 05 recipe and ends with `eval` + `show` into the run directory. `eval` prints the ticket 04 tables (clean, noisy, per situation, per Probe) and exits **0 on clean PASS, 1 on FAIL or INVALID**; works on a run directory or a promoted name. `show` needs `--group viz`. `promote` per ticket 02. Hidden options for tests / smoke runs: `--eval-frames`, `--stats-frames`, `--batch`.
- Q2 automated tests (CPU, no full training, < 60 s added; a tiny `arch`, e.g. channels 8/8/16/16 with 1 block, and the Step 3 tiny dataset fixture). `tests/test_compressor.py`: (1) a subprocess importing every `toycar` module leaves `torch` out of `sys.modules`; (2) encoder → `(B, 32, 8, 8)`, decoder → `(B, 3, 64, 64)`, full `blocks=2` arch = 3,166,435 params; (3) `encode` / `decode` dtypes, shapes, any leading dims (e.g. `(2, 17, 64, 64, 3)`), numpy and torch inputs, output device; (4) `encode` = `(μ − mean) / std` and `decode(encode(x))` = raw decoder of μ; (5) repeated `encode` bit-identical, single vs in-batch within 1e-5; (6) `encode_windows` shapes and equality with separate encoding, future k at index k; (7) token i ↔ cell (`i // 8`, `i % 8`); (8) `decode(as_float=True)` gives gradients to `z`, none to weights; (9) same seed → equal weights after a few steps, other seed → different; (10) 20 steps == 10 + resume + 10, bit for bit; (11) promote / load round trip, refuses an existing name, refuses a failing run without `--force`, `arch` mismatch → error, code-hash mismatch → warning, no absolute paths in the card; (12) CLI smoke: `train --steps 5` → `eval` exit 1 → `promote --force` → `show` (skip without matplotlib). `tests/test_carcheck.py`: (13) estimator on 256 original val Frames vs true State: position p95 ≤ 0.2 m, heading p95 ≤ 10°; (14) the six ticket 04 fakes keep their verdicts on 256 Frames; (15) synthetic rows: `skip` counted, > 1% unreadable → INVALID. Also: Step 3 `test_dataloader_compat` now runs; a real-checkpoint test (skip if `checkpoints/vae-v1` or `data/step3-v1` is absent) loads `vae-v1` and PASSes the check on 512 val Frames.
- Q3 manual acceptance → the steps below. Off-Route weakness and colour-Probe failures do not block acceptance; they are recorded on the card and contact sheet for Steps 8 / 10. A failed step 2 follows ticket 05 (reopen ticket 03, no seed rerolls).

**Resolution (2026-11-04):** API and CLI as above; 15 automated tests plus the Step 3 DataLoader test and a skip-if-absent real-checkpoint test. **Manual acceptance:**
1. `uv sync` and `uv run pytest` — all pass.
2. `caffeinate -i uv run python -m wam.compressor train --name vae-v1` — about 1 h 35 min, ends with a clean **PASS** (prototype reference: position p95 0.12 m, heading p95 8.0° / max 24.7°; close, not identical).
3. `runs/vae/<run>/contact.png` — in every zoomed pair the red body and yellow nose sit in the same place and point the same way, heading arrows agreeing (PLAN: "reconstructions keep the car's position and heading clearly visible").
4. `uv run python -m wam.compressor promote runs/vae/<run> --name vae-v1`; commit `checkpoints/vae-v1/`.
5. `uv run python -m wam.compressor eval vae-v1` — same verdict, exit 0.
6. `checkpoints/vae-v1/card.json` names the commit, the dataset manifest hash and the PASS, with no absolute paths.

