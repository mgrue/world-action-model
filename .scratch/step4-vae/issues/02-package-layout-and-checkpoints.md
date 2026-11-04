# Model package layout, runs and checkpoints

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 01-torch-on-mac.md
Parent: ../MAP.md

## Question

Where does model code live, and how are trained artefacts stored and identified?

- Layout: a second package in this project (e.g. `src/wam/` next to `toycar`, torch in a dependency group) or a separate uv project depending on `toycar`? One package for all model steps (4–10) with a module per step?
- Torch as a dependency: main or optional group; how the "no torch in `toycar`" rule is enforced (a test?).
- Run directories: where training runs write (`runs/` is already gitignored), naming, what gets logged (loss curves as CSV/JSON, sample images), and whether to add a logger (TensorBoard) or keep it plain.
- Checkpoint format for the **frozen** compressor: `state_dict` + config (`safetensors` or `torch.save`), the provenance it carries (dataset path + manifest hash, git commit, code hash, training config), and how Steps 6–10 load it by name (e.g. `checkpoints/vae-v1/`), and whether it is committed or gitignored.
- Seeds and reproducibility expectations for training (exact vs statistical), informed by ticket 01.

Check: a layout sketch (directories, modules, dependency groups), the checkpoint contents and provenance fields, and the run-directory contents.

## Comments
- Q1 layout → a **second module `wam`** ("world action model") in the same uv project: `src/wam/` next to `src/toycar/`, `[tool.uv.build-backend] module-name = ["toycar", "wam"]`; one `pyproject.toml`, one `uv.lock`, one `.venv`. Modules by concern, not by Step: `wam/compressor.py` (Step 4), later `model.py`, `train.py`, `sample.py`, … A uv workspace was rejected as extra friction for a local project.
- Q2 torch → dependency group `model = ["torch>=2.15,<2.16"]` (exact version in `uv.lock`), with `[tool.uv] default-groups = ["dev", "model"]` so plain `uv run` / `uv sync` include it; `uv sync --no-group model` gives a torch-free environment. matplotlib stays in the optional `viz` group, imported lazily. Rule enforced by a test: a fresh subprocess imports every `toycar` module and asserts `"torch" not in sys.modules`.
- Q3 run directories → `runs/vae/<YYYYMMDD-HHMM>-<name>/` (later `runs/wam/…`), never overwritten, gitignored: `config.json` (training config + provenance), `log.csv` (every 100 steps: step, loss, rec, KL, lr, Frames/s, elapsed), `metrics.json` (ticket 04 check clean / noisy / per Probe + latent stats), `contact.png`, `checkpoint.pt`, `stdout.txt`. No TensorBoard / wandb. A run is scratch; only a promoted checkpoint is used by later Steps.
- Q4 checkpoint contents → `torch.save` of a plain dict, loaded with `weights_only=True` (no safetensors dependency). `model.pt` = `state_dict`, `arch` (channels, blocks per stage, Z), `latent_mean`, `latent_std`. `card.json` = name, date, source run; git commit + dirty flag, sha256 of `wam/compressor.py`, torch version, device; dataset path, sha256 of its `manifest.json`, Split, latent-stats Frame count; full training config and final losses; ticket 04 summary with verdicts.
- Q5 promote / load / git → `python -m wam.compressor promote <run> --name vae-v1` copies into `checkpoints/vae-v1/{model.pt, card.json}`; refuses an existing name (a better compressor is `vae-v2`) and a run failing the ticket 04 clean check unless `--force`. `wam.compressor.load("vae-v1", device)` returns a frozen compressor (eval, no grad); later Steps refer to it by name only. On load: error on `arch` mismatch, warning on a `compressor.py` hash mismatch, no dataset check (ticket 06 checks the manifest hash where latents are cached). **`checkpoints/` is committed** (≈ 13 MB for a 3.2 M-param float32 model; `!checkpoints/**` exception to `*.pt`): a retrain costs ~80 min and MPS ≠ CPU bitwise. The repo will go to GitHub: under the 50 MB warning, no LFS; cards and configs hold **repo-relative paths only**, nothing machine-specific beyond commit and device type.
- Q6 reproducibility → one seed in the config, separate derived streams for init, Frame sampling and VAE noise. Same seed + device + torch version + code → bit-identical; otherwise statistically equivalent (same ticket 04 verdict), which is why the checkpoint is committed. `encode()` is deterministic (mean latents, no sampling). CPU-only tests (no MPS on GitHub CI): same seed → equal weights after a few steps, different seed → different; `encode` repeatable; encode → decode round trip on a tiny checkpoint built in the test; tests needing the real `vae-v1` skip if it is absent.

**Resolution (2026-11-04):**
```
pyproject.toml          module-name = ["toycar", "wam"]; group model = torch 2.15.x (default); viz optional
src/toycar/             numpy only — test: importing every module leaves torch out of sys.modules
src/wam/compressor.py   Step 4: VAE, train, promote, load, CLI (python -m wam.compressor …)
runs/vae/<YYYYMMDD-HHMM>-<name>/   config.json  log.csv  metrics.json  contact.png  checkpoint.pt  stdout.txt   (gitignored)
checkpoints/vae-v1/                model.pt (state_dict, arch, latent_mean, latent_std)  card.json (provenance + ticket 04 verdicts)   (committed)
```
Provenance fields, promote/load rules and the reproducibility promise as in Q4–Q6 above.

