# Model package layout, runs and checkpoints

Labels: wayfinder:grilling
Status: open
Assignee:
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
