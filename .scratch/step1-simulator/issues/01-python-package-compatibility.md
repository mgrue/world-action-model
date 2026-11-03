# Python and package compatibility on macOS arm64

Labels: wayfinder:research
Status: closed
Assignee:
Blocked by:
Parent: ../MAP.md

## Question

Which Python version should the project pin so that, on macOS Apple Silicon, all of these install as prebuilt wheels via `uv` and work: numpy, pygame **or** pygame-ce (window + keyboard events on macOS), and — for later steps — PyTorch with the MPS backend? Python 3.14 is installed locally. Also: any known macOS issues with pygame keyboard input/focus or Retina window scaling, and which of pygame vs pygame-ce is the better-maintained choice today.

## Comments

Research findings: research/01-python-package-compatibility.md — Pin Python 3.14 + pygame-ce 2.5.8 (SDL2) + numpy 2.5.4 + torch 2.15.0 (MPS verified); upstream pygame has no 3.14 wheel.

**Resolution (2026-11-03):** Pin Python 3.14 (`>=3.14,<3.15`); use `pygame-ce>=2.5.8,<2.6` (SDL 2.32, not upstream `pygame` — no cp314 wheel; never install both); numpy 2.5.x; torch 2.15 with MPS verified working. Display: nearest-neighbour upscale via `pygame.transform.scale` or `set_mode(..., pygame.SCALED)`; build surfaces with `surfarray.make_surface(frame.swapaxes(0,1))`; read held keys with `key.get_pressed()` and pump events every frame. Details: research/01-python-package-compatibility.md.
