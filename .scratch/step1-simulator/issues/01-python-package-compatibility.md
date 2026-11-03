# Python and package compatibility on macOS arm64

Labels: wayfinder:research
Status: open
Assignee:
Blocked by:
Parent: ../MAP.md

## Question

Which Python version should the project pin so that, on macOS Apple Silicon, all of these install as prebuilt wheels via `uv` and work: numpy, pygame **or** pygame-ce (window + keyboard events on macOS), and — for later steps — PyTorch with the MPS backend? Python 3.14 is installed locally. Also: any known macOS issues with pygame keyboard input/focus or Retina window scaling, and which of pygame vs pygame-ce is the better-maintained choice today.

## Comments
