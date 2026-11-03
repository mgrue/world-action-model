# Research: Python and package compatibility on macOS arm64

Ticket: ../issues/01-python-package-compatibility.md. Researched 2026-11-03 on macOS (Darwin 25), Apple Silicon, uv.

## Recommendation

- **Pin Python 3.14** (`.python-version` = `3.14`, `requires-python = ">=3.14,<3.15"`). Every needed package has a cp314 macOS arm64 wheel, and it is the interpreter already installed.
- **Use `pygame-ce`, not `pygame`.** Upstream `pygame` has no cp314 wheel. On 3.14 it would build from source, and it has had no release in more than two years.
- **Dependencies:** `numpy`, `pygame-ce`, and `torch` later. Do not install `pygame` and `pygame-ce` together, because both provide the `import pygame` module.

## Verified empirically (throwaway uv venv, Python 3.14.6)

`uv pip install numpy pygame-ce torch` installed with wheels only. I ran the following headless, with `SDL_VIDEODRIVER=dummy`:

- numpy 2.5.4, torch 2.15.0: `torch.backends.mps.is_available()` is True and `is_built()` is True. A tensor op on `device='mps'` worked.
- pygame-ce 2.5.8 (`pygame.IS_CE == 1`), built against **SDL 2.32.10**. `display.set_mode`, `surfarray.make_surface` from a (64,64,3) uint8 array, and `transform.scale` to 512x512 all worked.
- `uv pip install --only-binary :all: pygame` on 3.14 failed. The newest wheel it found was for cp310.

## Wheel matrix (PyPI JSON API, latest versions)

| Package | Latest (date) | requires-python | macOS arm64 or universal2 wheels |
|---|---|---|---|
| numpy | 2.5.4 (2026-10-10) | >=3.12 | cp312, cp313, cp314, cp315 |
| pygame-ce | 2.5.8 (2026-08-09) | >=3.10 | cp310 to cp315 (universal2), pp311 |
| pygame | 2.6.1 (2024-09-29) | >=3.6 | cp38 to cp313 only, **no cp314** |
| torch | 2.15.0 (2026-10-28) | >=3.11 | cp311 to cp315 (cp314t and cp315t free-threaded too) |

Python 3.12 and 3.13 would also work for all three, and 3.13 would also allow upstream pygame. Neither has an advantage here. 3.14 is installed and fully supported.

## pygame vs pygame-ce

- **pygame-ce** (Pygame Community Edition, a fork by most of the former active pygame maintainers): it releases regularly. Recent releases were 2.5.6 (2025-10), 2.5.7 (2026-03) and 2.5.8 (2026-08). 2.5.8 added Python 3.15, `Window.handle` and MOUSEWHEEL position, and continued SDL3 porting work. Its API is a drop-in superset (`import pygame`). It adds extras such as `pygame.Window` and `transform.pixelate`.
- **pygame** (upstream): its last release was 2.6.1 (2024-09). It has no wheels for 3.14 or newer, so in practice it is stalled.
- **Choose pygame-ce.**

## macOS issues and how to handle them

1. **Keyboard input and focus.** The old "keys go to the Terminal, window never gets focus" reports were mostly SDL1 and pygame 1.9 era, fixed under SDL2 (pygame issue #203, Stack Overflow 47838446, pygame-users mailing list 2010 and 2020). The old workarounds were `pythonw` and pyobjc. They should not be needed with SDL2 wheels.
   - If a window opens behind the terminal or IDE, click it once.
   - For driving, call `pygame.key.get_pressed()` every frame for held keys rather than relying on KEYDOWN repeat.
   - Pump events every frame (`pygame.event.get()`). Otherwise macOS shows the beachball and stops delivering input.
   - Not tested here, because no GUI window was opened.
2. **Retina/HiDPI.** pygame-ce does not request `SDL_WINDOW_ALLOW_HIGHDPI` by default. A 512x512 window is therefore 512x512 *points*: the backing buffer is 512 px, and macOS upscales it 2x on Retina. The result is the right physical size, with very slight softening from the compositor. That is acceptable for a pixel-art 64x64 view.
3. **Nearest-neighbour upscaling**, either of two equivalent options:
   - (a) Render the 64x64 frame to a Surface, then call `pygame.transform.scale(surf, (W, H))`. This is nearest-neighbour; `smoothscale` is the bilinear one. Then blit to a W x H window.
   - (b) Call `pygame.display.set_mode((64, 64), pygame.SCALED)`. SDL then upscales the logical 64x64 with integer scaling. The default render scale quality is nearest unless `SDL_HINT_RENDER_SCALE_QUALITY` or `PYGAME_FORCE_SCALE=photo` is set. Source: pygame-ce `src_c/display.c`, via Context7.
   - Option (a) is simpler and gives an explicit window size.
   - Use `pygame.surfarray.make_surface(frame.swapaxes(0, 1))` or `blit_array`, because surfarray is indexed (x, y), not (row, col).
4. **SDL2 vs SDL3.** The current pygame-ce wheels bundle SDL2 (2.32.10). SDL3 support is in progress behind compile flags and is not what the wheels ship. Nothing to do now, but keep pygame-ce pinned to `>=2.5.8,<2.6` so a future SDL3-based major release does not arrive unannounced.
5. **torch/MPS.** No special index URL is needed on macOS, because the PyPI `torch` wheel includes MPS. Some ops still fall back or error on MPS. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if you hit "not implemented for MPS".

## Sources

- PyPI JSON: https://pypi.org/pypi/numpy/json, https://pypi.org/pypi/pygame-ce/json, https://pypi.org/pypi/pygame/json, https://pypi.org/pypi/torch/json (fetched 2026-11-03)
- pygame-ce 2.5.8 release notes: https://github.com/pygame-community/pygame-ce/releases/tag/2.5.8
- pygame-ce docs and source via Context7 `/pygame-community/pygame-ce` (`display.c` SCALED logic, `PYGAME_FORCE_SCALE`)
- macOS focus history: https://github.com/pygame/pygame/issues/203, https://stackoverflow.com/questions/47838446
- Empirical test: uv venv with Python 3.14.6 in a session scratch dir (not in the repo)
