# Step 3 spec: `toycar.dataset` recorded expert data and training windows

Buildable spec for PLAN.md Step 3: record expert episodes and cut them into training **Windows**. It also fixes the **interface contract** that Steps 4–10 build on: the on-disk format, the Window definition and its time alignment, the Splits and seed blocks, the visual randomisation and the reserved **Probes**, and a loader that yields Windows without depending on model code.

Sources: every decision here was made in a ticket of `MAP.md`, and each section links to its ticket. Vocabulary follows `CONTEXT.md` at the repo root (Track, Route, Station, Instruction, Passage, Split, Window, Probe, …). The Step 1 contract is `.scratch/step1-simulator/SPEC.md` and the Step 2 contract is `.scratch/step2-expert/SPEC.md`; this spec only adds to them. Measurements are in `research/01-storage-and-loading.md`; prototypes are in `prototypes/`. Section 14 lists clarifications made while compiling that weren't asked explicitly.

PLAN.md check: "plotting a window shows frames and actions that line up in time" (§10, §12).

---

## 1. Scope and layout

- New module **`src/toycar/dataset.py`** with a CLI in the same module (`python -m toycar.dataset`). **numpy only**; no torch anywhere in `toycar`. matplotlib is used only by `plot`, imported lazily, from a new `viz` dependency group.
- `pyproject.toml`: add `[dependency-groups] viz = ["matplotlib"]` (pin a range that installs on Python 3.14 at implementation time and lock it).
- `.gitignore`: add `data/`.
- New tests: `tests/test_dataset.py`.
- Step 1/2 dependencies: `CarEnv`, `StartSpec`, `State`, `Info`, `render`, `RenderParams`, `Track`, `TrackParams`, `step_state`, `Expert`, `ExpertParams`, `INSTRUCTIONS`, `start_spec` (Step 2 recording rule).
- **Already done during this map:** the Step 1 renderer caches the colour-independent background layout and applies colours per call. The output is bit-identical, and a new colour set costs about 0.1 ms instead of 16 ms (commit 55a6ef4; Step 1 SPEC §5 revised; [ticket 03](issues/03-visual-randomisation.md)). Step 3 relies on it for per-episode colours.

---

## 2. Recorded set

([Episode mix, dataset size and splits](issues/02-episode-mix-and-splits.md))

```python
Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")
SEED_BLOCKS: dict[Split, int] = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000}
PROBE_SEED_BLOCK = 4_000_000          # optional wide_fork recordings (Step 10)
CLOSED_LOOP_SEED_BLOCK = 9_000_000    # Step 9 closed-loop evaluation; never recorded
DEFAULT_EPISODES: dict[Split, int] = {"train": 4800, "val": 600, "test": 600}
EPISODE_TICKS = 300
COMBINATIONS: tuple[tuple[Instruction, Direction], ...] = tuple((i, d) for i in INSTRUCTIONS for d in ("cw", "ccw"))
```

| Split | Episodes | Per Instruction × Direction | Seed of episode k |
|---|---|---|---|
| train | 4800 | 800 | `1_000_000 + k` |
| val | 600 | 100 | `2_000_000 + k` |
| test | 600 | 100 | `3_000_000 + k` |

- Step 2 `eval` and the pytest suites keep seeds `0 … 999`. All blocks are disjoint, and a Split can grow without shifting any other.
- **Episode k of a Split:** `(instruction, direction) = COMBINATIONS[k % 6]`; start with `start_spec(instruction, direction)`, i.e. `StartSpec(direction=direction, route=None if instruction == "any" else instruction)`.
- **Length:** `EPISODE_TICKS = 300` ticks (20 s, about 1.4 laps, about 1.15 Passages), stored per episode.
- **Start ranges:** Step 1 `reset` defaults, unchanged (any Station, lateral U[±1 m], heading U[±10°], speed U[0, 6] m/s). "Randomise speed" means the start speed only.
- **Expert:** `ExpertParams()` defaults for every episode (`a_lat_plan = 5.0`, `v_cap = 10`, `burst_peak = 0.4`, `burst_gap_s = 3.0`, `p_outer = 0.5`).
- **Track:** `TrackParams()` defaults.
- **Crash rule:** if any episode ends with `termination == "crash"`, `record` raises `RuntimeError` naming the Split, k, seed, Instruction and Direction, and writes nothing (§5.3). The only accepted ending is `"max_steps"` after exactly `EPISODE_TICKS` ticks.
- Totals: 6000 episodes, 1.8M ticks, about 0.98M usable Windows (train about 0.78M), about 6900 Passages.

---

## 3. Visual randomisation and Probes

([Minor-visual randomisation and held-out world changes](issues/03-visual-randomisation.md))

### 3.1 Per-episode colour jitter

```python
COLOUR_FIELDS = ("grass", "road", "kerb", "body", "nose")      # order is part of the contract
BRIGHTNESS_RANGE = (0.85, 1.15)
CHANNEL_OFFSET = 10.0

def sample_render_params(seed: int) -> RenderParams:
    rng = np.random.default_rng([seed, 3])                    # visual stream; expert uses [seed, 1], [seed, 2]
    colours = {}
    for name in COLOUR_FIELDS:
        default = getattr(RenderParams(), name)
        b = rng.uniform(*BRIGHTNESS_RANGE)
        colours[name] = tuple(int(np.clip(round(c * b + rng.uniform(-CHANNEL_OFFSET, CHANNEL_OFFSET)), 0, 255))
                              for c in default)               # channels R, G, B in order
    return RenderParams(**colours)                            # supersample stays 4
```

- **Per episode, never per Frame.** The same rule applies to every Split.
- Visuals never affect driving: the expert reads only State and Info, physics never sees `RenderParams`, and the stream is separate.
- **No pixel augmentation** in the dataset. Any augmentation is a model-side option (Steps 4/6).
- The check is the contact sheet `prototypes/out/visual_contact_sheet.png`: the car stays readable at every extreme. Worst-case colour distances over 10,000 sampled episodes: nose–body 140, body–road 107, road–grass 26.6 (the kerb still outlines the road).

### 3.2 Probes (reserved for Step 10, never in training)

```python
@dataclass(frozen=True)
class Probe:
    name: str
    colours: dict[str, RGB | str]        # field → RGB, or the name of another field of the same episode to copy
    track_params: TrackParams | None     # None = default Track
    def apply(self, rp: RenderParams) -> RenderParams

PROBES: dict[str, Probe] = {
    "brown_grass": Probe("brown_grass", {"grass": (120, 100, 50)}, None),
    "light_road":  Probe("light_road",  {"road": (140, 140, 145)}, None),
    "blue_car":    Probe("blue_car",    {"body": (40, 80, 220)}, None),
    "no_kerb":     Probe("no_kerb",     {"kerb": "road"}, None),
    "wide_fork":   Probe("wide_fork",   {}, TrackParams(bulge_m=7.0)),
}
```

- A **colour Probe** overrides only its fields on top of the episode's own jittered `RenderParams` (`no_kerb` copies that episode's road colour). It re-renders the stored episodes, so it needs no re-recording.
- **`wide_fork`** changes driving (branch arc 4.88 → 4.64 m). Every loader method that takes `probe=` raises `ValueError` for it. Step 9/10 evaluates it closed-loop; an optional recording uses seed block `PROBE_SEED_BLOCK` after Step 2 `eval` has been re-run on that Track.

---

## 4. Storage format

([Storage format and loading throughput](issues/01-storage-and-loading.md), [Recorder, loader API, CLI](issues/05-recorder-loader-and-plot.md))

**Frames are not stored.** They are re-rendered from stored States and per-episode `RenderParams`. Re-rendering is pure and bit-identical across processes (measured), costs about 0.07 ms per Frame, and gives about 900 Windows/s per process, against about 13 Windows/s consumed by training on MPS. Raw Frames would be about 22 GB for this set (6000 × 301 × 12,288 B).

### 4.1 Directory

```
data/<name>/                 # default data/step3-v1 (relative to CWD), gitignored
  manifest.json
  train.npz  val.npz  test.npz
```

### 4.2 Split files

Each `<split>.npz` is written with `np.savez` (uncompressed, no pickle), E = episodes in the Split, T = `EPISODE_TICKS`:

| Array | Shape | dtype | Content |
|---|---|---|---|
| `states` | (E, T+1, 4) | float64 | (x, y, θ, v) of s₀ … s_T |
| `clean` | (E, T, 2) | float32 | expert label at ticks 0 … T−1, columns (steer, throttle) |
| `executed` | (E, T, 2) | float32 | action passed to `env.step` |
| `perturbed` | (E, T) | bool | burst active on that tick |
| `planned_branch` | (E, T) | int8 | `Expert.planned_branch` after `act` at that tick (0 outer, 1 inner) |
| `station_m` | (E, T+1) | float32 | `Info.station_m` of s₀ … s_T |
| `branch` | (E, T+1) | int8 | `Info.branch`: −1 outside the fork section, 0 outer, 1 inner |
| `k` | (E,) | int32 | index within the Split |
| `seed` | (E,) | int64 | `SEED_BLOCKS[split] + k` |
| `instruction` | (E,) | int8 | `INSTRUCTIONS` id |
| `direction` | (E,) | int8 | 0 cw, 1 ccw |
| `start_route` | (E,) | int8 | the requested `StartSpec.route`: 0 outer, 1 inner, −1 random (`any`) |
| `length` | (E,) | int32 | ticks (= T) |
| `render_params` | (E, 5, 3) | uint8 | colours in `COLOUR_FIELDS` order |
| `frames_sha256` | (E,) | `<U64` | SHA-256 hex over the bytes of all T+1 Frames rendered at 64×64 with the episode's `RenderParams`, in tick order |

About 16 KB per episode, about 100 MB in total; loaded fully into RAM.

### 4.3 Manifest

`manifest.json` holds:

- `format_version: 1`.
- **Provenance:**
  - `toycar_version`, `python_version`, `numpy_version`, `platform`, `created` (ISO 8601 UTC);
  - `git_commit` and `git_dirty` (from `git rev-parse HEAD` and `git status --porcelain` in the package's directory; `null` if not in a git checkout);
  - `source_sha256`: the SHA-256 of the source bytes of `render.py`, `track.py`, `physics.py`, `env.py`, `expert.py`.
- **Recipe:**
  - `episode_ticks`, `episodes` per Split, `seed_blocks`;
  - `combinations` (the list, and the rule `"k mod 6"`), `start_rule` (`"start_spec(instruction, direction)"`);
  - `expert_params`, `track_params`, `render_defaults`, and `colour_jitter` (`fields`, `brightness_range`, `channel_offset`, `stream: [seed, 3]`).
- **Summary:**
  - per Split: episodes, ticks, and usable Windows at H = 16 with the clean-future filter;
  - per Split × Instruction × Direction: episodes and usable Windows;
  - `any_outer_fraction` (completed Passages of `any` episodes; the first Passage of an episode that started inside the fork section is excluded, as in Step 2 `score_episode`);
  - `perturbed_fraction`, `approaches_section_fraction`.

---

## 5. Recording

### 5.1 One episode

```
rp    = sample_render_params(seed)
env   = CarEnv(track=Track(TrackParams()), max_steps=T)
state, info = env.reset(seed, start_spec(instruction, direction), render_params=rp)
expert = Expert(env.track, instruction, ExpertParams()); expert.reset(seed, state, info)
record s₀, info₀.station_m, info₀.branch; hash render(s₀, track, rp)
for t in 0 … T−1:
    a = expert.act(state, info)
    record clean[t], executed[t], perturbed[t], planned_branch[t] = expert.planned_branch
    r = env.step(a.executed)
    state, info = r.state, r.info
    record s_{t+1}, station_m[t+1], branch[t+1]; hash render(s_{t+1}, track, rp)
    if r.terminated: raise RuntimeError(...)        # crash
assert r.truncated and info.steps == T
```

The episode result is plain numpy arrays (picklable for workers). Passage completions for the summary come from `Info.passages`/`last_branch` during recording.

### 5.2 Whole dataset

`record(out, *, episodes=DEFAULT_EPISODES, length=EPISODE_TICKS, workers=None, force=False) -> Path`:
1. If `out` exists and not `force`, raise `FileExistsError`. Remove any stale `out.tmp`.
2. Run all (split, k) jobs with `ProcessPoolExecutor(max_workers=workers or os.cpu_count())` and `pool.map` in a fixed order (Splits in `SPLITS` order, k ascending). With `workers == 1`, run in-process.
3. Write the Split files and the manifest into `out.tmp/`. Then, if `force`, remove the old `out`, and rename `out.tmp` → `out`.
4. On any exception (including a crash or Ctrl-C), remove `out.tmp` and re-raise; `out` is untouched.

The written files are byte-identical for any `workers` value, except for `created` in the manifest. Expected runtime is under about a minute on an M-series Mac.

### 5.3 Determinism

An episode is a pure function of (Split, k, length, code). States, actions and Frames are bit-identical on the same machine with pinned Python/numpy (Step 1 guarantee). Random streams per seed: env `default_rng(seed)`, expert coin `[seed, 1]`, expert noise `[seed, 2]`, colours `[seed, 3]`, eval subset `[seed, 4]` (§6.4).

---

## 6. Windows

([Window definition and time alignment](issues/04-window-definition.md))

### 6.1 Definition

For an episode with T ticks, horizon H (default 16) and future-Frame selection `future_frames` ⊆ {0 … H−1} (default all, ascending, F = its length), the Window at tick t is:

| Field | Value | Shape / dtype |
|---|---|---|
| `obs_frame` | `render(sₜ, track, rp)` | (64, 64, 3) uint8 |
| `state` | `sₜ.features()` = (x/40, y/40, cos θ, sin θ, v/10) | (5,) float32 |
| `state_raw` | (x, y, θ, v) of sₜ | (4,) float64 |
| `actions` | `clean[t : t+H]` | (H, 2) float32 |
| `executed` | `executed[t : t+H]` | (H, 2) float32 |
| `perturbed` | `perturbed[t : t+H]` | (H,) bool |
| `future_frames` | `render(s_{t+1+k}, track, rp)` for k in `future_frames` | (F, 64, 64, 3) uint8 |
| `future_index` | those k | (F,) int64 |
| `instruction` | `INSTRUCTIONS` id | () int64 |
| `direction` | 0 cw, 1 ccw | () int64 |
| `episode`, `t`, `seed` | episode index in the Split, tick, seed | () int64 each |
| `approaches_section` | `branch[t] == −1` and any `branch[t+1 … t+H] != −1` | () bool |
| `in_section` | `branch[t] != −1` | () bool |

`rp` is the episode's `RenderParams`, or `probe.apply(rp)` when a colour Probe is given.

- **Alignment:** action k ↔ future Frame k = f_{t+1+k}, the Frame rendered after action k is applied. Each action sits on the time step of the Frame it causes. Step 9 replanning uses the same indexing (observe fₜ, sₜ; execute aₜ, a_{t+1}, …).
- Unrequested future Frames are never rendered. `future_frames=()` is allowed (no video target).
- Frames are uint8 HWC as rendered; any conversion (CHW, float, scaling) belongs to the model.

### 6.2 Valid t and the clean-future filter

- t is valid iff `0 ≤ t ≤ T − H` and, when `clean_future=True` (default), `not perturbed[t : t+H].any()`. In those Windows `executed == clean` over the whole future. The observed tick may still be off line after an earlier burst: that's the recovery data.
- Windows never cross the episode end; there is no padding.
- **Stride 1:** every valid t is a Window. At T = 300, H = 16 and default noise: 285 positions, about 163 valid per episode (about 57%).

### 6.3 Index

`Windows.index` is an (N, 2) int32 array of (episode, t), ordered by episode then t, built when `windows()` is called (vectorised over the stored arrays). `approaches_section` and `in_section` are (N,) bool arrays aligned with it, so Windows can be sliced without rendering. Shuffling is the training code's job.

### 6.4 Evaluation subsets

`eval_subset(n=2048, **window_kwargs)` returns a fixed Windows subset:
- Episode i (in Split order) gets `n // E + (1 if i < n % E else 0)` Windows.
- They are drawn without replacement from that episode's valid t with `np.random.default_rng([seed_i, 4])`, then sorted.
- If an episode has fewer valid t than its quota, it contributes all of them, and the subset is smaller than n.
- The subset is the same on every open and balanced across Instruction × Direction, because episodes are. It's intended for val and test; it works on train too.

---

## 7. Loader API

([Recorder, loader API, CLI](issues/05-recorder-loader-and-plot.md))

```python
def record(out: str | Path, *, episodes: Mapping[Split, int] = DEFAULT_EPISODES, length: int = EPISODE_TICKS,
           workers: int | None = None, force: bool = False) -> Path

def open_dataset(path: str | Path, *, check: bool = True) -> Dataset

class Dataset:
    path: Path
    manifest: dict
    def split(self, name: Split) -> EpisodeSet
    def verify(self) -> None                         # re-render every episode of every Split; raises on mismatch

@dataclass(frozen=True)
class Episode:
    split: Split; index: int; k: int; seed: int
    instruction: Instruction; direction: Direction; start_route: Branch | None; length: int
    render_params: RenderParams
    states: np.ndarray; clean: np.ndarray; executed: np.ndarray; perturbed: np.ndarray
    planned_branch: np.ndarray; station_m: np.ndarray; branch: np.ndarray
    frames_sha256: str

class EpisodeSet:
    name: Split
    def __len__(self) -> int
    def episode(self, i: int) -> Episode
    def frames(self, i: int, ticks: Sequence[int] | None = None, probe: str | None = None) -> np.ndarray  # (n, 64, 64, 3)
    def windows(self, *, horizon: int = 16, future_frames: Sequence[int] | None = None,
                clean_future: bool = True, probe: str | None = None) -> Windows
    def eval_subset(self, n: int = 2048, **window_kwargs) -> Windows

class Windows:
    index: np.ndarray                # (N, 2) int32
    approaches_section: np.ndarray   # (N,) bool
    in_section: np.ndarray           # (N,) bool
    def __len__(self) -> int
    def __getitem__(self, i: int) -> dict[str, np.ndarray | int | bool]   # §6.1 fields
    def subset(self, mask_or_indices) -> Windows

class FrameItems:                    # Step 4: every Frame of every episode as one item
    def __len__(self) -> int         # Σ (length + 1)
    def __getitem__(self, i: int) -> np.ndarray   # (64, 64, 3) uint8, ordered by episode then tick

def frame_dataset(es: EpisodeSet, probe: str | None = None) -> FrameItems
```

- `Windows` and `FrameItems` work unchanged with PyTorch's `DataLoader` (default collate turns numpy arrays and Python scalars into tensors). They hold only numpy arrays and small Python objects, so they pickle to spawn workers cheaply; the render layout cache is rebuilt once per process.
- Out-of-range `i` raises `IndexError`; an unknown Split or Probe, or `wide_fork`, raises `ValueError`.
- `horizon` must be in [1, T]; `future_frames` entries must be in [0, H−1], distinct, and are sorted.

---

## 8. Integrity checks

- **On open** (`check=True`): for each Split, re-render 4 episodes chosen with `np.random.default_rng([0, split_id])` (split_id = index in `SPLITS`) and compare `frames_sha256`. On mismatch, raise `ValueError` naming Split, episode and seed, with a hint to re-record or check the code version.
- **Code hashes:** compare `manifest.source_sha256` with the current sources. On a difference, `warnings.warn` listing the changed files (and don't raise): a source change needn't change output, and the Frame check is authoritative.
- **`format_version` ≠ 1:** raise `ValueError`.
- **`verify()`** re-renders every episode of every Split and raises on the first mismatch. It also re-simulates one seeded episode per Split from (seed, combination, length) and compares States and actions exactly.

---

## 9. CLI

```
uv run python -m toycar.dataset record [--out data/step3-v1] [--workers K] [--force]
uv run python -m toycar.dataset verify [DIR]
uv run python -m toycar.dataset stats  [DIR]
uv run --group viz python -m toycar.dataset plot [DIR] [--split val] [--episode I --t T | --seed N]
                                                [--approaching] [--probe NAME] [--out window.png]
```

- `DIR` defaults to `data/step3-v1`.
- **`record`** prints progress per Split, the totals and the path. It exits non-zero on a crash or existing output.
- **`verify`** prints `PASS` (exit 0) or the mismatch (exit 1).
- **`stats`** prints:
  - per Split × Instruction × Direction: episodes and usable Windows (H = 16, filter on);
  - the outer fraction under `any`, the perturbed share and the `approaches_section` share;
  - a speed histogram (1 m/s bins, 0–10) over all ticks.
- **`plot`** writes a PNG (default `window.png`) of one full Window (H = 16, all future Frames):
  - **Top row:** `obs_frame` labelled `t`, then the 16 future Frames labelled `k=0 (t+1)` … `k=15 (t+16)`, upscaled with nearest-neighbour.
  - **Below, sharing the same columns:** steer and throttle per k (`clean` solid, `executed` dashed, perturbed k shaded) and v from `states[t+1+k]`. Action k sits directly under future Frame k.
  - **Title:** dataset name, Split, episode, seed, t, Instruction, Direction, `approaches_section`, Probe.
  - **Selection:** `--episode/--t` select exactly. `--seed N` (default 0) picks a seeded random valid Window, restricted to `approaches_section` Windows with `--approaching`. `--probe` re-renders with a colour Probe.

---

## 10. Automated tests (pytest, `tests/test_dataset.py`)

Small datasets in `tmp_path`, e.g. `episodes={"train": 12, "val": 6, "test": 6}`, `length=60`, built once per module.

| Test | Asserts |
|---|---|
| `test_record_layout` | files, array names/shapes/dtypes (§4.2), manifest keys (§4.3); seeds `SEED_BLOCKS[split] + k`; combination `COMBINATIONS[k % 6]`; `start_route` −1 exactly for `any` |
| `test_record_workers_identical` | `workers=1` and `workers=2` give identical arrays in every Split file |
| `test_record_atomic_and_force` | an existing output raises without `force`; an exception injected mid-recording leaves neither `out` nor `out.tmp`; `force=True` replaces |
| `test_crash_aborts` | with `Expert.act` monkeypatched to steer hard into the grass, `record` raises `RuntimeError` naming the seed, and no output exists |
| `test_episode_replays` | re-simulating an episode from its seed and combination gives exactly the stored States, `clean`, `executed`, `perturbed` |
| `test_window_alignment` | for several Windows: `obs_frame == render(sₜ)`, `future_frames[k] == render(s_{t+1+k}) == render(step_state(s_{t+k}, executed[t+k]))`, `actions[k] == clean[t+k]`, `state == sₜ.features()` |
| `test_clean_future_filter` | the index equals a brute-force enumeration for H = 16 and H = 8, with and without the filter; filtered Windows have `executed == actions` |
| `test_future_frames_subset` | `future_frames=(3, 7, 11, 15)` gives shape (4, 64, 64, 3) and `future_index == [3, 7, 11, 15]`; `()` gives (0, 64, 64, 3); invalid selections raise |
| `test_section_flags` | `approaches_section`/`in_section` match the §6.1 definitions from `branch`; at least one `approaches_section` Window exists |
| `test_hash_check` | tampering with one stored `render_params` makes `open_dataset` (or `verify`) raise; a fake `source_sha256` only warns |
| `test_eval_subset_fixed` | same index on two opens; size `min(n, …)`; per-episode quotas as in §6.4 |
| `test_colour_jitter` | `sample_render_params` stays within the brightness/offset bounds for 1000 seeds; recording with jitter disabled (monkeypatched) gives identical States and actions |
| `test_probes` | each colour Probe changes only its field(s) relative to the episode's `RenderParams`; `no_kerb` copies the episode's road colour; `probe="wide_fork"` raises |
| `test_frame_dataset` | length Σ (T+1); item i matches `frames(episode)[tick]` |
| `test_dataloader_compat` | `pytest.importorskip("torch")`: a `DataLoader(windows, batch_size=4, num_workers=0)` batch has the expected tensor shapes |
| `test_cli` | `main(["record", "--out", …])` with a small-size override, then `verify` and `stats` return 0; `plot` writes a PNG (`importorskip("matplotlib")`) |

`test_cli` needs a size override for speed. `record`'s CLI accepts hidden `--episodes train=N,val=N,test=N` and `--length L` options for tests and experiments (§14.9).

---

## 11. Determinism and performance targets

- The same code, machine and pinned Python/numpy give byte-identical Split files for any `--workers`.
- Recording the full set takes about a minute on an M-series Mac.
- `windows()` on train (4800 episodes) builds its index in well under a second.
- Window read at defaults: at least about 500 Windows/s in one process (17 renders at about 0.07 ms plus slicing).

---

## 12. Manual acceptance (PLAN.md Step 3 check)

1. `uv run python -m toycar.dataset record` finishes in about a minute and writes about 100 MB to `data/step3-v1/`. `stats` shows 6000 episodes (4800/600/600, 800/100/100 per combination), about 0.98M usable Windows, and an `any` outer fraction of about 50%.
2. `uv run python -m toycar.dataset verify` prints PASS.
3. **PLAN check:** `uv run --group viz python -m toycar.dataset plot --split val` for a few seeds, plus `--approaching` on an `any` episode and `--probe brown_grass`. Future Frame k shows the effect of the action in column k: steering left visibly turns the car in the following columns, and on an approaching Window the car enters the branch the actions steer towards.
4. `uv run pytest` passes (Steps 1–3).

---

## 13. Hand-offs to later steps

- **Step 4:** `frame_dataset(train)` for the VAE; val Frames for its check. Latent caching is its own storage decision (8×8×16 float16 is 2 KB per Frame, about 3.7 GB for this set).
- **Step 6:** token budget. About 1100 tokens per Window with plain attention caps the batch at 8 on the 16 GB Mac. `horizon` and `future_frames` let Step 6 trade video tokens without re-recording. Frame history (more than one observed Frame) can be added as a loader option later.
- **Step 8:** the test `eval_subset()` is "held-out windows".
- **Step 9:** seed block `CLOSED_LOOP_SEED_BLOCK`. Recovery coverage is thin (lateral > 0.3 m on 3.4% of ticks, speed recovery only from starts); if the policy drifts, prefer DAgger relabelling with `Expert.control` over wider starts.
- **Step 10:**
  - `approaches_section` Windows are where the regression baseline should average into the Island.
  - The colour Probes re-render the test Split.
  - `wide_fork` needs closed-loop runs or a recording on `PROBE_SEED_BLOCK`.

---

## 14. Clarifications made while compiling

These fill gaps between ticket decisions. They don't change any decision. If one looks wrong, reopen the relevant ticket.

1. **Seed-block and Split constants** are module-level exports (`SPLITS`, `SEED_BLOCKS`, `PROBE_SEED_BLOCK`, `CLOSED_LOOP_SEED_BLOCK`, `DEFAULT_EPISODES`, `EPISODE_TICKS`, `COMBINATIONS`), so Steps 4–10 import them instead of re-typing numbers.
2. **`start_route`** stores the *requested* route (−1 for `any`), because `CarEnv` doesn't expose the sampled one. The actual start branch is visible from `branch[0]`, and outside the section both Routes coincide.
3. **`planned_branch[t]`** is read after `act` at tick t, so it is the plan that produced `clean[t]`.
4. **Jitter draw order:** per colour in `COLOUR_FIELDS` order, the brightness first, then the R, G, B offsets; Python `round` (half to even). `supersample` is not jittered.
5. **`frames_sha256`** hashes the concatenated bytes of all T+1 Frames in tick order at 64×64 (one hash per episode, unlike Step 1 recordings' per-Frame hashes) to keep the files small.
6. **Probes** are a small dataclass with an `apply` method rather than a `(RenderParams, TrackParams)` tuple, because a colour Probe overrides fields on top of each episode's jittered colours (and `no_kerb` copies a field).
7. **Window dict extras:** `direction`, `episode`, `t` and `seed` as int64 scalars, and the section flags as bools, so a batch can be traced back and sliced. The Split name is not in the dict (it's the same for the whole `Windows`).
8. **Evaluation subset algorithm** (§6.4): per-episode quotas plus stream `[seed, 4]`, which gives "seeded, spread evenly over episodes and combinations" exactly.
9. **Hidden CLI size options** (`--episodes`, `--length`) exist so tests and experiments don't need the full set; the defaults are the decided set.
10. **`source_sha256`** also covers `env.py` (it produces `Info.branch`/`station_m`, which are stored), alongside the four files named in the ticket.
11. **Hash-check episode choice** uses `default_rng([0, split_id])`, so the same 4 episodes are checked on every open.
12. **`verify`** also re-simulates one episode per Split, so it checks the driving as well as the Frames.
13. **`any_outer_fraction`** in the manifest uses the Step 2 scorer's passage rule (first passage excluded if the episode started in the section).
14. The **performance targets** (§11) are derived from the ticket 01 measurements, not separately decided.

## 15. Out of scope for Step 3

- Model code, normalisation inside models and batching on MPS (Steps 4–8).
- Latent caching (Step 4/6), frame history (Step 6).
- Closed-loop evaluation and DAgger (Step 9); the Step 10 experiments themselves.
- Further changes to the simulator or expert. The one Step 1 change this map needed (recolourable render cache) is already done.
