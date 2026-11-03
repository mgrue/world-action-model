# Expert API, metrics and CLI

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 02-instruction-set.md, 04-expert-variation.md
Parent: ../MAP.md

## Question

What is the interface contract Step 3 (dataset recording) and Step 9 (closed-loop evaluation) build on?

- Module and types: e.g. `toycar.expert` with `Instruction`, `ExpertParams`, `Expert(track, instruction, params, seed)`, `reset(state, info)`, `act(state, info) -> Action`; stateful (branch decision memory) vs pure.
- Privileged inputs: the expert reads State and Info (not Frames) — confirm, and whether it needs anything the Step 1 env doesn't expose (e.g. a per-passage branch history rather than only `last_branch`).
- **Metrics shared with Step 9**: laps completed, Crash rate, instruction followed (Direction held, no wrong-way; correct branch on every Fork passage where one is required; for open-branch instructions, the branch-choice split). Exact definitions and where they live (a reusable episode-scoring function).
- CLI: `python -m toycar.expert` to watch the expert drive (reusing drive.py's window/panel) and to run the headless reliability evaluation with a summary table.

Check: typed signatures, metric definitions precise enough that Step 9 can score a learned policy with the same function, and a CLI usage block.

## Comments
- Q1 module → `src/toycar/expert.py`: `Instruction`, `INSTRUCTIONS`, frozen `ExpertParams(a_lat_plan=5.0, v_cap=10.0, burst_peak=0.4, burst_gap_s=3.0, p_outer=0.5)`, frozen `ExpertAction(clean, executed, perturbed)`, and `Expert(track, instruction, params)` with `reset(seed, state, info)`, `act(state, info) -> ExpertAction`, `planned_branch`, and a pure `control(state, branch, direction)`. **Exit detection is internal** (act compares the previous `info.branch`), and **noise lives inside the expert** (same seed, both streams). `control` is the clean law used by the feasibility rollout and by future relabelling (DAgger / Step 3). The expert reads only State + Info, never Frames.
- Q2 scoring → policy-agnostic `score_episode(track, instruction, infos) -> EpisodeScore`, fed only the Info stream. **Step 1 addition: `Info.passages: int`**, incremented exactly when `last_branch` is set (otherwise two consecutive same-branch passages are indistinguishable). Scored passages = all completed passages except the first one if the episode started inside the fork section. Correct = branch == Instruction (`outer`/`inner`); under `any` every passage is correct and the split is reported. `instruction_followed = correct == scored and wrong_way_ticks == 0`; `success = not crashed and instruction_followed`. `summarize(scores)` aggregates. The expert's internal exempt flag stays internal.
- Q3 CLI → `python -m toycar.expert show` (drive window and keys; the panel adds instruction, planned branch, clean vs executed steer, burst indicator, passages, last_branch; noise on by default, `--clean`, `--param K=V`) and `python -m toycar.expert eval` (seeds 0..n−1 per Instruction × Direction, parallel, `score_episode` per episode, a summary table incl. safety-rule episodes; **non-zero exit on any crash or wrong passage**, which makes it the Step 2 acceptance check). No `--record`: recording is Step 3's job.

**Resolution (2026-11-03):**

```python
# src/toycar/expert.py
Instruction = Literal["outer", "inner", "any"]
INSTRUCTIONS: tuple[Instruction, ...] = ("outer", "inner", "any")   # frozen order = embedding id

@dataclass(frozen=True)
class ExpertParams:
    a_lat_plan: float = 5.0; v_cap: float = 10.0
    burst_peak: float = 0.4; burst_gap_s: float = 3.0; p_outer: float = 0.5

@dataclass(frozen=True)
class ExpertAction:
    clean: tuple[float, float]; executed: tuple[float, float]; perturbed: bool

class Expert:
    def __init__(self, track: Track, instruction: Instruction, params: ExpertParams = ExpertParams()) -> None
    def reset(self, seed: int, state: State, info: Info) -> None
    def act(self, state: State, info: Info) -> ExpertAction
    @property
    def planned_branch(self) -> Branch
    @property
    def safety_switches(self) -> int
    def control(self, state: State, branch: Branch, direction: Direction) -> tuple[float, float]

@dataclass(frozen=True)
class EpisodeScore:
    instruction: Instruction; direction: Direction; steps: int; termination: Termination
    crashed: bool; laps: int; progress_m: float; wrong_way_ticks: int
    passages: tuple[Branch, ...]; scored: int; correct: int
    instruction_followed: bool; success: bool

def score_episode(track: Track, instruction: Instruction, infos: Sequence[Info]) -> EpisodeScore
def summarize(scores: Sequence[EpisodeScore]) -> dict
```

- Caller loop: `a = expert.act(s, info); r = env.step(a.executed)`; Step 3 records `a.clean`, `a.executed`, `a.perturbed` (and may record `planned_branch`).
- CLI: `uv run python -m toycar.expert show [--seed N] [--instruction …] [--direction …] [--clean] [--param K=V …]` and `uv run python -m toycar.expert eval [--n 1000] [--instruction …] [--direction …] [--clean] [--param K=V …] [--workers K]`.
- Step 1 change: `Info.passages` (implemented with this resolution; see the Step 1 Env API ticket).
