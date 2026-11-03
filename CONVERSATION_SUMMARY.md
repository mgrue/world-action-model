# Background: FLUX 3 Action and world action models

Summary of an exploratory conversation (November 2026) analysing the `flux-action`
repository from Black Forest Labs, which led to the plan in `PLAN.md`.

## What FLUX 3 Action is
- An open-weights 7B "world action model" released by Black Forest Labs in September 2026.
- Inputs: camera frames, the robot's joint state, and a text instruction.
- Output: the next chunk of robot commands, for example 32 actions (about 2 s at 15 Hz),
  predicted together with the next video frames.
- The `flux-action` repository is the standalone training and inference code: data preparation,
  fine-tuning, export, offline inference, evaluation and a WebSocket server for benchmarks.
- Released checkpoints: a DROID policy for the Franka Panda arm and an SO-101 policy for the
  LeRobot hobby arm, plus a shared base with the frozen encoders.

## Architecture
- A FLUX-style diffusion transformer ("JointSingleSeq"): hidden size 3072, 24 heads.
- Several token streams in one sequence: text, observed frame, future frames, state, action chunk.
- Each stream first passes through 5 private blocks, then all streams go through 28 shared joint
  blocks with full attention, then each stream has its own output head.
- Blocks use a parallel attention + SwiGLU MLP layout with QK RMSNorm, 4-axis RoPE (time, height,
  width, sequence) and adaptive layer norm modulation driven by a per-token noise level.
- No cross-attention: text conditions the rest only by sharing the joint sequence.
- Frozen encoders: a video VAE (32x spatial, 4x temporal downsampling) and Qwen3-VL-4B for text.
- Actions sit on the same time axis as video latents, so RoPE aligns each action with the frame it causes.
- New robots: the shared action backbone is reused; only small input and output heads are new.

## Noising and denoising (flow matching)
- Training mixes clean targets with Gaussian noise at a random level t and trains the network to
  predict the velocity (noise minus clean data).
- Future video latents and future actions are noised together; observed frame, state and text stay clean.
- Inference starts from pure noise and runs a few solver steps (4 for the released DROID model).
- Classifier-free guidance strengthens instruction-following; the DROID recipe guides video but not actions.
- Why denoise actions instead of regressing them: the model can commit to one of several valid
  actions instead of averaging them into an invalid one.
- This is a generative model, not a JEPA: it predicts decodable video latents rather than abstract
  embeddings, and actions are a generated output stream.

## How it is trained
- The repository only fine-tunes. The expensive stages happened at BFL: a video generation model,
  then action pretraining on mixed robot data.
- Fine-tuning recipe: frozen trunk for the first 1000 steps while new heads warm up, then the trunk
  unfreezes; heads at 5x learning rate; two EMA copies; global batch 2048; FSDP2 across 8 GPUs.
- Training windows: 33 frames, the current state, the next 32 actions and the caption.
- Regularization: 10% caption dropout, camera dropout, crop and color jitter.

## What it is used for
- Visuomotor control of fixed-base manipulator arms with a gripper and one to three cameras.
- Also any embodiment with frames and a fixed-width continuous action vector: the repo includes a
  browser-game example (shooter and racer) and a stub for a drone.
- It is an imitation-learning policy: it reproduces behaviors present in demonstration data.
- Released checkpoint reaches about 43% success on NVIDIA's RoboLab-120 benchmark (company-reported),
  first among open models but far from production reliability.

## Fine-tuning for your own robot
- Easiest: SO-101 arm with LeRobot LoRA on one GPU.
- Other arms: index a LeRobot dataset, adapt the SO-101 config, full fine-tune.
- Anything else: write a dataset plug that yields training windows (see `examples/games/dataset.py`).
- Constraints: state width must equal action width; actions should be continuous and bounded.

## RC car idea
- A fixed overhead camera watching the car fits the model better than an onboard camera:
  static background, one moving object, and future-frame prediction becomes meaningful.
- Watch for: the car being tiny in latent space, heading ambiguity, speed invisible in a single
  frame (use history), latency of a 7B model, and the need for many demonstrations.
- For a fixed track, classical tracking plus pure pursuit would beat it on lap time and latency.

## Hardware reality
- Inference and training need Linux and an NVIDIA GPU. The NATTEN CUDA wheels do not exist for macOS.
- The real blocker on a 16 GB Mac is memory: about 14 GB for the transformer plus about 9 GB for
  the text encoder. BF16 inference needs about 32 GB; FP8 variants need CUDA.
- The CPU test suite runs locally (211 passed, 4 skipped), so data work and simulator work can stay on the Mac.
- Renting: 40 GB+ GPU for bf16 inference; 24 GB Ada/Hopper card with FP8 and text-encoder offload;
  multi-GPU for full fine-tuning.

## DROID dataset
- Distributed Robot Interaction Dataset (2024): teleoperated demonstrations on a Franka Panda
  with a wrist camera and two side cameras, recorded across many institutions and scenes.
- Version used by the repo (nvidia/Cosmos3-DROID, success subset): 57,639 episodes, 18.7 million
  frames at 15 Hz (about 346 hours), 53,086 distinct instructions.
- Each episode: three synchronized videos, per-frame robot state, per-frame commands, and a text
  instruction. The command labels are what make it usable for learning control.
- Explore via the Hugging Face dataset page, LeRobot's online visualizer, the metadata parquet
  files, or the repo's two-episode sample pipeline (`docs/prepare.md`).

## Practicality in industry
- For stable, repetitive tasks, programmed robots win on reliability, speed, safety certification,
  cost and debuggability.
- Learned policies target the expensive part of automation: per-task integration engineering.
  They fit high-variety, low-volume work, unstructured inputs, deformable objects, and tasks that
  are easy to demonstrate but hard to specify.
- Today's industrial ML is mostly hybrid: learned perception, classical motion control.

## Who is piloting learned robot policies (as of November 2026, mostly secondary sources)
- Agility Robotics (Digit): tote handling for GXO; agreement with Toyota's Canadian plant.
- Figure AI (Helix VLA model): pilots at BMW Spartanburg, including parts sequencing.
- Apptronik (Apollo): partnerships reported with Mercedes-Benz and Jabil.
- Covariant (RFM-1): warehouse picking, partnership with Otto Group; team largely moved to Amazon in 2024.
- Model providers without named industrial customers found: Physical Intelligence, Skild AI,
  Google DeepMind (Gemini Robotics), NEURA Robotics.
- No public production deployments of FLUX 3 Action itself; its weights may be under a
  non-commercial license (unverified, check before any commercial use).
- Tasks in pilots are simple motions in variable surroundings, not precision work.

## Sources
- FLUX 3 Action: https://huggingface.co/blog/black-forest-labs/flux-3-action
- BFL model page: https://bfl.ai/models/flux-3-action
- BFL docs: https://docs.bfl.ai/flux_3/flux3_action_overview
- The Decoder: https://the-decoder.com/black-forest-labs-launches-flux-3-action-an-open-robotics-ai-model/
- MarkTechPost: https://www.marktechpost.com/2026/09/24/black-forest-labs-releases-flux-3-action-a-7b-open-weights-world-action-model-that-tops-robolab-120/
- Cosmos3-DROID dataset: https://huggingface.co/datasets/nvidia/Cosmos3-DROID
- Humanoid deployments tracker: https://physicalailab.io/en/humanoid-robots-factories-warehouses-deployments-2026/
- Physical AI market report, Oct 2026: https://www.insidedeeptech.com/physical-ai-market-report-october-2026/
- Wikipedia, Figure AI: https://en.wikipedia.org/wiki/Figure_AI
- Wikipedia, Agility Robotics: https://en.wikipedia.org/wiki/Agility_Robotics
- Covariant RFM-1 (TechCrunch): https://techcrunch.com/?p=2677300
- Otto Group and Covariant: https://www.businesswire.com/news/home/20230509005491/en
