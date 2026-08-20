# Contra RL

Train a reinforcement learning agent to play the NES version of Contra.

This project is intentionally lightweight: it uses PyTorch, Stable-Baselines3, and
Gymnasium instead of Ray RLlib.

The original backend was `nes-py`. Because it can hit native emulator errors
under repeated Contra training resets, the project now supports a second backend:
`stable-retro`.

## Recommended local setup

- Python: 3.13
- GPU: NVIDIA RTX 3080 or similar
- CUDA: use the CUDA-enabled PyTorch wheel matching current PyTorch support; with a modern NVIDIA driver, start with the default PyTorch pip install.

PowerShell activation on Windows:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

The bypass is process-local; it applies only to the current PowerShell window.

Install one emulator backend per virtual environment:

```powershell
# Existing nes-py backend
pip install -e ".[dev,nes-py]"

# Stable Retro backend
pip install -e ".[dev,stable-retro]"
```

Do not install both backends in the same venv. `nes-py` and `stable-retro`
currently require incompatible `pyglet` versions.

## Legal ROM note

This repository does not include a Contra ROM. Place your own legally obtained ROM at:

```text
roms/Contra.nes
```

The `roms/` directory is ignored by Git.

## Planned commands

```powershell
contra-rl smoke --rom .\roms\Contra.nes
contra-rl watch --rom .\roms\Contra.nes --policy run-jump-shoot --fps 30
contra-rl train --config .\configs\ppo_baseline.yaml --rom .\roms\Contra.nes
contra-rl eval --checkpoint .\runs\ppo_baseline\best.zip --rom .\roms\Contra.nes --record
```

## Stable Retro backend

The Stable Retro backend uses named game integrations and disk-backed save states
instead of title-screen button mashing.

Install attempt:

```powershell
pip install -e ".[dev,stable-retro]"
```

Then check integration visibility:

```powershell
contra-rl retro-status --game Contra-Nes --integration-path .\integrations
```

Train with:

```powershell
contra-rl train --config .\configs\ppo_stable_retro.yaml --rom .\roms\Contra.nes
```

Long Stable Retro training session from WSL:

```bash
cd /mnt/c/Users/artur/projects/reinfroce/contra-rl
TOTAL_TIMESTEPS=2000000 N_ENVS=8 RUN_NAME=retro_2m_env8_v1 \
  bash scripts/train_stable_retro_wsl.sh
```

The script creates/uses `.venv-retro-wsl`, checks CUDA, verifies/imports the ROM
for Stable Retro, starts TensorBoard on `http://localhost:6006`, and runs
training.

Recurrent PPO/LSTM run:

```bash
cd /mnt/c/Users/artur/projects/reinfroce/contra-rl
CONFIG=./configs/ppo_recurrent_stable_retro.yaml \
TOTAL_TIMESTEPS=1000000 \
N_ENVS=8 \
RUN_NAME=retro_1m_recurrent_v1 \
INSTALL_DEPS=0 \
bash scripts/train_stable_retro_wsl.sh
```

To transfer the move-forward behaviour from a compatible full-control
checkpoint into the kill-first reward experiment, add `RESUME_CHECKPOINT` and
use a new run name. The checkpoint must use the same 36 actions, grayscale
frame stack, and CNN/LSTM architecture.

```bash
CONFIG=./configs/ppo_recurrent_stable_retro_kills.yaml \
RESUME_CHECKPOINT=./runs/retro_5m_full_gray_v1/checkpoints/retro_5m_full_gray_v1_5000000_steps.zip \
TOTAL_TIMESTEPS=5000000 \
N_ENVS=8 \
RUN_NAME=retro_5m_kills_from_full_v1 \
INSTALL_DEPS=0 \
bash scripts/train_stable_retro_wsl.sh
```

Full-control Recurrent PPO experiment (36 directional/A/B combinations and a
detail-preserving custom CNN):

```bash
cd /mnt/c/Users/artur/projects/reinfroce/contra-rl
CONFIG=./configs/ppo_recurrent_stable_retro_full.yaml \
TOTAL_TIMESTEPS=5000000 \
N_ENVS=8 \
RUN_NAME=retro_5m_full_gray_v1 \
INSTALL_DEPS=0 \
bash scripts/train_stable_retro_wsl.sh
```

That full-control configuration uses all available lives in one episode. A
non-final death costs only `-5` and gameplay continues from the respawn; losing
the last life adds `-95`, for a total final-death penalty of `-100`. The highest
horizontal progress remains an episode-wide maximum, so repeated deaths cannot
farm opening-section progress reward.

`configs/ppo_recurrent_stable_retro_full_rgb.yaml` is the matching RGB
experiment. It differs only in the observation colour mode, so compare it
against the grayscale run at equal timesteps and with the same seed.

Kill-first reward ablation: score is the only positive signal. It removes all
direct movement reward and adds an end-of-episode efficiency penalty:
`-5 * episode_steps / max(max_x, 256)`, capped at `-25`.
After 120 emulator frames without a new maximum X position or score event, it
also applies `-0.01` every emulator frame, making inactivity more costly than
a non-final death well before the stuck truncation.

The long-combat config also carries idle penalties as a capped debt. It refunds
up to `0.01` per pixel only when the agent sets a new episode maximum X; it does
not reward movement inside already explored ground or create a net-positive
movement bonus.

```bash
cd /mnt/c/Users/artur/projects/reinfroce/contra-rl
CONFIG=./configs/ppo_recurrent_stable_retro_kills.yaml \
TOTAL_TIMESTEPS=5000000 \
N_ENVS=8 \
RUN_NAME=retro_5m_kills_v1 \
INSTALL_DEPS=0 \
bash scripts/train_stable_retro_wsl.sh
```

## Project status

Gymnasium environment, visual playback, PPO/DQN training, evaluation videos, and
custom metrics are implemented. Current migration work is focused on replacing
`nes-py` with `stable-retro` for reliable save-state resets.
