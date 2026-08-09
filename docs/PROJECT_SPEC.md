# Contra RL Project Spec

## Goal

Create a clean reinforcement learning project that trains an agent to play NES Contra.

## Stack

- Python 3.13
- PyTorch with CUDA-enabled wheels
- Stable-Baselines3
- Gymnasium
- nes-py
- NumPy / OpenCV / Pillow
- TensorBoard
- Typer / Rich
- pytest / Ruff

## First algorithm

Use PPO with `CnnPolicy`.

Ray RLlib is intentionally excluded from the initial stack because it is overkill for a single-workstation RTX 3080 project.

## First milestone

Implement a Gymnasium-compatible environment and smoke test:

- load a local Contra ROM
- skip the start screen
- run random actions
- log reward components
- record a short evaluation video

## Training target

First benchmark: visible level-1 progress.

Later benchmark: survive longer, defeat enemies, collect powerups, and eventually complete stages.
