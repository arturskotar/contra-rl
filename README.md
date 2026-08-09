# Contra RL

Train a reinforcement learning agent to play the NES version of Contra.

This project is intentionally lightweight: it uses PyTorch, Stable-Baselines3, Gymnasium, and nes-py instead of Ray RLlib.

## Recommended local setup

- Python: 3.13
- GPU: NVIDIA RTX 3080 or similar
- CUDA: use the CUDA-enabled PyTorch wheel matching current PyTorch support; with a modern NVIDIA driver, start with the default PyTorch pip install.

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

## Project status

Scaffold only. The next step is implementing the Gymnasium-compatible Contra environment and smoke test.
