# Stable Retro Migration

## Why migrate

`nes-py` is simple and worked for the first scaffold, but repeated Contra
training runs can produce native emulator errors such as:

```text
failed to execute opcode: 23
```

That points to emulator-level instability, likely around reset/restore or ROM
execution state. Stable Retro is a better fit because it is built around
RL-style game integrations and disk-backed `.state` files.

## Target design

```text
Stable Retro integration
├─ data.json       # RAM variables
├─ scenario.json   # baseline done/reward definitions
├─ metadata.json   # default state name
├─ rom.sha         # hash of local ROM, no ROM committed
└─ Level1.state    # local generated savestate, ignored by git
```

Training config:

```yaml
env:
  backend: stable-retro
  stable_retro_game: Contra-Nes
  stable_retro_state: Level1
  stable_retro_integration_path: integrations
```

## Current local blocker

On this Windows Python 3.13 venv, installing `stable-retro` tried to build from
source and failed while building the native wheel. If this repeats, create a
Python 3.11 venv for the Stable Retro backend:

```powershell
py -3.11 -m venv .venv-retro
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv-retro\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -e ".[dev,stable-retro]"
```

Use a separate venv for Stable Retro. Do not install the `nes-py` extra in this
same environment; `nes-py` and `stable-retro` currently require incompatible
`pyglet` versions.

Then check the backend:

```powershell
contra-rl retro-status --game Contra-Nes --integration-path .\integrations
```

## ROM import

Stable Retro does not train directly from `roms/Contra.nes`. It expects the ROM
to be imported into a game integration.

Try:

```powershell
python -m stable_retro.import .\roms
```

The project includes:

```text
integrations/Contra-Nes/rom.sha
```

matching the current local ROM SHA-1:

```text
f8c3b6aa4c26371020224284159e2f5a23645daf
```

The ROM itself remains ignored by git.

## Savestate

After Stable Retro is installed and the ROM is visible, create:

```text
integrations/Contra-Nes/Level1.state
```

at the first playable frame of level 1. This file is generated locally and
ignored by git.

## Training

Once `retro-status` can see `Contra-Nes` and `Level1.state` exists:

```powershell
contra-rl train --config .\configs\ppo_stable_retro.yaml --rom .\roms\Contra.nes --total-timesteps 10000 --n-envs 1 --run-name retro_smoke --device cuda
```

Recurrent PPO is available through `sb3-contrib`:

```bash
pip install -e ".[dev,stable-retro,recurrent]"
```

Use:

```bash
contra-rl train --config ./configs/ppo_recurrent_stable_retro.yaml --rom ./roms/Contra.nes --total-timesteps 1000000 --n-envs 8 --run-name retro_1m_recurrent_v1 --device cuda
```

`eval` and `play` carry the LSTM hidden state automatically when the config uses:

```yaml
algorithm: RecurrentPPO
policy: CnnLstmPolicy
```

For multiple environments, Stable Retro must use subprocess vectorization. Its
native emulator allows only one emulator instance per process. The training code
therefore uses `SubprocVecEnv` automatically when:

```yaml
env:
  backend: stable-retro
  n_envs: 2
```

The default subprocess start method is:

```yaml
env:
  vec_env_start_method: forkserver
```

If WSL has process startup issues, try `spawn`.

Stuck detection is based on useful events, not movement alone:

```text
useful event = new max xscroll OR score increase
```

This prevents the agent from being marked stuck while it pauses to shoot enemies
and earns score.

The training callback logs action preference metrics:

```text
actions/right+B_rate
actions/down+B_rate
actions/right+up+B_rate
actions/selected_index
```

Use these to verify whether the policy actually explores the available Contra
actions or collapses into plain forward movement.

Progress reward is bucketed in the Stable Retro baseline:

```yaml
env:
  progress_reward_mode: bucket
  progress_bucket_size: 32
  progress_reward_per_bucket: 0.5
  progress_reward_start_x: 128
```

This avoids over-rewarding the safe opening runway where the agent can run right
before enemies become meaningful.

Then:

```powershell
contra-rl play --config .\configs\ppo_stable_retro.yaml --checkpoint .\runs\retro_smoke\final_model.zip --rom .\roms\Contra.nes --device cuda --fps 60 --scale 3
```
