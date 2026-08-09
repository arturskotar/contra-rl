# Contra RL Implementation Checklist

This checklist is the build order for the first working version of the Contra RL project.

The short answer: start with the environment, not the model.

If the environment is unreliable, the agent will train on bad signals. The first milestone is not “train PPO”; it is “prove we can repeatedly start Contra, step actions, observe frames, detect death/progress, and reset cleanly.”

## Phase 0: Local setup

- [ ] Install Python 3.13.
- [ ] Create the virtual environment:

  ```powershell
  cd C:\Users\artur\projects\reinfroce\contra-rl
  py -3.13 -m venv .venv
  ```

- [ ] Activate it:

  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
  .\.venv\Scripts\Activate.ps1
  ```

- [ ] Install dependencies:

  ```powershell
  python -m pip install --upgrade pip
  pip install torch torchvision torchaudio
  pip install -e ".[dev]"
  ```

- [ ] Verify GPU:

  ```powershell
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
  ```

- [ ] Expected result:

  ```text
  True
  NVIDIA GeForce RTX 3080
  ```

## Phase 1: ROM handling

- [ ] Put the local Contra ROM here:

  ```text
  roms/Contra.nes
  ```

- [ ] Confirm `roms/` is ignored by Git.
- [ ] Do not commit the ROM.
- [ ] Add CLI validation:

  - [ ] ROM path exists.
  - [ ] ROM path is a file.
  - [ ] Missing ROM gives a clear error.

## Phase 2: Minimal nes-py environment

Goal: create the smallest possible environment that loads the ROM and steps frames.

- [ ] Implement `src/contra_rl/envs/contra_env.py`.
- [ ] Use `nes_py.NESEnv` or a thin wrapper around it.
- [ ] Expose a Gymnasium-compatible API:

  ```python
  reset(seed=None, options=None) -> tuple[obs, info]
  step(action) -> tuple[obs, reward, terminated, truncated, info]
  render()
  close()
  ```

- [ ] Ensure rendering is off during training by default.
- [ ] Add a debug render mode only for manual inspection.
- [ ] Confirm one random episode runs without crashing.

## Phase 3: Controller/action mapping

Goal: convert a small discrete action number into NES button presses.

- [ ] Reuse the action sets in `src/contra_rl/envs/actions.py`.
- [ ] Start with `SIMPLE_MOVEMENT`.
- [ ] Add an action wrapper using `nes_py.wrappers.JoypadSpace`.
- [ ] Confirm all actions are legal.
- [ ] Add a test for action set names:

  - [ ] `RIGHT_ONLY`
  - [ ] `SIMPLE_MOVEMENT`
  - [ ] `COMPLEX_MOVEMENT`

Recommended progression:

1. `RIGHT_ONLY` for reset/smoke tests.
2. `SIMPLE_MOVEMENT` for first PPO run.
3. `COMPLEX_MOVEMENT` only after the simple agent learns forward progress.

## Phase 4: Start screen / loading screen skip

Goal: reset should always land at playable gameplay, not the title/menu screen.

Do this before reward design.

- [ ] Implement deterministic startup frame advancement.
- [ ] Avoid wall-clock sleeps.
- [ ] Press `START` for the required frame window.
- [ ] Advance idle frames until gameplay is reached.
- [ ] Detect gameplay using RAM or stable visual heuristics.
- [ ] Store no emulator state until after the game is actually playable.
- [ ] On future resets, restore from a backed-up post-start state if nes-py supports it reliably.

Candidate approach:

```text
reset emulator
advance a few idle frames
press START for N frames
release START
advance M frames
check RAM/gameplay state
if not playable, repeat bounded startup sequence
save backup state at first playable frame
return first observation
```

Acceptance criteria:

- [ ] 20 consecutive resets land in gameplay.
- [ ] No reset lands on title screen.
- [ ] Reset duration is consistent.
- [ ] Reset does not depend on real elapsed time.

Debug command to build:

```powershell
contra-rl smoke --rom .\roms\Contra.nes --render --resets 20
```

## Phase 5: RAM mapping

Goal: identify reliable Contra-specific RAM addresses.

Start with the old code addresses as hypotheses, not gospel.

- [ ] Validate player X position.
- [ ] Validate player Y position.
- [ ] Validate lives/death state.
- [ ] Validate score.
- [ ] Validate weapon state if possible.
- [ ] Add a temporary RAM debug overlay/log.
- [ ] Run controlled manual actions and observe address changes.

Minimum required for first training:

- [ ] horizontal progress
- [ ] death/life lost
- [ ] score increase if reliable

Nice later:

- [ ] boss state
- [ ] enemy positions
- [ ] weapon/powerup state
- [ ] level/stage marker

## Phase 6: Observation preprocessing

Goal: produce small, stable observations for CNN training.

- [ ] Raw emulator frame shape confirmed.
- [ ] Convert RGB to grayscale.
- [ ] Resize to 84x84.
- [ ] Stack 4 frames.
- [ ] Keep dtype efficient.
- [ ] Confirm final observation space matches Stable-Baselines3 expectations.

Baseline observation:

```text
4 x 84 x 84 grayscale frames
```

Open question for implementation:

- SB3 CNN policies usually expect channel-first tensors internally after vectorization, but Gym environments may emit channel-last. We should follow SB3’s recommended image-env conventions and verify with a smoke model before long training.

## Phase 7: Reward design v1

Goal: give the agent enough signal to move right, survive, and avoid standing still.

First reward components:

- [ ] progress reward: positive reward for increasing max X position.
- [ ] score reward: positive reward for score increase.
- [ ] death penalty: large negative reward on death/life loss.
- [ ] small time penalty: discourage doing nothing.
- [ ] stuck penalty/truncation: end episode if no progress for too long.

Every reward component must be logged separately:

```python
info["reward_parts"] = {
    "progress": ...,
    "score": ...,
    "death": ...,
    "time": ...,
    "stuck": ...,
}
```

Acceptance criteria:

- [ ] Standing still produces slightly negative total reward.
- [ ] Moving right produces positive reward.
- [ ] Death produces clear negative reward.
- [ ] Reward does not explode after reset.
- [ ] Reward parts are visible in smoke logs.

## Phase 8: Termination and truncation

Goal: episodes should end for real game failure or training usefulness.

- [ ] `terminated=True` on death/game over.
- [ ] `truncated=True` on max episode steps.
- [ ] `truncated=True` when stuck too long.
- [ ] Reset internal counters correctly:

  - [ ] max X position
  - [ ] previous score
  - [ ] previous lives/death state
  - [ ] stuck timer

Recommended defaults:

```yaml
max_episode_steps: 18000
stuck_timeout_steps: 900
```

At 60 FPS, this is about 5 minutes max episode length and 15 seconds of no-progress tolerance before truncating. With frame skip, wall-clock training time is different, but the gameplay logic remains understandable.

## Phase 9: Smoke CLI

Goal: one command proves the environment works.

Implement:

```powershell
contra-rl smoke --rom .\roms\Contra.nes
```

Options to add:

```powershell
--episodes 3
--steps 1000
--render
--record
--action-set RIGHT_ONLY
--seed 42
```

Acceptance criteria:

- [ ] Runs without training.
- [ ] Prints episode length and reward.
- [ ] Prints final info.
- [ ] Can optionally render.
- [ ] Can optionally record video.
- [ ] Can run repeated resets.

## Phase 10: Tests

Add tests that do not require committing the ROM.

- [ ] Unit tests for action sets.
- [ ] Unit tests for reward math.
- [ ] Unit tests for ROM path validation.
- [ ] Optional integration tests enabled only when `CONTRA_ROM_PATH` is set.

Integration test pattern:

```text
if no ROM path is configured:
  skip test
else:
  create env
  reset
  step 100 random actions
  assert observations and info are valid
```

## Phase 11: First PPO training

Only start this after the smoke command is boringly reliable.

Run:

```powershell
contra-rl train --config .\configs\ppo_baseline.yaml --rom .\roms\Contra.nes
```

First target:

- [ ] 100k timesteps to prove training loop works.
- [ ] TensorBoard logs appear.
- [ ] Checkpoints save.
- [ ] Evaluation video records.

Then:

- [ ] 1M timesteps.
- [ ] Review videos.
- [ ] Tune reward.
- [ ] 5M timesteps.

## Phase 12: Evaluation

Implement:

```powershell
contra-rl eval --checkpoint .\runs\ppo_baseline\best.zip --rom .\roms\Contra.nes --record
```

Metrics:

- [ ] max X position
- [ ] episode reward
- [ ] survival time
- [ ] score
- [ ] deaths
- [ ] video output

## Implementation order summary

Recommended order:

1. Environment loads ROM.
2. Reset skips title/loading screen reliably.
3. Random action stepping works.
4. RAM addresses are validated.
5. Reward parts are correct.
6. Observation preprocessing is stable.
7. Smoke CLI works.
8. Tests pass.
9. PPO training starts.
10. Evaluation videos guide reward iteration.

Do not start with fancy models. Start with boring environment certainty.

