from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from contra_rl.cli import _manual_buttons_from_keyboard
from contra_rl.envs.actions import ACTION_SETS, CONTRA_FULL
from contra_rl.envs.contra_env import ContraEnvError, validate_rom_path
from contra_rl.envs.rewards import RewardParts
from contra_rl.envs.stable_retro_env import StableRetroContraEnv
from contra_rl.training.policies import ContraCNN
from contra_rl.training.train import _resolve_policy_kwargs


def test_action_sets_present():
    assert "SIMPLE_MOVEMENT" in ACTION_SETS
    assert len(ACTION_SETS["SIMPLE_MOVEMENT"]) > 0


def test_reward_parts_total():
    parts = RewardParts(progress=1.0, death=-2.0, time=-0.5)
    assert parts.total == -1.5


def test_validate_rom_path_missing():
    missing_rom = Path("roms/definitely-missing-test-rom.nes")
    try:
        validate_rom_path(missing_rom)
    except ContraEnvError as exc:
        assert "ROM not found" in str(exc)
    else:
        raise AssertionError("expected missing ROM to raise ContraEnvError")


def test_manual_keyboard_supports_down_jump_combo():
    class FakePygame:
        K_w = 0
        K_a = 1
        K_s = 2
        K_d = 3
        K_j = 4
        K_k = 5
        K_RETURN = 6
        K_SPACE = 7

    keys = [False] * 8
    keys[FakePygame.K_s] = True
    keys[FakePygame.K_k] = True

    assert _manual_buttons_from_keyboard(keys, FakePygame) == frozenset({"down", "A"})


def test_full_action_set_covers_required_combinations():
    assert len(CONTRA_FULL) == 36
    assert ["right", "A", "B"] in CONTRA_FULL
    assert ["down", "A"] in CONTRA_FULL
    assert ["right", "down", "A", "B"] in CONTRA_FULL


def test_contra_cnn_extracts_stacked_frame_features():
    observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype="uint8")
    extractor = ContraCNN(observation_space, features_dim=64)

    features = extractor(torch.zeros((2, 4, 84, 84)))

    assert features.shape == (2, 64)


def test_policy_kwargs_resolve_custom_extractor():
    resolved = _resolve_policy_kwargs({"features_extractor_class": "ContraCNN"})

    assert resolved["features_extractor_class"] is ContraCNN


class _LifeSequenceEnv(gym.Env):
    """Small deterministic emulator stand-in for life-handling tests."""

    observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(1)

    def __init__(self, steps):
        self.steps = iter(steps)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.uint8), {"lives": 2, "x_pos": 100, "score": 0}

    def step(self, action):
        info, terminated = next(self.steps)
        return np.zeros(1, dtype=np.uint8), 0.0, terminated, False, info


def test_multilife_episode_continues_until_zero_lives():
    raw_env = _LifeSequenceEnv(
        [
            ({"lives": 1, "x_pos": 10, "score": 0, "death_flag": 1}, False),
            ({"lives": 1, "x_pos": 20, "score": 0, "death_flag": 1}, False),
            ({"lives": 0, "x_pos": 0, "score": 0, "death_flag": 1}, True),
        ]
    )
    env = StableRetroContraEnv(
        raw_env,
        terminate_on_life_loss=False,
        life_loss_penalty=-5.0,
        game_over_penalty=-95.0,
        stuck_timeout_steps=100,
    )
    env.reset()

    _, reward, terminated, _, info = env.step(0)
    assert not terminated
    assert info["life_lost"]
    assert not info["game_over"]
    assert info["max_x_pos"] == 100
    assert reward == -5.001

    _, reward, terminated, _, info = env.step(0)
    assert not terminated
    assert not info["life_lost"]
    assert reward == -0.001

    _, reward, terminated, _, info = env.step(0)
    assert terminated
    assert info["game_over"]
    assert info["reward_parts"]["life_lost"] == -5.0
    assert info["reward_parts"]["death"] == -95.0
    assert reward == -100.001


def test_terminal_life_loss_remains_default_behavior():
    env = StableRetroContraEnv(
        _LifeSequenceEnv([({"lives": 1, "x_pos": 10, "score": 0, "death_flag": 1}, False)]),
        stuck_timeout_steps=100,
    )
    env.reset()

    _, reward, terminated, _, info = env.step(0)

    assert terminated
    assert info["life_lost"]
    assert reward == -100.001
