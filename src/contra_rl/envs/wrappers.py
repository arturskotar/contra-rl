"""Environment wrappers for preprocessing, frame stacking, and monitoring."""

from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from contra_rl.envs.contra_env import make_contra_env
from contra_rl.envs.stable_retro_env import make_stable_retro_contra_env


class FrameSkip(gym.Wrapper):
    """Repeat each selected action for a fixed number of emulator steps."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        if skip < 1:
            raise ValueError("frame skip must be >= 1")
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        final_obs = None
        final_info = {}
        terminated = False
        truncated = False

        for _ in range(self.skip):
            final_obs, reward, terminated, truncated, final_info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return final_obs, total_reward, terminated, truncated, final_info


class ResizeAndGrayscale(gym.ObservationWrapper):
    """Resize RGB observations and optionally convert them to grayscale."""

    def __init__(
        self,
        env: gym.Env,
        *,
        size: int = 84,
        grayscale: bool = True,
    ) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale

        if grayscale:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(size, size),
                dtype=np.uint8,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(size, size, 3),
                dtype=np.uint8,
            )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.grayscale:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            observation,
            (self.size, self.size),
            interpolation=cv2.INTER_AREA,
        )
        return resized.astype(np.uint8, copy=False)


class ChannelFirstFrameStack(gym.Wrapper):
    """Stack recent frames and expose channel-first image observations."""

    def __init__(self, env: gym.Env, stack_size: int = 4) -> None:
        if stack_size < 1:
            raise ValueError("stack size must be >= 1")
        super().__init__(env)
        self.stack_size = stack_size
        self.frames: deque[np.ndarray] = deque(maxlen=stack_size)

        shape = env.observation_space.shape
        if len(shape) == 2:
            height, width = shape
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(stack_size, height, width),
                dtype=np.uint8,
            )
        elif len(shape) == 3:
            height, width, channels = shape
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(stack_size * channels, height, width),
                dtype=np.uint8,
            )
        else:
            raise ValueError(f"unsupported observation shape for frame stacking: {shape}")

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(observation)
        return self._stacked_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self._stacked_observation(), reward, terminated, truncated, info

    def _stacked_observation(self) -> np.ndarray:
        frames = list(self.frames)
        if frames[0].ndim == 2:
            return np.stack(frames, axis=0).astype(np.uint8, copy=False)

        channel_first_frames = [np.transpose(frame, (2, 0, 1)) for frame in frames]
        return np.concatenate(channel_first_frames, axis=0).astype(np.uint8, copy=False)


def action_meanings(env: gym.Env) -> list[str]:
    """Return readable action labels from a wrapped action environment."""
    current = env
    while True:
        if hasattr(current, "get_action_meanings"):
            return list(current.get_action_meanings())
        if not hasattr(current, "env"):
            break
        current = current.env

    if hasattr(env.unwrapped, "get_action_meanings"):
        return list(env.unwrapped.get_action_meanings())
    return [str(index) for index in range(env.action_space.n)]


class ActionInfoWrapper(gym.Wrapper):
    """Attach the selected discrete action index/name to each info dict."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_meanings = action_meanings(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        action_index = int(action)
        info = dict(info)
        info["action_index"] = action_index
        if 0 <= action_index < len(self.action_meanings):
            info["action_name"] = self.action_meanings[action_index]
        else:
            info["action_name"] = str(action_index)
        return observation, reward, terminated, truncated, info


def make_training_env(
    rom_path: Path,
    *,
    backend: str = "nes-py",
    action_set: str = "SIMPLE_MOVEMENT",
    frame_skip: int = 4,
    screen_size: int = 84,
    grayscale: bool = True,
    frame_stack: int = 4,
    max_episode_steps: int = 18_000,
    stuck_timeout_steps: int = 900,
    progress_reward_scale: float = 0.1,
    progress_reward_mode: str = "raw",
    progress_bucket_size: int = 32,
    progress_reward_per_bucket: float = 0.5,
    progress_reward_start_x: int = 0,
    score_reward_scale: float = 0.001,
    terminate_on_life_loss: bool = True,
    life_loss_penalty: float = -100.0,
    game_over_penalty: float = 0.0,
    terminal_efficiency_penalty_scale: float = 0.0,
    terminal_efficiency_min_x: int = 256,
    terminal_efficiency_max_penalty: float = 25.0,
    idle_penalty_per_step: float = 0.0,
    idle_penalty_start_steps: int = 0,
    forward_recovery_per_pixel: float = 0.0,
    forward_recovery_debt_cap: float = 5.0,
    stable_retro_game: str = "Contra-Nes",
    stable_retro_state: str = "Level1",
    stable_retro_scenario: str | None = None,
    stable_retro_info: str | None = None,
    stable_retro_integration_path: Path | None = None,
):
    """Create the single-env training wrapper stack."""
    if backend == "nes-py":
        env = make_contra_env(
            rom_path,
            action_set=action_set,
            render_mode=None,
            max_episode_steps=max_episode_steps,
            stuck_timeout_steps=stuck_timeout_steps,
            progress_reward_scale=progress_reward_scale,
            progress_reward_mode=progress_reward_mode,
            progress_bucket_size=progress_bucket_size,
            progress_reward_per_bucket=progress_reward_per_bucket,
            progress_reward_start_x=progress_reward_start_x,
            score_reward_scale=score_reward_scale,
            terminate_on_life_loss=terminate_on_life_loss,
            life_loss_penalty=life_loss_penalty,
            game_over_penalty=game_over_penalty,
            terminal_efficiency_penalty_scale=terminal_efficiency_penalty_scale,
            terminal_efficiency_min_x=terminal_efficiency_min_x,
            terminal_efficiency_max_penalty=terminal_efficiency_max_penalty,
            idle_penalty_per_step=idle_penalty_per_step,
            idle_penalty_start_steps=idle_penalty_start_steps,
            forward_recovery_per_pixel=forward_recovery_per_pixel,
            forward_recovery_debt_cap=forward_recovery_debt_cap,
        )
    elif backend == "stable-retro":
        env = make_stable_retro_contra_env(
            game=stable_retro_game,
            state=stable_retro_state,
            scenario=stable_retro_scenario,
            info=stable_retro_info,
            integration_path=stable_retro_integration_path,
            action_set=action_set,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps,
            stuck_timeout_steps=stuck_timeout_steps,
            progress_reward_scale=progress_reward_scale,
            progress_reward_mode=progress_reward_mode,
            progress_bucket_size=progress_bucket_size,
            progress_reward_per_bucket=progress_reward_per_bucket,
            progress_reward_start_x=progress_reward_start_x,
            score_reward_scale=score_reward_scale,
        )
    else:
        raise ValueError("backend must be one of: nes-py, stable-retro")
    env = FrameSkip(env, skip=frame_skip)
    env = ResizeAndGrayscale(env, size=screen_size, grayscale=grayscale)
    env = ChannelFirstFrameStack(env, stack_size=frame_stack)
    env = ActionInfoWrapper(env)
    return env


def current_rgb_frame(env: gym.Env) -> np.ndarray:
    """Return the current raw RGB frame from a wrapped Contra environment."""
    current = env
    while True:
        if hasattr(current, "get_rgb_frame"):
            return np.asarray(current.get_rgb_frame())
        if not hasattr(current, "env"):
            break
        current = current.env

    unwrapped = env.unwrapped
    if hasattr(unwrapped, "get_rgb_frame"):
        return np.asarray(unwrapped.get_rgb_frame())
    if hasattr(unwrapped, "observation"):
        return np.asarray(unwrapped.observation("rgb_array"))
    rendered = unwrapped.render()
    if rendered is None:
        raise RuntimeError("could not read an RGB frame from the environment")
    return np.asarray(rendered)
