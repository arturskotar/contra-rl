"""Environment wrappers for preprocessing, frame stacking, and monitoring."""

from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from contra_rl.envs.contra_env import make_contra_env


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


def make_training_env(
    rom_path: Path,
    *,
    action_set: str = "SIMPLE_MOVEMENT",
    frame_skip: int = 4,
    screen_size: int = 84,
    grayscale: bool = True,
    frame_stack: int = 4,
    max_episode_steps: int = 18_000,
    stuck_timeout_steps: int = 900,
):
    """Create the single-env training wrapper stack."""
    env = make_contra_env(
        rom_path,
        action_set=action_set,
        render_mode=None,
        max_episode_steps=max_episode_steps,
        stuck_timeout_steps=stuck_timeout_steps,
    )
    env = FrameSkip(env, skip=frame_skip)
    env = ResizeAndGrayscale(env, size=screen_size, grayscale=grayscale)
    env = ChannelFirstFrameStack(env, stack_size=frame_stack)
    return env
