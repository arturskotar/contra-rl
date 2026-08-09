"""Stable Retro backend for Contra.

Stable Retro uses named game integrations and disk-backed ``.state`` files.
That makes it a better long-term fit for Contra than replaying the title screen
or depending on nes-py native backup/restore.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from contra_rl.envs.actions import ACTION_SETS
from contra_rl.envs.rewards import RewardParts


class StableRetroUnavailableError(RuntimeError):
    """Raised when Stable Retro is requested but not installed/importable."""


def _import_stable_retro():
    try:
        import stable_retro as retro
    except ImportError as exc:
        raise StableRetroUnavailableError(
            "stable-retro is not installed. Install it in the active venv, then import the "
            "Contra ROM into a Stable Retro integration."
        ) from exc
    return retro


def _normal_button_name(button: str | None) -> str:
    if button is None:
        return ""
    return button.strip().lower()


class DiscreteStableRetroActions(gym.ActionWrapper):
    """Map our named discrete Contra actions to Stable Retro button arrays."""

    def __init__(self, env: gym.Env, action_set: str) -> None:
        try:
            actions = ACTION_SETS[action_set]
        except KeyError as exc:
            choices = ", ".join(sorted(ACTION_SETS))
            message = f"unknown action set '{action_set}'. Choose one of: {choices}"
            raise ValueError(message) from exc

        super().__init__(env)
        self.actions = actions
        self.action_meanings = [" ".join(action) for action in actions]
        self.buttons = [_normal_button_name(button) for button in getattr(env, "buttons", [])]
        if not self.buttons:
            # Standard libretro NES order used by Stable/Gym Retro cores.
            self.buttons = ["b", "a", "select", "start", "up", "down", "left", "right"]
        self.action_space = spaces.Discrete(len(actions))

    def action(self, action: int):
        pressed = {_normal_button_name(button) for button in self.actions[int(action)]}
        return np.array([button in pressed for button in self.buttons], dtype=np.int8)

    def get_action_meanings(self) -> list[str]:
        """Return readable action labels for debug output."""
        return self.action_meanings


class StableRetroContraEnv(gym.Wrapper):
    """Contra reward/info wrapper for Stable Retro environments."""

    reward_range = (-100.0, 100.0)

    def __init__(
        self,
        env: gym.Env,
        *,
        max_episode_steps: int = 18_000,
        stuck_timeout_steps: int = 900,
    ) -> None:
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.stuck_timeout_steps = stuck_timeout_steps
        self._episode_steps = 0
        self._max_x_position = 0
        self._previous_score = 0
        self._previous_lives = 0
        self._steps_since_progress = 0
        self._last_reward_parts = RewardParts()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_episode_state(info)
        return obs, self._with_contra_info(info)

    def step(self, action):
        obs, _retro_reward, terminated, truncated, info = self.env.step(action)
        reward = self._get_reward(info)
        terminated = bool(
            terminated or self._is_dead(info) or self._lives(info) < self._previous_lives
        )
        truncated = bool(
            truncated
            or self._episode_steps >= self.max_episode_steps
            or self._steps_since_progress >= self.stuck_timeout_steps
        )
        return obs, reward, terminated, truncated, self._with_contra_info(info)

    def get_rgb_frame(self) -> np.ndarray:
        """Return the current emulator frame as RGB."""
        if hasattr(self.env.unwrapped, "get_screen"):
            return np.asarray(self.env.unwrapped.get_screen())
        rendered = self.env.render()
        if rendered is None:
            raise RuntimeError("Stable Retro render() did not return an RGB frame")
        return np.asarray(rendered)

    def _reset_episode_state(self, info: dict[str, Any]) -> None:
        self._episode_steps = 0
        self._max_x_position = self._x_position(info)
        self._previous_score = self._score(info)
        self._previous_lives = self._lives(info)
        self._steps_since_progress = 0
        self._last_reward_parts = RewardParts()

    def _x_position(self, info: dict[str, Any]) -> int:
        if "xscroll" in info:
            return int(info["xscroll"])
        if all(key in info for key in ("player_x_low", "screen_number", "scroll_offset")):
            return (
                int(info["player_x_low"])
                + int(info["screen_number"]) * 255
                + int(info["scroll_offset"])
            )
        for key in ("x_pos", "max_x_pos", "x", "scroll_x", "screen_x"):
            if key in info:
                return int(info[key])
        return 0

    def _y_position(self, info: dict[str, Any]) -> int:
        for key in ("y_pos", "y"):
            if key in info:
                return int(info[key])
        return 0

    def _lives(self, info: dict[str, Any]) -> int:
        return int(info.get("lives", self._previous_lives))

    def _score(self, info: dict[str, Any]) -> int:
        if all(key in info for key in ("score_high", "score_low")):
            return int(f"{int(info['score_high'])}{int(info['score_low'])}")
        return int(info.get("score", self._previous_score))

    def _is_dead(self, info: dict[str, Any]) -> bool:
        stable_retro_dead = (
            "player_state" in info
            and "lives" in info
            and int(info["player_state"]) == 15
            and int(info["lives"]) <= 0
        )
        return bool(
            stable_retro_dead
            or
            info.get("is_dead", False)
            or info.get("dead", False)
            or int(info.get("death_flag", 0)) != 0
            or int(info.get("dying_state", 0)) == 12
        )

    def _get_reward(self, info: dict[str, Any]) -> float:
        self._episode_steps += 1

        x_position = self._x_position(info)
        progress_delta = max(0, x_position - self._max_x_position)
        if progress_delta > 0:
            self._max_x_position = x_position
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        score = self._score(info)
        score_delta = max(0, score - self._previous_score)
        self._previous_score = max(self._previous_score, score)

        death_penalty = (
            -50.0 if self._is_dead(info) or self._lives(info) < self._previous_lives else 0.0
        )
        self._previous_lives = self._lives(info)

        stuck_penalty = -1.0 if self._steps_since_progress >= self.stuck_timeout_steps else 0.0
        parts = RewardParts(
            progress=progress_delta / 10.0,
            score=score_delta / 1000.0,
            death=death_penalty,
            time=-0.001,
            stuck=stuck_penalty,
        )
        self._last_reward_parts = parts
        return parts.total

    def _with_contra_info(self, info: dict[str, Any]) -> dict[str, Any]:
        contra_info = dict(info)
        contra_info.update(
            {
                "x_pos": self._x_position(info),
                "y_pos": self._y_position(info),
                "max_x_pos": self._max_x_position,
                "lives": self._lives(info),
                "score": self._score(info),
                "is_dead": self._is_dead(info),
                "episode_steps": self._episode_steps,
                "steps_since_progress": self._steps_since_progress,
                "reward_parts": {
                    "progress": self._last_reward_parts.progress,
                    "score": self._last_reward_parts.score,
                    "weapon": self._last_reward_parts.weapon,
                    "death": self._last_reward_parts.death,
                    "time": self._last_reward_parts.time,
                    "stuck": self._last_reward_parts.stuck,
                },
            }
        )
        return contra_info


def make_stable_retro_contra_env(
    *,
    game: str = "Contra-Nes",
    state: str = "Level1",
    scenario: str | None = None,
    info: str | None = None,
    integration_path: Path | None = None,
    action_set: str = "SIMPLE_MOVEMENT",
    render_mode: str | None = "rgb_array",
    max_episode_steps: int = 18_000,
    stuck_timeout_steps: int = 900,
):
    """Create a Stable Retro Contra environment from an imported integration."""
    retro = _import_stable_retro()

    if integration_path is not None:
        retro.data.Integrations.add_custom_path(str(integration_path.expanduser().resolve()))
        inttype = retro.data.Integrations.ALL
    else:
        inttype = retro.data.Integrations.DEFAULT

    env = retro.make(
        game=game,
        state=state,
        scenario=scenario,
        info=info,
        inttype=inttype,
        use_restricted_actions=retro.Actions.ALL,
        render_mode=render_mode,
    )
    env = DiscreteStableRetroActions(env, action_set=action_set)
    return StableRetroContraEnv(
        env,
        max_episode_steps=max_episode_steps,
        stuck_timeout_steps=stuck_timeout_steps,
    )
