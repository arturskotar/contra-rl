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
        progress_reward_scale: float = 0.1,
        progress_reward_mode: str = "raw",
        progress_bucket_size: int = 32,
        progress_reward_per_bucket: float = 0.5,
        progress_reward_start_x: int = 0,
        score_reward_scale: float = 0.001,
        terminate_on_life_loss: bool = True,
        life_loss_penalty: float = -100.0,
        game_over_penalty: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.stuck_timeout_steps = stuck_timeout_steps
        self.progress_reward_scale = progress_reward_scale
        self.progress_reward_mode = progress_reward_mode
        self.progress_bucket_size = progress_bucket_size
        self.progress_reward_per_bucket = progress_reward_per_bucket
        self.progress_reward_start_x = progress_reward_start_x
        self.score_reward_scale = score_reward_scale
        self.terminate_on_life_loss = terminate_on_life_loss
        self.life_loss_penalty = life_loss_penalty
        self.game_over_penalty = game_over_penalty
        self._episode_steps = 0
        self._max_x_position = 0
        self._max_progress_bucket = 0
        self._previous_score = 0
        self._previous_lives = 0
        self._has_lives_baseline = False
        self._last_life_lost = False
        self._steps_since_progress = 0
        self._last_reward_parts = RewardParts()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_episode_state(info)
        return obs, self._with_contra_info(info)

    def step(self, action):
        obs, _retro_reward, retro_terminated, truncated, info = self.env.step(action)
        reward = self._get_reward(info)
        game_over = self._is_game_over(info)
        terminated = bool(
            retro_terminated
            or game_over
            or (self.terminate_on_life_loss and self._last_life_lost)
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
        self._max_progress_bucket = self._progress_bucket(self._max_x_position)
        self._previous_score = self._score(info)
        self._previous_lives = self._lives(info)
        self._has_lives_baseline = self._previous_lives > 0
        self._last_life_lost = False
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

    def _is_game_over(self, info: dict[str, Any]) -> bool:
        return self._has_lives_baseline and self._lives(info) <= 0

    def _get_reward(self, info: dict[str, Any]) -> float:
        self._episode_steps += 1

        x_position = self._x_position(info)
        progress_delta = max(0, x_position - self._max_x_position)
        score = self._score(info)
        score_delta = max(0, score - self._previous_score)

        if progress_delta > 0:
            self._max_x_position = x_position

        if progress_delta > 0 or score_delta > 0:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        self._previous_score = max(self._previous_score, score)

        current_lives = self._lives(info)
        if not self._has_lives_baseline and current_lives > 0:
            self._previous_lives = current_lives
            self._has_lives_baseline = True

        self._last_life_lost = (
            self._has_lives_baseline
            and current_lives >= 0
            and current_lives < self._previous_lives
        )
        game_over = self._is_game_over(info)
        # Charge each death only once, when the lives counter changes. The dying
        # animation can span many emulator frames, so checking ``is_dead`` here
        # would otherwise apply the penalty repeatedly.
        life_lost_penalty = self.life_loss_penalty if self._last_life_lost else 0.0
        death_penalty = self.game_over_penalty if self._last_life_lost and game_over else 0.0
        if self._last_life_lost:
            # Allow time for the respawn animation without treating it as a
            # stuck episode. Maximum progress deliberately remains unchanged.
            self._steps_since_progress = 0
        self._previous_lives = current_lives

        stuck_penalty = -1.0 if self._steps_since_progress >= self.stuck_timeout_steps else 0.0
        parts = RewardParts(
            progress=self._progress_reward(progress_delta, x_position),
            score=score_delta * self.score_reward_scale,
            death=death_penalty,
            life_lost=life_lost_penalty,
            time=-0.001,
            stuck=stuck_penalty,
        )
        self._last_reward_parts = parts
        return parts.total

    def _progress_bucket(self, x_position: int) -> int:
        if self.progress_bucket_size <= 0:
            raise ValueError("progress_bucket_size must be > 0")
        rewardable_x = max(0, x_position - self.progress_reward_start_x)
        return rewardable_x // self.progress_bucket_size

    def _progress_reward(self, progress_delta: int, x_position: int) -> float:
        if self.progress_reward_mode == "raw":
            return progress_delta * self.progress_reward_scale
        if self.progress_reward_mode == "bucket":
            current_bucket = self._progress_bucket(x_position)
            bucket_delta = max(0, current_bucket - self._max_progress_bucket)
            self._max_progress_bucket = max(self._max_progress_bucket, current_bucket)
            return bucket_delta * self.progress_reward_per_bucket
        raise ValueError("progress_reward_mode must be one of: raw, bucket")

    def _with_contra_info(self, info: dict[str, Any]) -> dict[str, Any]:
        contra_info = dict(info)
        contra_info.update(
            {
                "x_pos": self._x_position(info),
                "y_pos": self._y_position(info),
                "max_x_pos": self._max_x_position,
                "max_progress_bucket": self._max_progress_bucket,
                "lives": self._lives(info),
                "score": self._score(info),
                "is_dead": self._is_dead(info),
                "game_over": self._is_game_over(info),
                "life_lost": self._last_life_lost,
                "episode_steps": self._episode_steps,
                "steps_since_progress": self._steps_since_progress,
                "steps_since_useful_event": self._steps_since_progress,
                "reward_parts": {
                    "progress": self._last_reward_parts.progress,
                    "score": self._last_reward_parts.score,
                    "weapon": self._last_reward_parts.weapon,
                    "death": self._last_reward_parts.death,
                    "life_lost": self._last_reward_parts.life_lost,
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
    progress_reward_scale: float = 0.1,
    progress_reward_mode: str = "raw",
    progress_bucket_size: int = 32,
    progress_reward_per_bucket: float = 0.5,
    progress_reward_start_x: int = 0,
    score_reward_scale: float = 0.001,
    terminate_on_life_loss: bool = True,
    life_loss_penalty: float = -100.0,
    game_over_penalty: float = 0.0,
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
        progress_reward_scale=progress_reward_scale,
        progress_reward_mode=progress_reward_mode,
        progress_bucket_size=progress_bucket_size,
        progress_reward_per_bucket=progress_reward_per_bucket,
        progress_reward_start_x=progress_reward_start_x,
        score_reward_scale=score_reward_scale,
        terminate_on_life_loss=terminate_on_life_loss,
        life_loss_penalty=life_loss_penalty,
        game_over_penalty=game_over_penalty,
    )
