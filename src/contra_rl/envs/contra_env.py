"""Gymnasium-compatible Contra environment.

This module intentionally starts small: load the ROM, skip the title screen,
step actions, expose candidate RAM diagnostics, and provide conservative reward
shaping for smoke tests. RAM addresses are still hypotheses until validated.
"""

from pathlib import Path

import numpy as np

from contra_rl.envs.actions import ACTION_SETS
from contra_rl.envs.ram import (
    LIVES,
    PLAYER_X_LOW,
    PLAYER_Y,
    SCREEN_NUMBER,
    SCROLL_OFFSET,
)
from contra_rl.envs.rewards import RewardParts

START_BUTTON = 0b00001000
NOOP = 0b00000000
BUTTON_A = 0b00000001
BUTTON_B = 0b00000010
RIGHT_A_B = 0b10000011


class ContraEnvError(RuntimeError):
    """Raised when the Contra environment cannot be created or run."""


try:
    from nes_py import NESEnv as _NESEnv
    from nes_py.wrappers import JoypadSpace as _JoypadSpace
except ImportError as _NES_PY_IMPORT_ERROR:
    _NESEnv = object
    _JoypadSpace = None
else:
    _NES_PY_IMPORT_ERROR = None


def _import_nes_py():
    if _NES_PY_IMPORT_ERROR is not None or _JoypadSpace is None:
        raise ContraEnvError(
            "nes-py is not installed. Install the nes-py backend with "
            '`pip install -e ".[nes-py]"`, or use a stable-retro config.'
        ) from _NES_PY_IMPORT_ERROR
    return _NESEnv, _JoypadSpace


def validate_rom_path(rom_path: Path) -> Path:
    """Validate and normalize a ROM path."""
    resolved = rom_path.expanduser().resolve()
    if not resolved.exists():
        raise ContraEnvError(f"ROM not found: {resolved}")
    if not resolved.is_file():
        raise ContraEnvError(f"ROM path is not a file: {resolved}")
    return resolved


class ContraNesEnv(_NESEnv):
    """Raw NES Contra environment with deterministic startup handling."""

    reward_range = (-100.0, 100.0)

    def __init__(
        self,
        rom_path: Path,
        *,
        render_mode: str | None = None,
        startup_idle_frames: int = 240,
        startup_pre_action: int = BUTTON_B,
        startup_pre_presses: int = 4,
        startup_pre_press_frames: int = 1,
        startup_pre_release_frames: int = 1,
        start_press_frames: int = 1,
        startup_release_frames: int = 1,
        startup_attempt_frames: int = 480,
        post_start_frames: int = 60,
        startup_action: int = START_BUTTON,
        max_episode_steps: int = 18_000,
        stuck_timeout_steps: int = 900,
        progress_reward_scale: float = 0.1,
        progress_reward_mode: str = "raw",
        progress_bucket_size: int = 32,
        progress_reward_per_bucket: float = 0.5,
        progress_reward_start_x: int = 0,
        score_reward_scale: float = 0.001,
        create_start_backup: bool = True,
    ) -> None:
        self.rom_path = validate_rom_path(rom_path)
        self.startup_idle_frames = startup_idle_frames
        self.startup_pre_action = startup_pre_action
        self.startup_pre_presses = startup_pre_presses
        self.startup_pre_press_frames = startup_pre_press_frames
        self.startup_pre_release_frames = startup_pre_release_frames
        self.start_press_frames = start_press_frames
        self.startup_release_frames = startup_release_frames
        self.startup_attempt_frames = startup_attempt_frames
        self.post_start_frames = post_start_frames
        self.startup_action = startup_action
        self.max_episode_steps = max_episode_steps
        self.stuck_timeout_steps = stuck_timeout_steps
        self.progress_reward_scale = progress_reward_scale
        self.progress_reward_mode = progress_reward_mode
        self.progress_bucket_size = progress_bucket_size
        self.progress_reward_per_bucket = progress_reward_per_bucket
        self.progress_reward_start_x = progress_reward_start_x
        self.score_reward_scale = score_reward_scale

        self._episode_steps = 0
        self._max_x_position = 0
        self._max_progress_bucket = 0
        self._previous_score = 0
        self._previous_lives = 0
        self._steps_since_progress = 0
        self._last_reward_parts = RewardParts()

        super().__init__(str(self.rom_path), render_mode=render_mode)

        if create_start_backup:
            self._create_playable_start_backup()
        else:
            self._env.reset()
        self._reset_episode_state()
        self.done = False

    def _create_playable_start_backup(self) -> None:
        """Advance past the title screen once and save that emulator state."""
        self._env.reset()
        for _ in range(self.startup_idle_frames):
            self._frame_advance(NOOP)

        for _ in range(self.startup_pre_presses):
            for _ in range(self.startup_pre_press_frames):
                self._frame_advance(self.startup_pre_action)
            for _ in range(self.startup_pre_release_frames):
                self._frame_advance(NOOP)

        attempted_frames = 0
        while attempted_frames < self.startup_attempt_frames:
            for _ in range(self.start_press_frames):
                if attempted_frames >= self.startup_attempt_frames:
                    break
                self._frame_advance(self.startup_action)
                attempted_frames += 1
            for _ in range(self.startup_release_frames):
                if attempted_frames >= self.startup_attempt_frames:
                    break
                self._frame_advance(NOOP)
                attempted_frames += 1

        for _ in range(self.post_start_frames):
            self._frame_advance(NOOP)
        self._backup()

    def _reset_episode_state(self) -> None:
        self._episode_steps = 0
        self._max_x_position = self._x_position
        self._max_progress_bucket = self._progress_bucket(self._max_x_position)
        self._previous_score = self._score
        self._previous_lives = self._lives
        self._steps_since_progress = 0
        self._last_reward_parts = RewardParts()

    def _will_reset(self) -> None:
        self._reset_episode_state()

    def _did_reset(self) -> None:
        self._reset_episode_state()

    @property
    def _x_position(self) -> int:
        """Return a candidate absolute horizontal position from RAM."""
        return (
            int(self.ram[PLAYER_X_LOW])
            + int(self.ram[SCREEN_NUMBER]) * 255
            + int(self.ram[SCROLL_OFFSET])
        )

    @property
    def _y_position(self) -> int:
        """Return a candidate player Y position from RAM."""
        return int(self.ram[PLAYER_Y])

    @property
    def _lives(self) -> int:
        """Return a candidate lives value from RAM."""
        return int(self.ram[LIVES])

    @property
    def _score(self) -> int:
        """Return a candidate score value from RAM.

        This mirrors the old reference code as a hypothesis. It must be validated
        with controlled gameplay before we trust it for real training.
        """
        digits = [str(int(value)) for value in self.ram[0x07E2:0x07E4]]
        return int("".join(digits)) if digits else 0

    @property
    def _is_dead(self) -> bool:
        """Return a candidate death flag from RAM."""
        return bool(self.ram[0x00B4] != 0)

    def _get_reward(self) -> float:
        self._episode_steps += 1

        x_position = self._x_position
        progress_delta = max(0, x_position - self._max_x_position)
        score = self._score
        score_delta = max(0, score - self._previous_score)

        if progress_delta > 0:
            self._max_x_position = x_position

        if progress_delta > 0 or score_delta > 0:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        self._previous_score = max(self._previous_score, score)

        death_penalty = -50.0 if self._is_dead or self._lives < self._previous_lives else 0.0
        self._previous_lives = self._lives

        stuck_penalty = -1.0 if self._steps_since_progress >= self.stuck_timeout_steps else 0.0
        parts = RewardParts(
            progress=self._progress_reward(progress_delta, x_position),
            score=score_delta * self.score_reward_scale,
            death=death_penalty,
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

    def _get_terminated(self) -> bool:
        return self._is_dead or self._lives < self._previous_lives

    def _get_truncated(self) -> bool:
        return (
            self._episode_steps >= self.max_episode_steps
            or self._steps_since_progress >= self.stuck_timeout_steps
        )

    def _get_info(self) -> dict:
        return {
            "x_pos": self._x_position,
            "y_pos": self._y_position,
            "max_x_pos": self._max_x_position,
            "max_progress_bucket": self._max_progress_bucket,
            "lives": self._lives,
            "score": self._score,
            "is_dead": self._is_dead,
            "life_lost": self._lives < self._previous_lives,
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

    def observation(self, mode="rgb_array", output=None):
        obs = super().observation(mode=mode, output=output)
        return np.asarray(obs)

    def get_rgb_frame(self) -> np.ndarray:
        """Return the current emulator frame as an RGB array."""
        return self.observation("rgb_array")


def make_contra_env(
    rom_path: Path,
    *,
    action_set: str = "SIMPLE_MOVEMENT",
    render_mode: str | None = None,
    startup_idle_frames: int = 240,
    startup_pre_action: int = BUTTON_B,
    startup_pre_presses: int = 4,
    startup_pre_press_frames: int = 1,
    startup_pre_release_frames: int = 1,
    start_press_frames: int = 1,
    startup_release_frames: int = 1,
    startup_attempt_frames: int = 480,
    post_start_frames: int = 60,
    startup_action: int = START_BUTTON,
    max_episode_steps: int = 18_000,
    stuck_timeout_steps: int = 900,
    progress_reward_scale: float = 0.1,
    progress_reward_mode: str = "raw",
    progress_bucket_size: int = 32,
    progress_reward_per_bucket: float = 0.5,
    progress_reward_start_x: int = 0,
    score_reward_scale: float = 0.001,
):
    """Create a Joypad-wrapped Contra environment."""
    _, joypad_space = _import_nes_py()
    try:
        actions = ACTION_SETS[action_set]
    except KeyError as exc:
        choices = ", ".join(sorted(ACTION_SETS))
        raise ContraEnvError(
            f"Unknown action set '{action_set}'. Choose one of: {choices}"
        ) from exc

    env = ContraNesEnv(
        rom_path,
        render_mode=render_mode,
        startup_idle_frames=startup_idle_frames,
        startup_pre_action=startup_pre_action,
        startup_pre_presses=startup_pre_presses,
        startup_pre_press_frames=startup_pre_press_frames,
        startup_pre_release_frames=startup_pre_release_frames,
        start_press_frames=start_press_frames,
        startup_release_frames=startup_release_frames,
        startup_attempt_frames=startup_attempt_frames,
        post_start_frames=post_start_frames,
        startup_action=startup_action,
        max_episode_steps=max_episode_steps,
        stuck_timeout_steps=stuck_timeout_steps,
        progress_reward_scale=progress_reward_scale,
        progress_reward_mode=progress_reward_mode,
        progress_bucket_size=progress_bucket_size,
        progress_reward_per_bucket=progress_reward_per_bucket,
        progress_reward_start_x=progress_reward_start_x,
        score_reward_scale=score_reward_scale,
    )
    return joypad_space(env, actions)
