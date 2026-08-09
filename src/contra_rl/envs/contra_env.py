"""Gymnasium-compatible Contra environment.

This module intentionally starts small: load the ROM, skip the title screen,
step actions, expose candidate RAM diagnostics, and provide conservative reward
shaping for smoke tests. RAM addresses are still hypotheses until validated.
"""

from pathlib import Path

import numpy as np
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace

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


def validate_rom_path(rom_path: Path) -> Path:
    """Validate and normalize a ROM path."""
    resolved = rom_path.expanduser().resolve()
    if not resolved.exists():
        raise ContraEnvError(f"ROM not found: {resolved}")
    if not resolved.is_file():
        raise ContraEnvError(f"ROM path is not a file: {resolved}")
    return resolved


class ContraNesEnv(NESEnv):
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

        self._episode_steps = 0
        self._max_x_position = 0
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
        if progress_delta > 0:
            self._max_x_position = x_position
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        score = self._score
        score_delta = max(0, score - self._previous_score)
        self._previous_score = max(self._previous_score, score)

        death_penalty = -50.0 if self._is_dead or self._lives < self._previous_lives else 0.0
        self._previous_lives = self._lives

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
            "lives": self._lives,
            "score": self._score,
            "is_dead": self._is_dead,
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

    def observation(self, mode="rgb_array", output=None):
        obs = super().observation(mode=mode, output=output)
        return np.asarray(obs)


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
):
    """Create a Joypad-wrapped Contra environment."""
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
    )
    return JoypadSpace(env, actions)
