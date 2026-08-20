"""Stable-Baselines3 evaluation helpers."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from stable_baselines3 import DQN, PPO

from contra_rl.envs.wrappers import current_rgb_frame, make_training_env

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

ALGORITHMS = {
    "PPO": PPO,
    "DQN": DQN,
}
if RecurrentPPO is not None:
    ALGORITHMS["RECURRENTPPO"] = RecurrentPPO


def load_model(checkpoint: Path, algorithm: str, *, device: str = "auto"):
    """Load an SB3 model checkpoint."""
    algorithm = algorithm.upper()
    try:
        model_cls = ALGORITHMS[algorithm]
    except KeyError as exc:
        choices = ", ".join(sorted(ALGORITHMS))
        if algorithm == "RECURRENTPPO" and RecurrentPPO is None:
            raise ValueError(
                "RecurrentPPO requires sb3-contrib. Install with "
                '`pip install -e ".[recurrent]"` or `pip install sb3-contrib`.'
            ) from exc
        raise ValueError(f"unsupported algorithm '{algorithm}'. Choose one of: {choices}") from exc
    return model_cls.load(checkpoint, device=device)


def is_recurrent_algorithm(config: dict[str, Any]) -> bool:
    """Return whether the configured algorithm uses recurrent policy state."""
    return str(config.get("algorithm", "")).upper() == "RECURRENTPPO"


def _open_video_writer(path: Path, first_frame: np.ndarray, fps: float):
    """Open an OpenCV MP4 video writer for RGB emulator frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"could not open video writer: {path}")
    return writer


def _write_rgb_frame(writer, frame: np.ndarray) -> None:
    """Write an RGB frame to an OpenCV BGR video writer."""
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def evaluate_model(
    rom_path: Path,
    checkpoint: Path,
    config: dict[str, Any],
    *,
    episodes: int = 1,
    max_steps: int = 5_000,
    deterministic: bool = True,
    record_path: Path | None = None,
    video_fps: float = 15.0,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint in the Contra environment."""
    algorithm = str(config.get("algorithm", "PPO"))
    env_config = config.get("env", {})
    model_device = device or str(config.get("device", "auto"))

    model = load_model(checkpoint, algorithm, device=model_device)
    is_recurrent = is_recurrent_algorithm(config)
    integration_path = env_config.get("stable_retro_integration_path")
    env = make_training_env(
        rom_path,
        backend=env_config.get("backend", "nes-py"),
        action_set=env_config.get("action_set", "SIMPLE_MOVEMENT"),
        frame_skip=int(env_config.get("frame_skip", 4)),
        screen_size=int(env_config.get("screen_size", 84)),
        grayscale=bool(env_config.get("grayscale", True)),
        frame_stack=int(env_config.get("frame_stack", 4)),
        max_episode_steps=int(env_config.get("max_episode_steps", 18_000)),
        stuck_timeout_steps=int(env_config.get("stuck_timeout_steps", 900)),
        progress_reward_scale=float(env_config.get("progress_reward_scale", 0.1)),
        progress_reward_mode=env_config.get("progress_reward_mode", "raw"),
        progress_bucket_size=int(env_config.get("progress_bucket_size", 32)),
        progress_reward_per_bucket=float(env_config.get("progress_reward_per_bucket", 0.5)),
        progress_reward_start_x=int(env_config.get("progress_reward_start_x", 0)),
        score_reward_scale=float(env_config.get("score_reward_scale", 0.001)),
        terminate_on_life_loss=bool(env_config.get("terminate_on_life_loss", True)),
        life_loss_penalty=float(env_config.get("life_loss_penalty", -100.0)),
        game_over_penalty=float(env_config.get("game_over_penalty", 0.0)),
        terminal_efficiency_penalty_scale=float(
            env_config.get("terminal_efficiency_penalty_scale", 0.0)
        ),
        terminal_efficiency_min_x=int(env_config.get("terminal_efficiency_min_x", 256)),
        terminal_efficiency_max_penalty=float(
            env_config.get("terminal_efficiency_max_penalty", 25.0)
        ),
        idle_penalty_per_step=float(env_config.get("idle_penalty_per_step", 0.0)),
        idle_penalty_start_steps=int(env_config.get("idle_penalty_start_steps", 0)),
        forward_recovery_per_pixel=float(env_config.get("forward_recovery_per_pixel", 0.0)),
        forward_recovery_debt_cap=float(env_config.get("forward_recovery_debt_cap", 5.0)),
        stable_retro_game=env_config.get("stable_retro_game", "Contra-Nes"),
        stable_retro_state=env_config.get("stable_retro_state", "Level1"),
        stable_retro_scenario=env_config.get("stable_retro_scenario"),
        stable_retro_info=env_config.get("stable_retro_info"),
        stable_retro_integration_path=Path(integration_path) if integration_path else None,
    )

    video_writer = None
    video_frames = 0
    episode_results = []
    try:
        for episode in range(1, episodes + 1):
            obs, info = env.reset(seed=10_000 + episode)
            total_reward = 0.0
            max_x = float(info.get("max_x_pos", info.get("x_pos", 0)))
            final_score = info.get("score", 0)
            terminated = False
            truncated = False
            step_count = 0
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)

            if record_path is not None and video_writer is None:
                first_frame = current_rgb_frame(env)
                video_writer = _open_video_writer(record_path, first_frame, video_fps)
                _write_rgb_frame(video_writer, first_frame)
                video_frames += 1

            for step_index in range(1, max_steps + 1):
                step_count = step_index
                if is_recurrent:
                    action, recurrent_state = model.predict(
                        obs,
                        state=recurrent_state,
                        episode_start=episode_start,
                        deterministic=deterministic,
                    )
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                episode_start = np.array([terminated or truncated], dtype=bool)
                total_reward += float(reward)
                max_x = max(max_x, float(info.get("max_x_pos", info.get("x_pos", 0))))
                final_score = info.get("score", final_score)

                if video_writer is not None:
                    _write_rgb_frame(video_writer, current_rgb_frame(env))
                    video_frames += 1

                if terminated or truncated:
                    break

            episode_results.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "steps": step_count,
                    "max_x_pos": max_x,
                    "score": final_score,
                    "terminated": terminated,
                    "truncated": truncated,
                    "final_info": info,
                }
            )

        rewards = [result["reward"] for result in episode_results]
        max_x_positions = [result["max_x_pos"] for result in episode_results]
        return {
            "episodes": episode_results,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_max_x_pos": float(np.mean(max_x_positions)) if max_x_positions else 0.0,
            "best_max_x_pos": float(np.max(max_x_positions)) if max_x_positions else 0.0,
            "record_path": record_path,
            "video_frames": video_frames,
            "video_fps": video_fps,
            "video_duration_seconds": video_frames / video_fps if video_fps > 0 else 0.0,
        }
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()
