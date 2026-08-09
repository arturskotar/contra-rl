"""Stable-Baselines3 training helpers."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from contra_rl.envs.wrappers import make_training_env

ALGORITHMS = {
    "PPO": PPO,
    "DQN": DQN,
}

PPO_KWARGS = {
    "n_steps",
    "batch_size",
    "n_epochs",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
}

DQN_KWARGS = {
    "learning_rate",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "gamma",
    "train_freq",
    "target_update_interval",
    "exploration_fraction",
    "exploration_final_eps",
}


def load_training_config(path: Path) -> dict[str, Any]:
    """Load a YAML training config."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"training config must be a mapping: {path}")
    return config


def apply_training_overrides(
    config: dict[str, Any],
    *,
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    run_name: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Return a config copy with CLI overrides applied."""
    updated = deepcopy(config)
    updated.setdefault("env", {})
    updated.setdefault("training", {})
    updated.setdefault("logging", {})

    if total_timesteps is not None:
        updated["training"]["total_timesteps"] = total_timesteps
    if n_envs is not None:
        updated["env"]["n_envs"] = n_envs
    if run_name is not None:
        updated["logging"]["run_name"] = run_name
    if device is not None:
        updated["device"] = device

    return updated


def make_vector_env(
    rom_path: Path,
    *,
    env_config: dict[str, Any],
    n_envs: int,
    seed: int,
):
    """Create a vectorized Contra training environment."""

    def make_one_env(rank: int):
        def _init():
            env = make_training_env(
                rom_path,
                action_set=env_config.get("action_set", "SIMPLE_MOVEMENT"),
                frame_skip=int(env_config.get("frame_skip", 4)),
                screen_size=int(env_config.get("screen_size", 84)),
                grayscale=bool(env_config.get("grayscale", True)),
                frame_stack=int(env_config.get("frame_stack", 4)),
                max_episode_steps=int(env_config.get("max_episode_steps", 18_000)),
                stuck_timeout_steps=int(env_config.get("stuck_timeout_steps", 900)),
            )
            env.reset(seed=seed + rank)
            return env

        return _init

    set_random_seed(seed)
    env = DummyVecEnv([make_one_env(rank) for rank in range(n_envs)])
    return VecMonitor(env)


def build_model(config: dict[str, Any], env, *, tensorboard_dir: Path):
    """Build an SB3 model from config."""
    algorithm = str(config.get("algorithm", "PPO")).upper()
    try:
        model_cls = ALGORITHMS[algorithm]
    except KeyError as exc:
        choices = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"unsupported algorithm '{algorithm}'. Choose one of: {choices}") from exc

    policy = config.get("policy", "CnnPolicy")
    device = config.get("device", "auto")
    seed = int(config.get("seed", 42))
    training_config = config.get("training", {})

    allowed_kwargs = PPO_KWARGS if algorithm == "PPO" else DQN_KWARGS
    model_kwargs = {key: training_config[key] for key in allowed_kwargs if key in training_config}

    return model_cls(
        policy,
        env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log=str(tensorboard_dir),
        **model_kwargs,
    )


def train_model(
    rom_path: Path,
    config: dict[str, Any],
    *,
    run_root: Path = Path("runs"),
):
    """Train an SB3 model and save the final checkpoint."""
    env_config = config.get("env", {})
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})

    seed = int(config.get("seed", 42))
    n_envs = int(env_config.get("n_envs", 1))
    total_timesteps = int(training_config.get("total_timesteps", 100_000))
    run_name = str(logging_config.get("run_name", "contra_run"))

    run_dir = run_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = make_vector_env(
        rom_path,
        env_config=env_config,
        n_envs=n_envs,
        seed=seed,
    )
    try:
        model = build_model(config, env, tensorboard_dir=tensorboard_dir)
        checkpoint_freq = int(logging_config.get("checkpoint_freq", 100_000))
        callbacks = []
        if checkpoint_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(1, checkpoint_freq // max(1, n_envs)),
                    save_path=str(checkpoint_dir),
                    name_prefix=run_name,
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks or None,
            tb_log_name=run_name,
            progress_bar=True,
        )

        final_model_path = run_dir / "final_model.zip"
        model.save(final_model_path)
        return {
            "run_dir": run_dir,
            "final_model_path": final_model_path,
            "tensorboard_dir": tensorboard_dir,
            "checkpoint_dir": checkpoint_dir,
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
        }
    finally:
        env.close()
