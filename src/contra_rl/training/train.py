"""Stable-Baselines3 training helpers."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from contra_rl.envs.wrappers import make_training_env
from contra_rl.training.callbacks import ContraMetricsCallback
from contra_rl.training.policies import ContraCNN

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
    "target_kl",
}

RECURRENT_PPO_KWARGS = PPO_KWARGS

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
    metrics_log_freq: int | None = None,
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
    if metrics_log_freq is not None:
        updated["logging"]["metrics_log_freq"] = metrics_log_freq

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
                progress_reward_per_bucket=float(
                    env_config.get("progress_reward_per_bucket", 0.5)
                ),
                progress_reward_start_x=int(env_config.get("progress_reward_start_x", 0)),
                score_reward_scale=float(env_config.get("score_reward_scale", 0.001)),
                terminate_on_life_loss=bool(env_config.get("terminate_on_life_loss", True)),
                life_loss_penalty=float(env_config.get("life_loss_penalty", -100.0)),
                game_over_penalty=float(env_config.get("game_over_penalty", 0.0)),
                terminal_efficiency_penalty_scale=float(
                    env_config.get("terminal_efficiency_penalty_scale", 0.0)
                ),
                terminal_efficiency_min_x=int(
                    env_config.get("terminal_efficiency_min_x", 256)
                ),
                terminal_efficiency_max_penalty=float(
                    env_config.get("terminal_efficiency_max_penalty", 25.0)
                ),
                idle_penalty_per_step=float(env_config.get("idle_penalty_per_step", 0.0)),
                idle_penalty_start_steps=int(env_config.get("idle_penalty_start_steps", 0)),
                forward_recovery_per_pixel=float(
                    env_config.get("forward_recovery_per_pixel", 0.0)
                ),
                forward_recovery_debt_cap=float(
                    env_config.get("forward_recovery_debt_cap", 5.0)
                ),
                stable_retro_game=env_config.get("stable_retro_game", "Contra-Nes"),
                stable_retro_state=env_config.get("stable_retro_state", "Level1"),
                stable_retro_scenario=env_config.get("stable_retro_scenario"),
                stable_retro_info=env_config.get("stable_retro_info"),
                stable_retro_integration_path=Path(integration_path)
                if integration_path
                else None,
            )
            env.reset(seed=seed + rank)
            return env

        return _init

    set_random_seed(seed)
    env_fns = [make_one_env(rank) for rank in range(n_envs)]
    if env_config.get("backend", "nes-py") == "stable-retro" and n_envs > 1:
        start_method = env_config.get("vec_env_start_method", "forkserver")
        env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        env = DummyVecEnv(env_fns)
    return VecMonitor(env)


def build_model(config: dict[str, Any], env, *, tensorboard_dir: Path):
    """Build an SB3 model from config."""
    algorithm = str(config.get("algorithm", "PPO")).upper()
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

    policy = config.get("policy", "CnnPolicy")
    device = config.get("device", "auto")
    seed = int(config.get("seed", 42))
    training_config = config.get("training", {})

    if algorithm == "PPO":
        allowed_kwargs = PPO_KWARGS
    elif algorithm == "RECURRENTPPO":
        allowed_kwargs = RECURRENT_PPO_KWARGS
    else:
        allowed_kwargs = DQN_KWARGS
    model_kwargs = {key: training_config[key] for key in allowed_kwargs if key in training_config}
    policy_kwargs = _resolve_policy_kwargs(config.get("policy_kwargs", {}))

    return model_cls(
        policy,
        env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log=str(tensorboard_dir),
        policy_kwargs=policy_kwargs,
        **model_kwargs,
    )


def _resolve_policy_kwargs(raw_policy_kwargs: Any) -> dict[str, Any]:
    """Translate YAML-friendly policy component names into Python classes."""
    if raw_policy_kwargs is None:
        return {}
    if not isinstance(raw_policy_kwargs, dict):
        raise ValueError("policy_kwargs must be a mapping")

    policy_kwargs = deepcopy(raw_policy_kwargs)
    extractor = policy_kwargs.get("features_extractor_class")
    if extractor is None:
        return policy_kwargs
    if extractor == "ContraCNN":
        policy_kwargs["features_extractor_class"] = ContraCNN
        return policy_kwargs
    raise ValueError(f"unknown features_extractor_class '{extractor}'")


def _resolve_resume_checkpoint(checkpoint: Path) -> Path:
    """Resolve and validate a checkpoint selected for continued training."""
    resolved = checkpoint.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"resume checkpoint not found: {resolved}")
    return resolved


def _apply_resume_exploration_overrides(model, config: dict[str, Any]) -> None:
    """Apply exploration settings from the new config to a loaded checkpoint."""
    training_config = config.get("training", {})
    if "ent_coef" in training_config:
        model.ent_coef = float(training_config["ent_coef"])


def load_resume_model(
    checkpoint: Path,
    config: dict[str, Any],
    env,
    *,
    tensorboard_dir: Path,
):
    """Load a compatible SB3 checkpoint and attach the newly configured environment."""
    resolved_checkpoint = _resolve_resume_checkpoint(checkpoint)
    algorithm = str(config.get("algorithm", "PPO")).upper()
    try:
        model_cls = ALGORITHMS[algorithm]
    except KeyError as exc:
        choices = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"unsupported algorithm '{algorithm}'. Choose one of: {choices}") from exc

    try:
        model = model_cls.load(
            resolved_checkpoint,
            env=env,
            device=config.get("device", "auto"),
            tensorboard_log=str(tensorboard_dir),
        )
        _apply_resume_exploration_overrides(model, config)
        return model
    except (AssertionError, ValueError) as exc:
        raise ValueError(
            "resume checkpoint is incompatible with this environment. The algorithm, "
            "observation shape, and action count must match."
        ) from exc


def train_model(
    rom_path: Path,
    config: dict[str, Any],
    *,
    run_root: Path = Path("runs"),
    resume_checkpoint: Path | None = None,
):
    """Train an SB3 model and save the final checkpoint."""
    env_config = config.get("env", {})
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})

    seed = int(config.get("seed", 42))
    n_envs = int(env_config.get("n_envs", 1))
    total_timesteps = int(training_config.get("total_timesteps", 100_000))
    run_name = str(logging_config.get("run_name", "contra_run"))
    metrics_log_freq = int(logging_config.get("metrics_log_freq", 1_000))

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
        if resume_checkpoint is None:
            model = build_model(config, env, tensorboard_dir=tensorboard_dir)
        else:
            model = load_resume_model(
                resume_checkpoint,
                config,
                env,
                tensorboard_dir=tensorboard_dir,
            )
        checkpoint_freq = int(logging_config.get("checkpoint_freq", 100_000))
        callbacks = []
        if metrics_log_freq > 0:
            callbacks.append(ContraMetricsCallback(log_freq=metrics_log_freq))
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
            reset_num_timesteps=resume_checkpoint is None,
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
            "resume_checkpoint": _resolve_resume_checkpoint(resume_checkpoint)
            if resume_checkpoint is not None
            else None,
        }
    finally:
        env.close()
