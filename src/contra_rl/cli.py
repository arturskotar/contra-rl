import hashlib
import shutil
from pathlib import Path
from time import sleep
from typing import Annotated

import cv2
import numpy as np
import typer
from rich.console import Console
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from contra_rl.envs.actions import ACTION_SETS
from contra_rl.envs.contra_env import (
    BUTTON_A,
    NOOP,
    RIGHT_A_B,
    START_BUTTON,
    ContraEnvError,
    ContraNesEnv,
    make_contra_env,
    validate_rom_path,
)
from contra_rl.envs.stable_retro_env import (
    StableRetroContraEnv,
    StableRetroUnavailableError,
    _import_stable_retro,
)
from contra_rl.envs.wrappers import current_rgb_frame, make_training_env
from contra_rl.training.evaluate import evaluate_model, is_recurrent_algorithm, load_model
from contra_rl.training.train import (
    apply_training_overrides,
    load_training_config,
    train_model,
)

app = typer.Typer(help="Train and evaluate Contra RL agents.")
console = Console()
WATCH_WINDOW_NAME = "Contra RL Watch"
MANUAL_WINDOW_NAME = "Contra RL Manual Play"
BUTTON_BITS = {
    "right": 0b10000000,
    "left": 0b01000000,
    "down": 0b00100000,
    "up": 0b00010000,
    "start": START_BUTTON,
    "select": 0b00000100,
    "B": 0b00000010,
    "A": 0b00000001,
}
STARTUP_ACTIONS = {
    "start": START_BUTTON,
    "a": BUTTON_A,
    "b": BUTTON_BITS["B"],
    "a+b": BUTTON_BITS["A"] | BUTTON_BITS["B"],
    "right+a+b": RIGHT_A_B,
}
PRE_ACTIONS = {"none": NOOP, **STARTUP_ACTIONS}


def _require_action_set(action_set: str) -> None:
    if action_set not in ACTION_SETS:
        choices = ", ".join(sorted(ACTION_SETS))
        raise ContraEnvError(f"Unknown action set '{action_set}'. Choose one of: {choices}")


def _action_index(action_meanings: list[str], preferred: list[str]) -> int:
    for meaning in preferred:
        if meaning in action_meanings:
            return action_meanings.index(meaning)
    return 0


def _startup_action_mask(name: str) -> int:
    try:
        return STARTUP_ACTIONS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(STARTUP_ACTIONS))
        raise ContraEnvError(f"Unknown startup action '{name}'. Choose one of: {choices}") from exc


def _pre_action_mask(name: str) -> int:
    try:
        return PRE_ACTIONS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PRE_ACTIONS))
        raise ContraEnvError(f"Unknown pre-action '{name}'. Choose one of: {choices}") from exc


def _choose_visual_action(
    policy: str,
    step_index: int,
    action_meanings: list[str],
    rng: np.random.Generator,
) -> int:
    """Choose a simple visual/debug action."""
    if policy == "random":
        return int(rng.integers(len(action_meanings)))
    if policy == "noop":
        return _action_index(action_meanings, ["NOOP"])
    if policy == "right":
        return _action_index(action_meanings, ["right", "right B", "right A"])
    if policy == "run-jump-shoot":
        # Mostly run and shoot, with a little hop every couple seconds.
        if step_index % 90 in range(0, 12):
            return _action_index(action_meanings, ["right A B", "right A", "right B", "right"])
        return _action_index(action_meanings, ["right B", "right A B", "right"])
    raise ContraEnvError(
        "Unknown visual policy. Choose one of: random, noop, right, run-jump-shoot"
    )


def _show_frame(frame: np.ndarray, scale: int) -> bool:
    """Show an RGB emulator frame with OpenCV. Return False when user requests exit."""
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if scale > 1:
        bgr_frame = cv2.resize(
            bgr_frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.imshow(WATCH_WINDOW_NAME, bgr_frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in {27, ord("q")}


def _import_pygame():
    """Import the optional manual-play frontend only when it is needed."""
    try:
        import pygame
    except ImportError as exc:
        raise ValueError(
            "manual play requires pygame. Install it with "
            '`pip install -e ".[manual]"` in the active environment.'
        ) from exc
    return pygame


def _manual_buttons_from_keyboard(keys, pygame) -> frozenset[str]:
    """Return every NES button currently held in the Pygame window."""
    bindings = {
        pygame.K_w: "up",
        pygame.K_a: "left",
        pygame.K_s: "down",
        pygame.K_d: "right",
        pygame.K_j: "B",
        pygame.K_k: "A",
        pygame.K_RETURN: "start",
        pygame.K_SPACE: "select",
    }
    return frozenset(button for key, button in bindings.items() if keys[key])


def _show_pygame_frame(screen, frame: np.ndarray, scale: int, pygame) -> None:
    """Render an RGB emulator frame into the manual-play Pygame window."""
    height, width = frame.shape[:2]
    source = pygame.image.frombuffer(frame.tobytes(), (width, height), "RGB")
    if scale > 1:
        source = pygame.transform.scale(source, (width * scale, height * scale))
    screen.blit(source, (0, 0))
    pygame.display.flip()


def _manual_stable_retro_action(env: StableRetroContraEnv, buttons: frozenset[str]) -> np.ndarray:
    """Translate named NES buttons to Stable Retro's controller array."""
    controller_buttons = [str(button).strip().lower() for button in env.unwrapped.buttons]
    pressed = {button.lower() for button in buttons}
    return np.array([button in pressed for button in controller_buttons], dtype=np.int8)


@app.command()
def smoke(
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
    episodes: Annotated[int, typer.Option(min=1, help="Number of smoke-test episodes.")] = 1,
    steps: Annotated[int, typer.Option(min=1, help="Maximum random steps per episode.")] = 1000,
    resets: Annotated[int, typer.Option(min=1, help="Number of reset checks before stepping.")] = 1,
    action_set: Annotated[str, typer.Option(help="Action set name.")] = "SIMPLE_MOVEMENT",
    render: Annotated[bool, typer.Option(help="Show the emulator window while stepping.")] = False,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    startup_idle_frames: Annotated[
        int,
        typer.Option(min=0, help="Frames to wait before startup pulses."),
    ] = 240,
    startup_pre_action: Annotated[
        str,
        typer.Option(help="Pre-action before startup pulses: none, start, a, b, a+b, right+a+b."),
    ] = "b",
    startup_pre_presses: Annotated[
        int,
        typer.Option(min=0, help="Number of pre-action presses."),
    ] = 4,
    startup_pre_press_frames: Annotated[
        int,
        typer.Option(min=1, help="Frames to hold each pre-action press."),
    ] = 1,
    startup_pre_release_frames: Annotated[
        int,
        typer.Option(min=0, help="Frames to release between pre-action presses."),
    ] = 1,
    start_press_frames: Annotated[
        int,
        typer.Option(min=1, help="Frames to hold each startup pulse."),
    ] = 1,
    startup_release_frames: Annotated[
        int,
        typer.Option(min=1, help="Frames to release between startup pulses."),
    ] = 1,
    startup_attempt_frames: Annotated[
        int,
        typer.Option(min=1, help="Total frame budget for startup pulses."),
    ] = 480,
    post_start_frames: Annotated[
        int,
        typer.Option(min=0, help="Frames to wait after releasing the startup action."),
    ] = 60,
    startup_action: Annotated[
        str,
        typer.Option(help="Startup action: start, a, b, a+b, or right+a+b."),
    ] = "start",
) -> None:
    """Run a quick environment smoke test."""
    try:
        resolved_rom = validate_rom_path(rom)
        _require_action_set(action_set)

        rng = np.random.default_rng(seed)
        render_mode = "human" if render else None
        console.print(f"[cyan]ROM:[/cyan] {resolved_rom}")
        console.print(
            f"[cyan]Action set:[/cyan] {action_set} "
            f"({len(ACTION_SETS[action_set])} actions)"
        )
        console.print("[cyan]Creating environment and skipping title screen...[/cyan]")

        env = make_contra_env(
            resolved_rom,
            action_set=action_set,
            render_mode=render_mode,
            startup_idle_frames=startup_idle_frames,
            startup_pre_action=_pre_action_mask(startup_pre_action),
            startup_pre_presses=startup_pre_presses,
            startup_pre_press_frames=startup_pre_press_frames,
            startup_pre_release_frames=startup_pre_release_frames,
            start_press_frames=start_press_frames,
            startup_release_frames=startup_release_frames,
            startup_attempt_frames=startup_attempt_frames,
            post_start_frames=post_start_frames,
            startup_action=_startup_action_mask(startup_action),
        )
        try:
            for reset_index in range(resets):
                obs, info = env.reset(seed=seed + reset_index)
                console.print(
                    f"reset {reset_index + 1}/{resets}: "
                    f"obs_shape={getattr(obs, 'shape', None)} "
                    f"x={info.get('x_pos')} y={info.get('y_pos')} "
                    f"lives={info.get('lives')} score={info.get('score')}"
                )

            for episode in range(episodes):
                obs, info = env.reset(seed=seed + resets + episode)
                total_reward = 0.0
                terminated = False
                truncated = False
                step_count = 0

                for step_index in range(1, steps + 1):
                    step_count = step_index
                    action = int(rng.integers(env.action_space.n))
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if render:
                        env.render()
                    if terminated or truncated:
                        break

                status = "terminated" if terminated else "truncated" if truncated else "max_steps"
                console.print(
                    f"episode {episode + 1}/{episodes}: "
                    f"status={status} steps={step_count} reward={total_reward:.3f} "
                    f"x={info.get('x_pos')} max_x={info.get('max_x_pos')} "
                    f"lives={info.get('lives')} score={info.get('score')}"
                )
                console.print(f"reward_parts={info.get('reward_parts')}")
        finally:
            env.close()
    except ContraEnvError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def watch(
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
    steps: Annotated[int, typer.Option(min=1, help="Number of rendered steps.")] = 1800,
    action_set: Annotated[str, typer.Option(help="Action set name.")] = "SIMPLE_MOVEMENT",
    policy: Annotated[
        str,
        typer.Option(help="Visual policy: random, noop, right, or run-jump-shoot."),
    ] = "run-jump-shoot",
    fps: Annotated[float, typer.Option(min=1.0, max=60.0, help="Playback speed.")] = 30.0,
    scale: Annotated[int, typer.Option(min=1, max=6, help="Window pixel scale.")] = 3,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    startup_idle_frames: Annotated[
        int,
        typer.Option(min=0, help="Frames to wait before startup pulses."),
    ] = 240,
    startup_pre_action: Annotated[
        str,
        typer.Option(help="Pre-action before startup pulses: none, start, a, b, a+b, right+a+b."),
    ] = "b",
    startup_pre_presses: Annotated[
        int,
        typer.Option(min=0, help="Number of pre-action presses."),
    ] = 4,
    startup_pre_press_frames: Annotated[
        int,
        typer.Option(min=1, help="Frames to hold each pre-action press."),
    ] = 1,
    startup_pre_release_frames: Annotated[
        int,
        typer.Option(min=0, help="Frames to release between pre-action presses."),
    ] = 1,
    start_press_frames: Annotated[
        int,
        typer.Option(min=1, help="Frames to hold each startup pulse."),
    ] = 1,
    startup_release_frames: Annotated[
        int,
        typer.Option(min=1, help="Frames to release between startup pulses."),
    ] = 1,
    startup_attempt_frames: Annotated[
        int,
        typer.Option(min=1, help="Total frame budget for startup pulses."),
    ] = 480,
    post_start_frames: Annotated[
        int,
        typer.Option(min=0, help="Frames to wait after releasing the startup action."),
    ] = 60,
    startup_action: Annotated[
        str,
        typer.Option(help="Startup action: start, a, b, a+b, or right+a+b."),
    ] = "start",
) -> None:
    """Watch the environment in a render window."""
    try:
        resolved_rom = validate_rom_path(rom)
        _require_action_set(action_set)

        rng = np.random.default_rng(seed)
        delay_seconds = 1.0 / fps
        console.print(f"[cyan]ROM:[/cyan] {resolved_rom}")
        console.print(f"[cyan]Action set:[/cyan] {action_set}")
        console.print(f"[cyan]Visual policy:[/cyan] {policy}")
        console.print("[cyan]Opening OpenCV render window...[/cyan]")
        console.print("[dim]Press q or Esc in the render window to stop.[/dim]")

        env = make_contra_env(
            resolved_rom,
            action_set=action_set,
            render_mode="rgb_array",
            startup_idle_frames=startup_idle_frames,
            startup_pre_action=_pre_action_mask(startup_pre_action),
            startup_pre_presses=startup_pre_presses,
            startup_pre_press_frames=startup_pre_press_frames,
            startup_pre_release_frames=startup_pre_release_frames,
            start_press_frames=start_press_frames,
            startup_release_frames=startup_release_frames,
            startup_attempt_frames=startup_attempt_frames,
            post_start_frames=post_start_frames,
            startup_action=_startup_action_mask(startup_action),
        )
        action_meanings = env.get_action_meanings()
        console.print(f"[cyan]Actions:[/cyan] {action_meanings}")

        try:
            frame, info = env.reset(seed=seed)
            keep_running = _show_frame(frame, scale)
            total_reward = 0.0

            for step_index in range(1, steps + 1):
                if not keep_running:
                    console.print("[yellow]Stopped by user.[/yellow]")
                    break

                action = _choose_visual_action(policy, step_index, action_meanings, rng)
                frame, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                keep_running = _show_frame(frame, scale)
                sleep(delay_seconds)

                if step_index % 120 == 0 or terminated or truncated:
                    console.print(
                        f"step={step_index} reward={total_reward:.3f} "
                        f"x={info.get('x_pos')} max_x={info.get('max_x_pos')} "
                        f"lives={info.get('lives')} score={info.get('score')}"
                    )

                if terminated or truncated:
                    status = "terminated" if terminated else "truncated"
                    console.print(f"[yellow]Episode {status}; resetting.[/yellow]")
                    frame, info = env.reset(seed=seed + step_index)
                    keep_running = _show_frame(frame, scale)

            console.print(
                f"[green]Finished watch run.[/green] reward={total_reward:.3f} "
                f"x={info.get('x_pos')} max_x={info.get('max_x_pos')}"
            )
        finally:
            env.close()
            cv2.destroyWindow(WATCH_WINDOW_NAME)
    except ContraEnvError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def play(
    checkpoint: Annotated[Path, typer.Option(help="Path to a trained checkpoint.")],
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
    config: Annotated[
        Path,
        typer.Option(help="Path to the training config used for the checkpoint."),
    ] = Path("configs/ppo_baseline.yaml"),
    steps: Annotated[int, typer.Option(min=1, help="Maximum rendered agent steps.")] = 5000,
    deterministic: Annotated[bool, typer.Option(help="Use deterministic policy actions.")] = True,
    fps: Annotated[float, typer.Option(min=1.0, max=60.0, help="Playback speed.")] = 60.0,
    scale: Annotated[int, typer.Option(min=1, max=6, help="Window pixel scale.")] = 3,
    device: Annotated[
        str | None,
        typer.Option(help="Override device: cuda, cpu, or auto."),
    ] = None,
) -> None:
    """Watch a trained agent play in a live OpenCV window."""
    try:
        resolved_rom = validate_rom_path(rom)
        resolved_checkpoint = checkpoint.expanduser().resolve()
        if not resolved_checkpoint.exists():
            raise ValueError(f"checkpoint not found: {resolved_checkpoint}")

        resolved_config = config.expanduser().resolve()
        play_config = load_training_config(resolved_config)
        algorithm = str(play_config.get("algorithm", "PPO"))
        is_recurrent = is_recurrent_algorithm(play_config)
        env_config = play_config.get("env", {})
        model_device = device or str(play_config.get("device", "auto"))

        console.print(f"[cyan]ROM:[/cyan] {resolved_rom}")
        console.print(f"[cyan]Checkpoint:[/cyan] {resolved_checkpoint}")
        console.print(f"[cyan]Config:[/cyan] {resolved_config}")
        console.print(f"[cyan]Device:[/cyan] {model_device}")
        console.print("[cyan]Opening OpenCV render window...[/cyan]")
        console.print("[dim]Press q or Esc in the render window to stop.[/dim]")

        model = load_model(resolved_checkpoint, algorithm, device=model_device)
        integration_path = env_config.get("stable_retro_integration_path")
        env = make_training_env(
            resolved_rom,
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
            forward_recovery_per_pixel=float(
                env_config.get("forward_recovery_per_pixel", 0.0)
            ),
            forward_recovery_debt_cap=float(env_config.get("forward_recovery_debt_cap", 5.0)),
            stable_retro_game=env_config.get("stable_retro_game", "Contra-Nes"),
            stable_retro_state=env_config.get("stable_retro_state", "Level1"),
            stable_retro_scenario=env_config.get("stable_retro_scenario"),
            stable_retro_info=env_config.get("stable_retro_info"),
            stable_retro_integration_path=Path(integration_path) if integration_path else None,
        )

        delay_seconds = 1.0 / fps
        total_reward = 0.0
        try:
            obs, info = env.reset(seed=42)
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            keep_running = _show_frame(current_rgb_frame(env), scale)

            for step_index in range(1, steps + 1):
                if not keep_running:
                    console.print("[yellow]Stopped by user.[/yellow]")
                    break

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
                keep_running = _show_frame(current_rgb_frame(env), scale)
                sleep(delay_seconds)

                if step_index % 120 == 0 or terminated or truncated:
                    console.print(
                        f"step={step_index} reward={total_reward:.3f} "
                        f"x={info.get('x_pos')} max_x={info.get('max_x_pos')} "
                        f"lives={info.get('lives')} score={info.get('score')}"
                    )

                if terminated or truncated:
                    status = "terminated" if terminated else "truncated"
                    console.print(f"[yellow]Episode {status}; stopping playback.[/yellow]")
                    break
        finally:
            env.close()
            cv2.destroyWindow(WATCH_WINDOW_NAME)
    except (ContraEnvError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def startup_preview(
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
    idle_frames: Annotated[
        int,
        typer.Option(min=0, help="Frames before startup action press."),
    ] = 240,
    pre_action: Annotated[
        str,
        typer.Option(help="Pre-action before startup pulses: none, start, a, b, a+b, right+a+b."),
    ] = "b",
    pre_presses: Annotated[
        int,
        typer.Option(min=0, help="Number of pre-action presses."),
    ] = 4,
    pre_press_frames: Annotated[
        int,
        typer.Option(min=1, help="Frames to hold each pre-action press."),
    ] = 1,
    pre_release_frames: Annotated[
        int,
        typer.Option(min=0, help="Frames to release between pre-action presses."),
    ] = 1,
    press_frames: Annotated[
        int,
        typer.Option(min=1, help="Frames holding each startup pulse."),
    ] = 1,
    release_frames: Annotated[
        int,
        typer.Option(min=1, help="Frames released between startup pulses."),
    ] = 1,
    attempt_frames: Annotated[
        int,
        typer.Option(min=1, help="Total frame budget for startup pulses."),
    ] = 480,
    after_frames: Annotated[
        int,
        typer.Option(min=0, help="Frames after startup action release."),
    ] = 60,
    startup_action: Annotated[
        str,
        typer.Option(help="Startup action: start, a, b, a+b, or right+a+b."),
    ] = "start",
    fps: Annotated[float, typer.Option(min=1.0, max=60.0, help="Playback speed.")] = 30.0,
    scale: Annotated[int, typer.Option(min=1, max=6, help="Window pixel scale.")] = 3,
) -> None:
    """Preview startup timing frame-by-frame without creating a gameplay backup."""
    resolved_rom = validate_rom_path(rom)
    delay_seconds = 1.0 / fps
    console.print(f"[cyan]ROM:[/cyan] {resolved_rom}")
    console.print(
        f"[cyan]Startup preview:[/cyan] idle={idle_frames}, "
        f"pre={pre_action}x{pre_presses}, "
        f"press={press_frames}, release={release_frames}, "
        f"attempt={attempt_frames}, after={after_frames}, action={startup_action}"
    )
    console.print("[dim]Press q or Esc in the render window to stop.[/dim]")

    env = ContraNesEnv(
        resolved_rom,
        render_mode="rgb_array",
        startup_idle_frames=0,
        start_press_frames=1,
        post_start_frames=0,
        create_start_backup=False,
    )
    try:
        env._env.reset()
        keep_running = _show_frame(env.observation("rgb_array"), scale)

        absolute_frame = 0
        startup_mask = _startup_action_mask(startup_action)

        phases = [("idle", idle_frames, NOOP)]
        for phase_name, frame_count, action in phases:
            for phase_frame in range(1, frame_count + 1):
                if not keep_running:
                    console.print("[yellow]Stopped by user.[/yellow]")
                    return
                absolute_frame += 1
                env._frame_advance(action)
                keep_running = _show_frame(env.observation("rgb_array"), scale)
                sleep(delay_seconds)
                if absolute_frame % 30 == 0:
                    console.print(
                        f"frame={absolute_frame} phase={phase_name} "
                        f"phase_frame={phase_frame} x={env._x_position} "
                        f"y={env._y_position} lives={env._lives}"
                    )

        pre_mask = _pre_action_mask(pre_action)
        for pre_index in range(1, pre_presses + 1):
            for pre_press_frame in range(1, pre_press_frames + 1):
                if not keep_running:
                    console.print("[yellow]Stopped by user.[/yellow]")
                    return
                absolute_frame += 1
                env._frame_advance(pre_mask)
                keep_running = _show_frame(env.observation("rgb_array"), scale)
                sleep(delay_seconds)
                if absolute_frame % 30 == 0:
                    console.print(
                        f"frame={absolute_frame} phase=pre "
                        f"pre={pre_index} pre_press_frame={pre_press_frame} "
                        f"x={env._x_position} y={env._y_position} lives={env._lives}"
                    )
            for pre_release_frame in range(1, pre_release_frames + 1):
                if not keep_running:
                    console.print("[yellow]Stopped by user.[/yellow]")
                    return
                absolute_frame += 1
                env._frame_advance(NOOP)
                keep_running = _show_frame(env.observation("rgb_array"), scale)
                sleep(delay_seconds)
                if absolute_frame % 30 == 0:
                    console.print(
                        f"frame={absolute_frame} phase=pre-release "
                        f"pre={pre_index} pre_release_frame={pre_release_frame} "
                        f"x={env._x_position} y={env._y_position} lives={env._lives}"
                    )

        attempted_frames = 0
        while attempted_frames < attempt_frames:
            for pulse_frame in range(1, press_frames + 1):
                if attempted_frames >= attempt_frames:
                    break
                if not keep_running:
                    console.print("[yellow]Stopped by user.[/yellow]")
                    return
                absolute_frame += 1
                attempted_frames += 1
                env._frame_advance(startup_mask)
                keep_running = _show_frame(env.observation("rgb_array"), scale)
                sleep(delay_seconds)
                if absolute_frame % 30 == 0:
                    console.print(
                        f"frame={absolute_frame} phase=pulse "
                        f"pulse_frame={pulse_frame} x={env._x_position} "
                        f"y={env._y_position} lives={env._lives}"
                    )
            for release_frame in range(1, release_frames + 1):
                if attempted_frames >= attempt_frames:
                    break
                if not keep_running:
                    console.print("[yellow]Stopped by user.[/yellow]")
                    return
                absolute_frame += 1
                attempted_frames += 1
                env._frame_advance(NOOP)
                keep_running = _show_frame(env.observation("rgb_array"), scale)
                sleep(delay_seconds)
                if absolute_frame % 30 == 0:
                    console.print(
                        f"frame={absolute_frame} phase=release "
                        f"release_frame={release_frame} x={env._x_position} "
                        f"y={env._y_position} lives={env._lives}"
                    )

        for after_frame in range(1, after_frames + 1):
            if not keep_running:
                console.print("[yellow]Stopped by user.[/yellow]")
                return
            absolute_frame += 1
            env._frame_advance(NOOP)
            keep_running = _show_frame(env.observation("rgb_array"), scale)
            sleep(delay_seconds)
            if absolute_frame % 30 == 0:
                console.print(
                    f"frame={absolute_frame} phase=after "
                    f"after_frame={after_frame} x={env._x_position} "
                    f"y={env._y_position} lives={env._lives}"
                )
    finally:
        env.close()
        cv2.destroyWindow(WATCH_WINDOW_NAME)


@app.command()
def manual(
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
    config: Annotated[
        Path,
        typer.Option(
            help="Stable Retro training config that defines the game state and RAM setup."
        ),
    ] = Path("configs/ppo_recurrent_stable_retro.yaml"),
    steps: Annotated[
        int,
        typer.Option(min=0, help="Maximum emulator frames; 0 means run until you quit."),
    ] = 0,
    fps: Annotated[float, typer.Option(min=1.0, max=60.0, help="Playback speed.")] = 30.0,
    scale: Annotated[int, typer.Option(min=1, max=6, help="Window pixel scale.")] = 3,
    log_every: Annotated[
        int,
        typer.Option(min=1, help="Print state every N emulator frames."),
    ] = 30,
) -> None:
    """Play the Stable Retro start state with smooth held-key keyboard input."""
    try:
        resolved_rom = validate_rom_path(rom)
        resolved_config = config.expanduser().resolve()
        manual_config = load_training_config(resolved_config)
        env_config = manual_config.get("env", {})
        if env_config.get("backend", "nes-py") != "stable-retro":
            raise ValueError("manual play requires a stable-retro config")

        retro = _import_stable_retro()
        integration_path = env_config.get("stable_retro_integration_path")
        if integration_path:
            retro.data.Integrations.add_custom_path(
                str(Path(integration_path).expanduser().resolve())
            )
            integration_type = retro.data.Integrations.ALL
        else:
            integration_type = retro.data.Integrations.DEFAULT

        raw_env = retro.make(
            game=env_config.get("stable_retro_game", "Contra-Nes"),
            state=env_config.get("stable_retro_state", "Level1"),
            scenario=env_config.get("stable_retro_scenario"),
            info=env_config.get("stable_retro_info"),
            inttype=integration_type,
            use_restricted_actions=retro.Actions.ALL,
            render_mode="rgb_array",
        )
        env = StableRetroContraEnv(
            raw_env,
            max_episode_steps=10**9,
            stuck_timeout_steps=10**9,
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
            terminal_efficiency_min_x=int(env_config.get("terminal_efficiency_min_x", 256)),
            terminal_efficiency_max_penalty=float(
                env_config.get("terminal_efficiency_max_penalty", 25.0)
            ),
            idle_penalty_per_step=float(env_config.get("idle_penalty_per_step", 0.0)),
            idle_penalty_start_steps=int(env_config.get("idle_penalty_start_steps", 0)),
            forward_recovery_per_pixel=float(
                env_config.get("forward_recovery_per_pixel", 0.0)
            ),
            forward_recovery_debt_cap=float(env_config.get("forward_recovery_debt_cap", 5.0)),
        )
        pygame = _import_pygame()
        console.print(f"[cyan]ROM:[/cyan] {resolved_rom}")
        console.print(f"[cyan]Config:[/cyan] {resolved_config}")
        console.print("[cyan]Manual controls:[/cyan]")
        console.print("  W / A / S / D = up / left / down / right")
        console.print("  J = B (shoot), K = A (jump)")
        console.print("  Enter = Start, Space = Select, q or Esc = quit")
        console.print("[dim]Hold any combination: D+J runs and shoots; S+K is down+jump.[/dim]")
        console.print("[dim]The Pygame window must have keyboard focus.[/dim]")

        try:
            frame, info = env.reset()
            height, width = frame.shape[:2]
            pygame.init()
            pygame.display.set_caption(MANUAL_WINDOW_NAME)
            screen = pygame.display.set_mode((width * scale, height * scale))
            clock = pygame.time.Clock()
            frame_index = 0
            previous_buttons: frozenset[str] | None = None
            keep_running = True
            while keep_running and (steps == 0 or frame_index < steps):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        keep_running = False
                    elif event.type == pygame.KEYDOWN and event.key in {
                        pygame.K_ESCAPE,
                        pygame.K_q,
                    }:
                        keep_running = False

                if not keep_running:
                    break

                frame_index += 1
                buttons = _manual_buttons_from_keyboard(pygame.key.get_pressed(), pygame)
                action = _manual_stable_retro_action(env, buttons)
                frame, reward, terminated, truncated, info = env.step(action)

                if buttons != previous_buttons or frame_index % log_every == 0:
                    pressed = "+".join(sorted(buttons)) if buttons else "NOOP"
                    console.print(
                        f"frame={frame_index} buttons={pressed} reward={reward:.3f} "
                        f"x={info.get('x_pos')} max_x={info.get('max_x_pos')} "
                        f"score={info.get('score')} lives={info.get('lives')}"
                    )
                    previous_buttons = buttons

                if terminated or truncated:
                    status = "terminated" if terminated else "truncated"
                    console.print(f"[yellow]Episode {status}; resetting to Level1.[/yellow]")
                    frame, info = env.reset()

                _show_pygame_frame(screen, frame, scale, pygame)
                clock.tick(fps)
        finally:
            env.close()
            pygame.quit()
    except (ContraEnvError, StableRetroUnavailableError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def capture_start_state(
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
) -> None:
    """Explain why persistent start-state capture is unavailable."""
    resolved_rom = validate_rom_path(rom)
    console.print(f"[cyan]ROM:[/cyan] {resolved_rom}")
    console.print(
        "[yellow]Persistent save-state capture is not supported by this nes-py build.[/yellow]"
    )
    console.print(
        "The native emulator snapshot returned by dump_state() cannot be serialized to disk. "
        "The environment now uses the validated menu-skip sequence and creates an in-memory "
        "backup after gameplay starts."
    )
    console.print("Use this instead:")
    console.print(
        "[green]contra-rl watch --rom .\\roms\\Contra.nes --policy noop --fps 60 --scale 2[/green]"
    )
    raise typer.Exit(code=1)


@app.command()
def check_env(
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
    config: Annotated[
        Path | None,
        typer.Option(help="Optional training config YAML to source env settings from."),
    ] = None,
    action_set: Annotated[str, typer.Option(help="Action set name.")] = "SIMPLE_MOVEMENT",
    frame_skip: Annotated[int, typer.Option(min=1, help="Frame skip/action repeat.")] = 4,
    screen_size: Annotated[int, typer.Option(min=1, help="Processed image size.")] = 84,
    grayscale: Annotated[bool, typer.Option(help="Use grayscale observations.")] = True,
    frame_stack: Annotated[int, typer.Option(min=1, help="Number of frames to stack.")] = 4,
    steps: Annotated[int, typer.Option(min=1, help="Random steps after checker.")] = 20,
) -> None:
    """Build the training env and run Stable-Baselines3 compatibility checks."""
    try:
        resolved_rom = validate_rom_path(rom)
        env_config = {}
        resolved_config = None
        if config is not None:
            resolved_config = config.expanduser().resolve()
            env_config = load_training_config(resolved_config).get("env", {})

        selected_action_set = env_config.get("action_set", action_set)
        integration_path = env_config.get("stable_retro_integration_path")
        _require_action_set(selected_action_set)
        console.print(f"[cyan]ROM:[/cyan] {resolved_rom}")
        if resolved_config is not None:
            console.print(f"[cyan]Config:[/cyan] {resolved_config}")
        console.print(f"[cyan]Backend:[/cyan] {env_config.get('backend', 'nes-py')}")
        console.print("[cyan]Building training environment...[/cyan]")
        env = make_training_env(
            resolved_rom,
            backend=env_config.get("backend", "nes-py"),
            action_set=selected_action_set,
            frame_skip=int(env_config.get("frame_skip", frame_skip)),
            screen_size=int(env_config.get("screen_size", screen_size)),
            grayscale=bool(env_config.get("grayscale", grayscale)),
            frame_stack=int(env_config.get("frame_stack", frame_stack)),
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
            forward_recovery_per_pixel=float(
                env_config.get("forward_recovery_per_pixel", 0.0)
            ),
            forward_recovery_debt_cap=float(env_config.get("forward_recovery_debt_cap", 5.0)),
            stable_retro_game=env_config.get("stable_retro_game", "Contra-Nes"),
            stable_retro_state=env_config.get("stable_retro_state", "Level1"),
            stable_retro_scenario=env_config.get("stable_retro_scenario"),
            stable_retro_info=env_config.get("stable_retro_info"),
            stable_retro_integration_path=Path(integration_path) if integration_path else None,
        )
        try:
            obs, info = env.reset(seed=42)
            console.print(f"observation_space={env.observation_space}")
            console.print(f"action_space={env.action_space}")
            console.print(f"reset obs shape={obs.shape} dtype={obs.dtype}")
            console.print(f"reset info x={info.get('x_pos')} lives={info.get('lives')}")

            console.print("[cyan]Running SB3 env checker...[/cyan]")
            sb3_check_env(env, warn=True)
            console.print("[green]SB3 env checker passed.[/green]")

            total_reward = 0.0
            rng = np.random.default_rng(42)
            for step_index in range(1, steps + 1):
                action = int(rng.integers(env.action_space.n))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    console.print(f"episode ended during random step {step_index}; resetting")
                    obs, info = env.reset(seed=42 + step_index)

            console.print(
                f"[green]Random step smoke passed.[/green] steps={steps} "
                f"reward={total_reward:.3f} final_obs_shape={obs.shape}"
            )
        finally:
            env.close()
    except ContraEnvError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def retro_status(
    game: Annotated[str, typer.Option(help="Stable Retro game id to check.")] = "Contra-Nes",
    integration_path: Annotated[
        Path,
        typer.Option(help="Custom Stable Retro integrations directory."),
    ] = Path("integrations"),
) -> None:
    """Check Stable Retro install and Contra integration visibility."""
    try:
        retro = _import_stable_retro()
        resolved_integration_path = integration_path.expanduser().resolve()
        if resolved_integration_path.exists():
            retro.data.Integrations.add_custom_path(str(resolved_integration_path))

        games = set(retro.data.list_games(inttype=retro.data.Integrations.ALL))
        console.print("[green]stable-retro import succeeded.[/green]")
        console.print(f"integration_path={resolved_integration_path}")
        console.print(f"game={game}")
        console.print(f"game_visible={game in games}")
        if game not in games:
            console.print(
                "[yellow]Contra integration is not visible yet. Check the integration "
                "folder name and install/import Stable Retro artifacts.[/yellow]"
            )
    except StableRetroUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            "[yellow]On Windows, stable-retro may need a Python 3.11 venv or local build "
            "tools if no wheel is available for Python 3.13.[/yellow]"
        )
        raise typer.Exit(code=1) from exc


@app.command()
def retro_import_rom(
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
    game: Annotated[str, typer.Option(help="Stable Retro game id.")] = "Contra-Nes",
    integration_path: Annotated[
        Path,
        typer.Option(help="Custom Stable Retro integrations directory."),
    ] = Path("integrations"),
    force: Annotated[bool, typer.Option(help="Overwrite an existing imported rom.nes.")] = False,
) -> None:
    """Import a local NES ROM into this project's custom Stable Retro integration."""
    resolved_rom = validate_rom_path(rom)
    game_dir = integration_path.expanduser().resolve() / game
    sha_path = game_dir / "rom.sha"
    output_path = game_dir / "rom.nes"

    if not game_dir.exists():
        raise typer.BadParameter(f"integration folder not found: {game_dir}")
    if not sha_path.exists():
        raise typer.BadParameter(f"rom.sha not found: {sha_path}")
    if output_path.exists() and not force:
        console.print(f"[yellow]ROM already imported:[/yellow] {output_path}")
        console.print("Use --force to overwrite it.")
        return

    rom_bytes = resolved_rom.read_bytes()
    if len(rom_bytes) <= 16:
        raise typer.BadParameter("NES ROM is too small to contain an iNES header and body.")

    # Stable Retro hashes .nes files without the 16-byte iNES header.
    stable_retro_sha = hashlib.sha1(rom_bytes[16:]).hexdigest()
    expected_hashes = {
        line.strip().lower()
        for line in sha_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if stable_retro_sha not in expected_hashes:
        expected = ", ".join(sorted(expected_hashes)) or "<empty rom.sha>"
        raise typer.BadParameter(
            "ROM hash does not match this integration. "
            f"stable_retro_sha={stable_retro_sha} expected={expected}"
        )

    shutil.copyfile(resolved_rom, output_path)
    console.print(f"[green]Imported ROM:[/green] {output_path}")
    console.print(f"stable_retro_sha={stable_retro_sha}")


@app.command()
def train(
    config: Annotated[Path, typer.Option(help="Path to a training config YAML file.")],
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
    total_timesteps: Annotated[
        int | None,
        typer.Option(min=1, help="Override total training timesteps."),
    ] = None,
    n_envs: Annotated[
        int | None,
        typer.Option(min=1, help="Override number of parallel environments."),
    ] = None,
    run_name: Annotated[
        str | None,
        typer.Option(help="Override run name under runs/."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(help="Override device: cuda, cpu, or auto."),
    ] = None,
    metrics_log_freq: Annotated[
        int | None,
        typer.Option(min=1, help="Override Contra metrics log frequency."),
    ] = None,
    resume_checkpoint: Annotated[
        Path | None,
        typer.Option(
            help="Compatible PPO/RecurrentPPO checkpoint to continue training from."
        ),
    ] = None,
) -> None:
    """Train an agent."""
    try:
        resolved_rom = validate_rom_path(rom)
        resolved_config = config.expanduser().resolve()
        training_config = load_training_config(resolved_config)
        training_config = apply_training_overrides(
            training_config,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            run_name=run_name,
            device=device,
            metrics_log_freq=metrics_log_freq,
        )

        algorithm = training_config.get("algorithm", "PPO")
        env_config = training_config.get("env", {})
        train_config = training_config.get("training", {})
        logging_config = training_config.get("logging", {})

        console.print(f"[cyan]ROM:[/cyan] {resolved_rom}")
        console.print(f"[cyan]Config:[/cyan] {resolved_config}")
        console.print(f"[cyan]Algorithm:[/cyan] {algorithm}")
        console.print(f"[cyan]Run name:[/cyan] {logging_config.get('run_name')}")
        console.print(f"[cyan]Device:[/cyan] {training_config.get('device')}")
        console.print(f"[cyan]Parallel envs:[/cyan] {env_config.get('n_envs')}")
        console.print(f"[cyan]Total timesteps:[/cyan] {train_config.get('total_timesteps')}")
        if resume_checkpoint is not None:
            resolved_resume = resume_checkpoint.expanduser().resolve()
            console.print(f"[cyan]Resume checkpoint:[/cyan] {resolved_resume}")
        console.print("[cyan]Starting training...[/cyan]")

        result = train_model(
            resolved_rom,
            training_config,
            resume_checkpoint=resume_checkpoint,
        )
        console.print("[green]Training finished.[/green]")
        console.print(f"run_dir={result['run_dir']}")
        console.print(f"final_model={result['final_model_path']}")
        console.print(f"tensorboard={result['tensorboard_dir']}")
        console.print(f"checkpoints={result['checkpoint_dir']}")
    except (ContraEnvError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def eval(
    checkpoint: Annotated[Path, typer.Option(help="Path to a trained checkpoint.")],
    rom: Annotated[Path, typer.Option(help="Path to the local Contra NES ROM.")],
    config: Annotated[
        Path,
        typer.Option(help="Path to the training config used for the checkpoint."),
    ] = Path("configs/ppo_baseline.yaml"),
    episodes: Annotated[int, typer.Option(min=1, help="Evaluation episodes.")] = 1,
    max_steps: Annotated[int, typer.Option(min=1, help="Max agent steps per episode.")] = 5000,
    deterministic: Annotated[bool, typer.Option(help="Use deterministic policy actions.")] = True,
    record: Annotated[bool, typer.Option(help="Record an evaluation video.")] = False,
    video_path: Annotated[
        Path | None,
        typer.Option(help="Optional MP4 output path when recording."),
    ] = None,
    video_fps: Annotated[
        float,
        typer.Option(
            min=1.0,
            max=120.0,
            help="Recorded video FPS. Default matches frame_skip=4 at 60 FPS NES.",
        ),
    ] = 15.0,
    device: Annotated[
        str | None,
        typer.Option(help="Override device: cuda, cpu, or auto."),
    ] = None,
) -> None:
    """Evaluate a trained agent."""
    try:
        resolved_rom = validate_rom_path(rom)
        resolved_checkpoint = checkpoint.expanduser().resolve()
        if not resolved_checkpoint.exists():
            raise ValueError(f"checkpoint not found: {resolved_checkpoint}")

        resolved_config = config.expanduser().resolve()
        evaluation_config = load_training_config(resolved_config)
        if device is not None:
            evaluation_config["device"] = device

        if record and video_path is None:
            checkpoint_stem = resolved_checkpoint.stem
            video_path = Path("runs") / "eval_videos" / f"{checkpoint_stem}_eval.mp4"
        resolved_video_path = video_path.expanduser().resolve() if video_path else None

        console.print(f"[cyan]ROM:[/cyan] {resolved_rom}")
        console.print(f"[cyan]Checkpoint:[/cyan] {resolved_checkpoint}")
        console.print(f"[cyan]Config:[/cyan] {resolved_config}")
        console.print(f"[cyan]Episodes:[/cyan] {episodes}")
        console.print(f"[cyan]Max steps:[/cyan] {max_steps}")
        if resolved_video_path is not None:
            console.print(f"[cyan]Recording:[/cyan] {resolved_video_path}")

        result = evaluate_model(
            resolved_rom,
            resolved_checkpoint,
            evaluation_config,
            episodes=episodes,
            max_steps=max_steps,
            deterministic=deterministic,
            record_path=resolved_video_path if record else None,
            video_fps=video_fps,
            device=device,
        )

        console.print("[green]Evaluation finished.[/green]")
        console.print(f"mean_reward={result['mean_reward']:.3f}")
        console.print(f"mean_max_x_pos={result['mean_max_x_pos']:.1f}")
        console.print(f"best_max_x_pos={result['best_max_x_pos']:.1f}")
        for episode in result["episodes"]:
            console.print(
                f"episode={episode['episode']} reward={episode['reward']:.3f} "
                f"steps={episode['steps']} max_x={episode['max_x_pos']:.1f} "
                f"score={episode['score']} terminated={episode['terminated']} "
                f"truncated={episode['truncated']}"
            )
        if result["record_path"] is not None:
            console.print(f"video={result['record_path']}")
            console.print(
                f"video_frames={result['video_frames']} "
                f"video_fps={result['video_fps']} "
                f"duration_seconds={result['video_duration_seconds']:.2f}"
            )
    except (ContraEnvError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
