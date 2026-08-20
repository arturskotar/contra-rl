"""Training callbacks for Contra-specific metrics."""

import re
from collections import defaultdict
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ContraMetricsCallback(BaseCallback):
    """Log Contra-specific metrics from Gymnasium info dictionaries."""

    def __init__(self, *, log_freq: int = 1_000) -> None:
        super().__init__()
        self.log_freq = log_freq
        self._latest_values: dict[str, list[float]] = defaultdict(list)
        self._action_counts: dict[str, int] = defaultdict(int)
        self._action_total = 0

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            self._record_info(info)

        if self.num_timesteps % self.log_freq == 0:
            self._dump_latest_values()

        return True

    def _record_info(self, info: dict[str, Any]) -> None:
        scalar_keys = [
            "x_pos",
            "max_x_pos",
            "max_progress_bucket",
            "y_pos",
            "lives",
            "score",
            "episode_steps",
            "steps_since_progress",
            "steps_since_useful_event",
            "idle_debt",
        ]
        for key in scalar_keys:
            value = info.get(key)
            if isinstance(value, int | float):
                self._latest_values[f"contra/{key}"].append(float(value))

        if info.get("is_dead") is not None:
            self._latest_values["contra/is_dead"].append(float(bool(info["is_dead"])))
        if info.get("life_lost") is not None:
            self._latest_values["contra/life_lost"].append(float(bool(info["life_lost"])))
        if info.get("game_over") is not None:
            self._latest_values["contra/game_over"].append(float(bool(info["game_over"])))

        action_index = info.get("action_index")
        action_name = info.get("action_name")
        if isinstance(action_index, int):
            self._latest_values["actions/selected_index"].append(float(action_index))
        if isinstance(action_name, str):
            safe_name = _safe_metric_name(action_name)
            self._action_counts[safe_name] += 1
            self._action_total += 1

        reward_parts = info.get("reward_parts")
        if isinstance(reward_parts, dict):
            for key, value in reward_parts.items():
                if isinstance(value, int | float):
                    self._latest_values[f"reward_parts/{key}"].append(float(value))

    def _dump_latest_values(self) -> None:
        if not self._latest_values:
            return

        for key, values in self._latest_values.items():
            if values:
                self.logger.record(key, float(np.mean(values)))
                self.logger.record(f"{key}_max", float(np.max(values)))

        if self._action_total > 0:
            for key, count in self._action_counts.items():
                self.logger.record(f"actions/{key}_rate", count / self._action_total)
                self.logger.record(f"actions/{key}_count", count)

        self._latest_values.clear()
        self._action_counts.clear()
        self._action_total = 0


def _safe_metric_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_+-]+", "_", name.strip())
    return safe or "NOOP"
