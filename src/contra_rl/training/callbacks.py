"""Training callbacks for Contra-specific metrics."""

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
            "y_pos",
            "lives",
            "score",
            "episode_steps",
            "steps_since_progress",
        ]
        for key in scalar_keys:
            value = info.get(key)
            if isinstance(value, int | float):
                self._latest_values[f"contra/{key}"].append(float(value))

        if info.get("is_dead") is not None:
            self._latest_values["contra/is_dead"].append(float(bool(info["is_dead"])))

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

        self._latest_values.clear()
