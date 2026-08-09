"""Reward shaping helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardParts:
    progress: float = 0.0
    score: float = 0.0
    weapon: float = 0.0
    death: float = 0.0
    time: float = 0.0
    stuck: float = 0.0

    @property
    def total(self) -> float:
        return self.progress + self.score + self.weapon + self.death + self.time + self.stuck
