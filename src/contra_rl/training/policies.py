"""Neural-network components used by Contra training experiments."""

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class ContraCNN(BaseFeaturesExtractor):
    """A detail-preserving visual encoder for 84x84 stacked Contra frames.

    Compared with SB3's Atari NatureCNN, its first layer uses a 5x5 kernel and
    stride 2 rather than an 8x8 kernel and stride 4. That preserves more of
    Contra's small bullets, enemy sprites, and foreground hazards.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512) -> None:
        if len(observation_space.shape) != 3:
            raise ValueError("ContraCNN expects channel-first image observations")

        super().__init__(observation_space, features_dim)
        input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sampled_observation = th.as_tensor(observation_space.sample()[None]).float()
            flattened_size = self.cnn(sampled_observation).shape[1]

        self.linear = nn.Sequential(nn.Linear(flattened_size, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
