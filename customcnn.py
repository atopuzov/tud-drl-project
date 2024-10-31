"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TetrisFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: spaces.Dict, features_dim: int = 128, num_kernels: int = 16, kernel_size: int = 3
    ):
        super().__init__(observation_space, features_dim)
        (n_chan, board_height, board_width) = observation_space.shape

        # TODO: reduce kernel size -> 2
        # TODO: stride - 2 (downscale)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_chan, num_kernels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():  # Don't track operations, we just need to calculate the output size
            # example_input = torch.zeros(1, n_chan, board_height, board_width)
            example_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output_dim = self.cnn(example_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        # Process board through CNN and linear
        board_features = self.cnn(observations)
        return self.linear(board_features)


class TetrisFeatureExtractor2(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        (n_chan, board_height, board_width) = observation_space.shape

        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(n_chan, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # TODO: Verify
        with torch.no_grad():  # Don't track operations, we just need to calculate the output size
            example_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output_dim = self.cnn(example_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        # Process board through CNN and linear
        board_features = self.cnn(observations)
        return self.linear(board_features)


class TetrisFeatureExtractor3(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        (n_chan, board_height, board_width) = observation_space.shape

        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(n_chan, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Third conv layer
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():  # Don't track operations, we just need to calculate the output size
            example_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output_dim = self.cnn(example_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        # Process board through CNN and linear
        board_features = self.cnn(observations)
        return self.linear(board_features)


class TetrisFeatureExtractor4(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        (n_chan, board_height, board_width) = observation_space.shape

        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(n_chan, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third conv layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():  # Don't track operations, we just need to calculate the output size
            example_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output_dim = self.cnn(example_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        # Process board through CNN and linear
        board_features = self.cnn(observations)
        return self.linear(board_features)


class TetrisFeatureExtractor5(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        (n_chan, board_height, board_width) = observation_space.shape

        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(n_chan, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Third conv layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():  # Don't track operations, we just need to calculate the output size
            example_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output_dim = self.cnn(example_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        # Process board through CNN and linear
        board_features = self.cnn(observations)
        return self.linear(board_features)


class TetrisFeatureExtractor5(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        (n_chan, board_height, board_width) = observation_space.shape

        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(n_chan, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # For stability
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Optional MaxPool to reduce spatial dimensions
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        with torch.no_grad():  # Don't track operations, we just need to calculate the output size
            example_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output_dim = self.cnn(example_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        # Process board through CNN and linear
        board_features = self.cnn(observations)
        return self.linear(board_features)
