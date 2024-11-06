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
    """
    A feature extractor for Tetris game observations using a Convolutional Neural Network (CNN).

    Args:
        observation_space (spaces.Box): The observation space of the environment.
        features_dim (int, optional): The dimension of the output features. Default is 128.
        num_kernels (int, optional): The number of kernels (filters) in the convolutional layer. Default is 16.
        kernel_size (int, optional): The size of the convolutional kernels. Default is 3.

    Attributes:
        cnn (nn.Sequential): The convolutional neural network consisting of a Conv2D layer, ReLU activation, and Flatten layer.
        linear (nn.Sequential): A linear layer followed by ReLU activation to produce the final feature vector.

    Methods:
        forward(observations: torch.Tensor) -> torch.Tensor:
            Processes the input observations through the CNN and linear layers to extract features.
    """

    def __init__(
        self, observation_space: spaces.Box, features_dim: int = 128, num_kernels: int = 16, kernel_size: int = 3
    ):
        super().__init__(observation_space, features_dim)
        n_chan = observation_space.shape[0]

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
        """Process board through CNN and linear"""
        board_features = self.cnn(observations)
        return self.linear(board_features)


class TetrisFeatureExtractor2(BaseFeaturesExtractor):
    """
    A custom feature extractor for Tetris observations using a Convolutional Neural Network (CNN).
    The network consists of two convolutional layers followed by a linear layer.

    Args:
        observation_space (spaces.Box): The observation space of the environment.
        features_dim (int, optional): The dimension of the output features. Default is 128.
        num_kernels (tuple[int, int], optional): A tuple containing the number of kernels for each convolutional layer. Default is (32, 64).
        kernel_sizes (tuple[int, int], optional): A tuple containing the kernel sizes for each convolutional layer. Default is (3, 3).

    Attributes:
        cnn (nn.Sequential): The CNN used to process the input observations.
        linear (nn.Sequential): A linear layer to transform the CNN output to the desired feature dimension.

    Methods:
        forward(observations: torch.Tensor) -> torch.Tensor:
            Processes the input observations through the CNN and linear layers to extract features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        num_kernels: tuple[int, int] = (32, 64),
        kernel_sizes: tuple[int, int] = (3, 3),
    ):
        super().__init__(observation_space, features_dim)
        n_chan = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(n_chan, num_kernels[0], kernel_size=kernel_sizes[0], stride=1, padding=1),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(num_kernels[0], num_kernels[1], kernel_size=kernel_sizes[1], stride=1, padding=1),
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


class TetrisFeatureExtractor3(BaseFeaturesExtractor):
    """
    A custom feature extractor for Tetris observations using a convolutional neural network (CNN).
    The network consists of three convolutional layers followed by a linear layer.

    Args:
        observation_space (spaces.Box): The observation space of the environment.
        features_dim (int, optional): The dimension of the output features. Defaults to 128.
        num_kernels (tuple[int, int, int], optional): A tuple specifying the number of kernels for each convolutional layer. Defaults to (32, 64, 128).
        kernel_sizes (tuple[int, int, int], optional): A tuple specifying the kernel sizes for each convolutional layer. Defaults to (3, 3, 3).
        kernel_strides (tuple[int, int, int], optional): A tuple specifying the strides for each convolutional layer. Defaults to (2, 2, 2).

    Attributes:
        cnn (nn.Sequential): The convolutional neural network used to extract features from the observations.
        linear (nn.Sequential): A linear layer to map the CNN output to the desired feature dimension.

    Methods:
        forward(observations: torch.Tensor) -> torch.Tensor:
            Processes the input observations through the CNN and linear layers to extract features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        num_kernels: tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: tuple[int, int, int] = (3, 3, 3),
        kernel_strides: tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__(observation_space, features_dim)
        n_chan = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(n_chan, num_kernels[0], kernel_size=kernel_sizes[0], stride=kernel_strides[0], padding=1),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(num_kernels[0], num_kernels[1], kernel_size=kernel_sizes[1], stride=kernel_strides[1], padding=1),
            nn.ReLU(),
            # Third conv layer
            nn.Conv2d(num_kernels[1], num_kernels[2], kernel_size=kernel_sizes[2], stride=kernel_strides[2], padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():  # Don't track operations, we just need to calculate the output size
            example_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output_dim = self.cnn(example_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        """Process board through CNN and linear"""
        board_features = self.cnn(observations)
        return self.linear(board_features)


class TetrisFeatureExtractor4(BaseFeaturesExtractor):
    """
    A custom feature extractor for Tetris game observations using a Convolutional Neural Network (CNN).
    The network consists of three convolutional layers with max pooling followed by a linear layer.

    Args:
        observation_space (spaces.Box): The observation space of the environment.
        features_dim (int, optional): The dimension of the output features. Default is 128.
        num_kernels (tuple[int, int, int], optional): A tuple specifying the number of kernels for each convolutional layer. Default is (32, 64, 64).
        kernel_sizes (tuple[int, int, int], optional): A tuple specifying the kernel sizes for each convolutional layer. Default is (5, 3, 3).
        pooling_kernel_sizes (tuple[int, int], optional): A tuple specifying the kernel sizes for each max pooling layer. Default is (2, 2).

    Attributes:
        cnn (nn.Sequential): The sequential container for the convolutional layers and activation functions.
        linear (nn.Sequential): The sequential container for the linear layer and activation function.

    Methods:
        forward(observations: torch.Tensor) -> torch.Tensor:
            Process the input observations through the CNN and linear layers to extract features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        num_kernels: tuple[int, int, int] = (32, 64, 64),
        kernel_sizes: tuple[int, int, int] = (5, 3, 3),
        pooling_kernel_sizes: tuple[int, int] = (2, 2),
    ):
        super().__init__(observation_space, features_dim)
        n_chan = observation_space.shape[0]

        # Playing Tetris with Deep Reinforcement Learning paper
        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(n_chan, num_kernels[0], kernel_size=kernel_sizes[0], stride=1, padding=1),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(num_kernels[0], num_kernels[1], kernel_size=kernel_sizes[1], stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_sizes[0], stride=2),
            # Third conv layer
            nn.Conv2d(num_kernels[1], num_kernels[2], kernel_size=kernel_sizes[2], stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_sizes[1], stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():  # Don't track operations, we just need to calculate the output size
            example_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output_dim = self.cnn(example_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        """Process board through CNN and linear"""
        board_features = self.cnn(observations)
        return self.linear(board_features)


class TetrisFeatureExtractor5(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_chan = observation_space.shape[0]

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
        """Process board through CNN and linear"""
        board_features = self.cnn(observations)
        return self.linear(board_features)


class TetrisFeatureExtractor6(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_chan = observation_space.shape[0]

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
        """Process board through CNN and linear"""
        board_features = self.cnn(observations)
        return self.linear(board_features)
