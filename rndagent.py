"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is" without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv

import tetrisenv  # noqa: F401  # pylint: disable=unused-import


class RandomPolicy(BasePolicy):
    """Random policy that follows Stable Baselines3 policy interface."""

    def __init__(self, observation_space, action_space, lr_schedule):
        super().__init__(observation_space, action_space)
        self.action_space = action_space

    def _predict(self, observation, deterministic=False):
        return self.action_space.sample()

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Match SB3 interface"""
        if isinstance(observation, tuple):
            observation = observation[0]
        actions = np.array([self.action_space.sample() for _ in range(observation.shape[0])])
        return actions, state

    def forward(self, *args, **kwargs):
        raise NotImplementedError("RandomPolicy does not use forward pass.")

    def _get_data(self):
        return {}


class RandomAgent(OffPolicyAlgorithm):
    """Random agent that follows Stable Baselines3 algorithm interface."""

    def __init__(self, env, seed=None, device="auto", tensorboard_log=None):
        super().__init__(
            policy=RandomPolicy,
            env=env,
            policy_kwargs={},
            learning_rate=1e-3,  # Dummy value
            buffer_size=1,
            learning_starts=0,
            batch_size=1,
            tau=1.0,
            gamma=0.0,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            stats_window_size=100,
            tensorboard_log=tensorboard_log,
            verbose=0,
            device=device,
            seed=seed,
        )
        self.policy = RandomPolicy(self.observation_space, self.action_space, None)

    def train(self, gradient_steps: int, batch_size: int):
        """No training for random agent."""
        return None

    def learn(self, *args, **kwargs):
        """No learning for random agent."""
        return self
