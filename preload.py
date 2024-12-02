"""
Preload module for Tetris evaluation and play functionality.
Import this in IPython to avoid module loading overhead.
"""

from typing import Optional
import IPython
import argparse
from pathlib import Path

# Preload all heavy imports
import gymnasium as gym
import numpy as np
import pygame
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Import local modules
import tetrisenv
from play import main as play_main
from eval import main as eval_main


def play(cmdline: Optional[str] = None) -> None:
    """
    Play Tetris using a trained model or random agent.

    Examples:
        play("--model-file model.zip --pygame")  # Play using trained model with pygame
        play("--random --pygame")                # Play using random agent
        play("--ascii")                          # Play in terminal with ASCII renderer
    """
    play_main(cmdline)


def evaluate(cmdline: Optional[str] = None) -> None:
    """
    Evaluate a trained Tetris model.

    Examples:
        evaluate("--model-file model.zip --episodes 10")  # Evaluate model for 10 episodes
        evaluate("--model-file model.zip --render")       # Evaluate with visualization
    """
    eval_main(cmdline)


print("Tetris environment loaded. Available commands:")
print("  play()     - Play Tetris (use play? for help)")
print("  evaluate() - Evaluate model (use evaluate? for help)")

IPython.embed()
