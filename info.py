"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is" without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from stable_baselines3 import DQN

# from torchviz import make_dot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game of Tetris")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-file", type=Path, default="tetris_model.zip", help="Model file")
    parser.add_argument("--env-name", type=str, default="Tetris-v3", help="Use SubprocVecEnv")
    args = parser.parse_args()

    try:
        model = DQN.load(args.model_file)
    except FileNotFoundError:
        print(f"Unable to find {args.model_file}")
        sys.exit(-1)

    print("\nBasic Information:")
    print(f"Algorithm: {type(model).__name__}")
    print(f"Device: {model.device}")
    print(f"Learning Rate: {model.learning_rate}")
    print(f"Gamma (discount): {model.gamma}")
    print(f"Buffer Size: {model.buffer_size}")
    print(f"Learning Starts: {model.learning_starts}")
    print(f"Target Update Interval: {model.target_update_interval}")
    print(f"Train Frequency: {model.train_freq}")
    print(f"Gradient Steps: {model.gradient_steps}")
    print(f"Exploration Fraction: {model.exploration_fraction}")
    print(f"Exploration Initial Epsilon: {model.exploration_initial_eps}")
    print(f"Exploration Final Epsilon: {model.exploration_final_eps}")

    # Policy network architecture
    print("\nPolicy Network:")
    print(f"Network Type: {type(model.policy).__name__}")
    print("\nNetwork Architecture:")
    print(model.policy)

    # Visualize the policy network
    # https://github.com/LucasRedshaw/PokeAI/blob/f53f2484df6910b9f9cf970188050c42d8900f1d/vis.py#L46-L65
    # Create a dictionary to store intermediate outputs
    # activations = OrderedDict()

    # # Register hooks on each layer in `model.policy`
    # def register_hooks(model):
    #     for name, layer in model.named_children():
    #         print(f"Layer: {name}")
    #         layer.register_forward_hook(
    #             lambda layer, input, output: activations.update({name: output})
    #         )

    # # Register hooks for both `features_extractor` and `q_net`
    # register_hooks(model.policy.q_net)
    # register_hooks(model.policy.q_net_target)

    # Perform a forward pass to populate activations
    # dummy_input = torch.randn(1, 1, 10, 20)
    # output = model.policy(dummy_input)
    # dot = make_dot(activations["q_net"], params=dict(model.policy.named_parameters()))
    # dot = make_dot(output, params=dict(model.policy.named_parameters()))

    # Save or render the visualization
    # dot.format = "png"
    # dot.render("network_architecture")

    # from torchsummary import summary

    # summary(model.policy, input_size=(1, 10, 20))  # Adjust input_size if needed

    # Perform a forward pass and visualize only the structural details
    # output = model.policy(dummy_input)
    # dot = make_dot(
    #     activations["q_net"],
    #     params=dict(model.policy.named_parameters()),
    #     show_attrs=False,
    #     show_saved=False,
    # )
    # dot.format = "png"
    # dot.render("network_architecture2")

    # Observation/Action space info
    print("\nSpaces:")
    print(f"Observation Space: {model.observation_space}")
    print(f"Action Space: {model.action_space}")

    # Get trainable parameters count
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\nTrainable Parameters: {trainable_params:,}")

    # Policy network weights statistics
    # print("\nNetwork Weights Statistics:")
    # for name, param in model.policy.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}:")
    #         print(f"  Shape: {param.shape}")
    #         print(f"  Mean: {param.data.mean().item():.6f}")
    #         print(f"  Std: {param.data.std().item():.6f}")
    #         print(f"  Min: {param.data.min().item():.6f}")
    #         print(f"  Max: {param.data.max().item():.6f}")
