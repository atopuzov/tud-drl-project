import subprocess
import time
from pathlib import Path
import sys
import yaml
import argparse


def run_experiment(config: dict):
    """
    Run an experiment based on the configuration provided in a dictionary.

    Args:
        config (dict): Configuration dictionary.
    """
    experiment_dir = Path(config["work_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log_file = experiment_dir / "experiment.log"

    # Base command
    args = [
        sys.executable,  # Path to the Python interpreter
        "learn2.py",
    ]

    # Model parameters
    param_map = {
        "env_name": "--env-name",
        "num_envs": "--num-envs",
        "extractor_name": "--extractor-name",
        "gamma": "--gamma",
        "buffer_size": "--buffer-size",
        "batch_size": "--batch-size",
        "learning_starts": "--learning-starts",
        "target_update_interval": "--target-update-interval",
        "learning_rate": "--learning-rate",
        "exploration_fraction": "--exploration-fraction",
        "exploration_initial_eps": "--exploration-initial-eps",
        "exploration_final_eps": "--exploration-final-eps",
        "extractor_features": "--extractor-features",
        "train_freq": "--train-freq",
        "gradient_steps": "--gradient-steps",
        "timestamps": "--timestamps",
        "random_seed": "--random-seed",
        "device": "--device",
        "frame_stack_size": "--frame-stack-size",
        "checkpoint_interval": "--checkpoint-interval",
        "checkpoint_prefix": "--checkpoint-prefix",
        "log_name": "--log-name",
    }

    # Add parameters if they exist in config
    for param, flag in param_map.items():
        if param in config:
            args.extend([flag, str(config[param])])

    # Special handling for paths
    path_map = {
        "tensorboard_log": "--tensorboard-log",
        "checkpoint_path": "--checkpoint-path",
        "best_model_path": "--best-model-path",
        "model_file": "--model-file",
        "replay_buffer_file": "--replay-buffer-file",
    }
    for param, flag in path_map.items():
        if param in config:
            args.extend([flag, str(Path(config[param]))])

    # Handle net_arch (list parameter)
    if "net_arch" in config:
        args.append("--net-arch")
        args.extend(str(a) for a in config["net_arch"])

    # Handle tetrominoes (list parameter)
    if "tetrominoes" in config:
        args.append("--tetrominoes")
        args.extend(str(t) for t in config["tetrominoes"])

    # Boolean flags
    bool_flags = {
        "subproc": "--subproc",
        "frame_stack": "--frame-stack",
        "quiet": "--quiet",
        "checkpoint": "--checkpoint",
        "save_replay": "--save-replay",
        "load_replay": "--load-replay",
        "new": "--new",
        "continue": "--continue",
    }
    for param, flag in bool_flags.items():
        if param in config and config[param]:
            args.append(flag)

    # Always add work directory
    args.extend(["--work-dir", str(experiment_dir)])

    print(f"Running experiment with command: {' '.join(args)}")
    with open(log_file, "w", encoding="utf-8") as log:
        process = subprocess.Popen(args, stdout=log, stderr=log)
        return process, log_file


def monitor_processes(processes: list[tuple[subprocess.Popen, Path]]) -> None:
    """
    Monitor the status of multiple subprocesses.

    Args:
        processes (list): List of tuples containing subprocess and log file path.
    """
    try:
        while processes:
            for process, log_file in processes[:]:  # Create a copy of the list for iteration
                retcode = process.poll()
                if retcode is not None:  # Process finished
                    print(f"Process {process.pid} finished with return code {retcode}. Log: {log_file}")
                    processes.remove((process, log_file))
                else:
                    print(f"Process {process.pid} is still running. Log: {log_file}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Terminating all processes...")
        for process, _ in processes:
            process.terminate()
        for process, _ in processes:
            process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple experiments from a configuration file.")
    parser.add_argument(
        "--config-file",
        type=str,
        default="experiments.yaml",
        help="Path to the YAML file with multiple experiments (default: experiments.yaml)",
    )
    args = parser.parse_args()

    config_file = Path(args.config_file)
    with open(config_file, "r", encoding="utf-8") as file:
        configs = yaml.safe_load(file)

    processes = [run_experiment(config) for config in configs["experiments"]]
    monitor_processes(processes)
