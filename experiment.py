import subprocess
import time
from pathlib import Path
import sys
from typing import Tuple
import yaml
import argparse
import signal


def run_experiment(config: dict) -> Tuple[subprocess.Popen, Path]:
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
        "piece_gen": "--piece-gen",
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
    with open(log_file, "a", encoding="utf-8") as log:
        process = subprocess.Popen(args, stdout=log, stderr=log)
        return process, log_file


def monitor_processes(processes: list[tuple[subprocess.Popen, Path]], max_restarts: int = 10, restart_delay: float = 30.0, min_runtime: float = 60.0) -> None:
    """
    Monitor the status of multiple subprocesses. Reruns failed processes with safeguards.

    Args:
        processes (list): List of tuples containing subprocess Popen object and log file path
        max_restarts (int): Maximum number of restart attempts per process
        restart_delay (float): Delay in seconds between restart attempts
        min_runtime (float): Minimum runtime in seconds before reset restart counter
    """
    # Track restart attempts and start times for each process
    process_stats = {p.pid: {"restarts": 0, "last_start": time.time()} for p, _ in processes}

    try:
        while processes:
            for proc_tuple in processes[:]:  # Use a different name to avoid confusion
                process, log_file = proc_tuple
                retcode = process.poll()
                current_time = time.time()

                if retcode is not None:  # Process finished
                    runtime = current_time - process_stats[process.pid]["last_start"]

                    if retcode == 0:  # Clean exit
                        print(f"Process {process.pid} finished successfully. Log: {log_file}")
                        processes.remove(proc_tuple)
                        process_stats.pop(process.pid)
                    else:  # Failed exit
                        # Reset restart counter if process ran for minimum time
                        if runtime > min_runtime:
                            process_stats[process.pid]["restarts"] = 0

                        # Check if we should restart
                        if process_stats[process.pid]["restarts"] < max_restarts:
                            print(f"Process {process.pid} failed with return code {retcode}. "
                                  f"Restart attempt {process_stats[process.pid]['restarts'] + 1}/{max_restarts} "
                                  f"after {restart_delay} seconds delay...")

                            # Remove old process first
                            processes.remove(proc_tuple)
                            process_stats.pop(process.pid)

                            # Wait before restarting
                            time.sleep(restart_delay)

                            # Start new process
                            new_process = subprocess.Popen(
                                process.args,
                                stdout=open(log_file, "a", encoding="utf-8"),
                                stderr=subprocess.STDOUT
                            )

                            # Add new process tracking info
                            processes.append((new_process, log_file))
                            process_stats[new_process.pid] = {
                                "restarts": process_stats.get(process.pid, {}).get("restarts", 0) + 1,
                                "last_start": time.time()
                            }
                        else:
                            print(f"Process {process.pid} failed with return code {retcode}. "
                                  f"Max restarts ({max_restarts}) reached. Giving up.")
                            processes.remove(proc_tuple)
                            process_stats.pop(process.pid)
                else:
                    print(f"Process {process.pid} is still running. Log: {log_file}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nAttempting graceful shutdown of processes...")
        # Send SIGINT to all processes first
        for process, _ in processes:
            if sys.platform.startswith("win"):
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)

        # Then wait for each process with timeout
        for process, log_file in processes:
            try:
                process.wait(timeout=30)  # Wait up to 30 seconds for graceful shutdown
                print(f"Process {process.pid} shut down gracefully. Log: {log_file}")
            except subprocess.TimeoutExpired:
                print(f"Process {process.pid} did not shut down gracefully. Terminating...")
                process.terminate()
                process.wait()  # Wait for forced termination
                print(f"Process {process.pid} was forcefully terminated. Log: {log_file}")



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
