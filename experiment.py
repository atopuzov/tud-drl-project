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

    args = [
        sys.executable,  # Path to the Python interpreter
        "learn2.py",
        "--env-name",
        config["env_name"],
        "--num-envs",
        str(config["num_envs"]),
        "--extractor-name",
        config["extractor_name"],
        "--work-dir",
        str(experiment_dir),
    ]

    if config.get("net_arch"):
        args.append("--net-arch")
        args.extend(str(a) for a in config["net_arch"])

    if config.get("frame_stack"):
        args.extend(["--frame-stack", "--frame-stack-size", str(config["frame_stack_size"])])

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
            for process, log_file in processes:
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
