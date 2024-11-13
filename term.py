import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotille
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tfevent_files_as_dataframe(log_dir):
    """
    Reads all tfevents files from a directory and extracts metrics into a single DataFrame.

    Args:
        log_dir (str): Directory containing tfevents files.

    Returns:
        pd.DataFrame: DataFrame containing 'metric_name', 'timestamp', 'step', and 'value'.
    """
    data = []
    log_dir = Path(log_dir)
    event_files = list(log_dir.glob("events.out.tfevents.*"))

    for file_path in event_files:
        try:
            event_acc = EventAccumulator(str(file_path))
            event_acc.Reload()
            available_tags = event_acc.Tags().get("scalars", [])

            for metric in available_tags:
                events = event_acc.Scalars(metric)
                for event in events:
                    # Convert wall_time to datetime
                    timestamp = datetime.datetime.fromtimestamp(event.wall_time)
                    data.append(
                        {"metric_name": metric, "timestamp": timestamp, "step": event.step, "value": event.value}
                    )

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure sorting by metric_name and step for consistent plotting
    df = df.sort_values(by=["metric_name", "step"]).reset_index(drop=True)

    start_time = df["timestamp"].iloc[0]
    df["relative_time"] = (df["timestamp"] - start_time).dt.total_seconds() / 60  # Convert to minutes

    return df


def plot_metric_terminal(df, metric_name, use_time=True, width=80, height=20, last_n_minutes=None):
    """
    Create an ASCII plot of metrics in the terminal.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics
        metric_name (str): The metric to visualize
        use_time (bool): If True, use timestamps on x-axis. If False, use steps
        width (int): Width of the plot in characters
        height (int): Height of the plot in characters
        last_n_minutes (float, optional): If provided, only show last n minutes of data
    """
    # Filter data for the selected metric
    metric_data = df[df["metric_name"] == metric_name]

    # Filter for the last n minutes if requested
    if last_n_minutes is not None:
        max_time = metric_data["relative_time"].max()
        metric_data = metric_data[metric_data["relative_time"] >= (max_time - last_n_minutes)]

    # Get x and y values
    x_values = metric_data["relative_time"] if use_time else metric_data["step"]
    y_values = metric_data["value"]

    if len(y_values) == 0:
        print("No data to plot")
        return

    # Calculate boundaries
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Add small padding to y-axis
    y_padding = (y_max - y_min) * 0.1
    y_min -= y_padding
    y_max += y_padding

    # Create empty plot matrix
    plot = [[" " for _ in range(width)] for _ in range(height)]

    # Plot axis
    for i in range(height):
        plot[i][0] = "│"
    for i in range(width):
        plot[-1][i] = "─"
    plot[-1][0] = "└"

    # Plot data points
    for i in range(len(x_values)):
        # Convert data points to plot coordinates
        x = int((x_values.iloc[i] - x_min) / (x_max - x_min) * (width - 2)) + 1
        y = int((y_values.iloc[i] - y_min) / (y_max - y_min) * (height - 2))
        y = height - 2 - y

        if 0 <= x < width and 0 <= y < height:
            plot[y][x] = "●"

    # Add labels
    title = f" {metric_name.capitalize()} over {'Time' if use_time else 'Steps'} "
    title_pos = (width - len(title)) // 2
    print(" " * title_pos + title)

    # Add y-axis labels
    y_labels = [f"{y_max:.2f}", f"{(y_max + y_min)/2:.2f}", f"{y_min:.2f}"]
    max_label_width = max(len(label) for label in y_labels)

    # Print the plot with y-axis labels
    for i in range(height):
        if i == 0:
            print(f"{y_labels[0]:>{max_label_width}} ", end="")
        elif i == height // 2:
            print(f"{y_labels[1]:>{max_label_width}} ", end="")
        elif i == height - 1:
            print(f"{y_labels[2]:>{max_label_width}} ", end="")
        else:
            print(" " * max_label_width + " ", end="")
        print("".join(plot[i]))

    # Add x-axis labels
    x_labels = [f"{x_min:.1f}", f"{(x_min + x_max)/2:.1f}", f"{x_max:.1f}"]
    label_positions = [1, width // 2, width - 1]
    x_axis_label = " " * (max_label_width + 1)
    for pos, label in zip(label_positions, x_labels):
        space = pos - len(x_axis_label)
        x_axis_label += " " * space + label
    print(x_axis_label)

    # Add x-axis label
    x_label = "Time (minutes)" if use_time else "Steps"
    x_label_pos = (width - len(x_label)) // 2 + max_label_width + 1
    print(" " * x_label_pos + x_label)


def plot_metric_plotille(df, metric_name, use_time=True, width=80, height=20, last_n_minutes=None):
    """
    Visualize a specific metric using plotille for terminal-based plotting.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics
        metric_name (str): The metric to visualize (e.g., 'reward', 'loss')
        use_time (bool): If True, use timestamps on x-axis. If False, use steps
        width (int): Width of the plot in characters
        height (int): Height of the plot in characters
        last_n_minutes (float, optional): If provided, only show last n minutes of data

    Returns:
        str: The plotille figure as a string
    """
    # Filter data for the selected metric
    metric_data = df[df["metric_name"] == metric_name]

    # Filter for the last n minutes if requested
    if last_n_minutes is not None:
        max_time = metric_data["relative_time"].max()
        metric_data = metric_data[metric_data["relative_time"] >= (max_time - last_n_minutes)]

    if len(metric_data) == 0:
        return "No data to plot"

    # Get x and y values
    x_values = metric_data["relative_time"].values if use_time else metric_data["step"].values
    y_values = metric_data["value"].values

    # Create figure
    fig = plotille.Figure()
    fig.width = width
    fig.height = height

    # fig.set_formatter(float, lambda x: '{:.2f}'.format(x))  # Add this line to format all float value
    def _num_formatter(val, chars, delta, left=False):
        align = "<" if left else ""
        return "{:{}{}.2f}".format(int(val), align, chars)

    fig.register_label_formatter(float, _num_formatter)

    # Set title and labels
    title = f"{metric_name.capitalize()} over {'Time' if use_time else 'Steps'}"

    # Configure figure
    fig.set_x_limits(min_=x_values.min(), max_=x_values.max())
    fig.set_y_limits(min_=y_values.min(), max_=y_values.max())
    fig.color_mode = "byte"

    # Add plot with different styles
    fig.plot(
        x_values, y_values, label=metric_name, lc=44, interp="linear"  # Light gray color
    )  # Linear interpolation for smoother lines

    # Return the complete visualization
    return f"{title}\n\n" f"{fig.show(legend=False)}\n"


# df = read_tfevent_files_as_dataframe("logs/DQN_0")
# metric = 'episode/lines_cleared'
# # metric = 'episode/pieces_placed'
# # metric = 'episode/score'
# # metric = 'rollout/ep_rew_mean'
# # metric = 'rollout/ep_len_mean'

# print(plot_metric_plotille(df, metric, use_time=True, width=80, height=20))


def get_available_metrics(df):
    """Get list of all available metrics in the dataframe."""
    return sorted(df["metric_name"].unique())


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from TensorBoard event files")
    parser.add_argument("log_dir", type=str, help="Directory containing TensorBoard event files")
    parser.add_argument("--metric", type=str, help="Metric to plot (if not specified, will show available metrics)")
    parser.add_argument("--width", type=int, default=80, help="Width of the plot")
    parser.add_argument("--height", type=int, default=20, help="Height of the plot")
    parser.add_argument("--use-steps", action="store_true", help="Use steps instead of time on x-axis")
    parser.add_argument("--last-minutes", type=float, help="Only show last N minutes of data")

    args = parser.parse_args()

    # Read the data
    df = read_tfevent_files_as_dataframe(args.log_dir)

    # If no metric specified, show available metrics and exit
    if args.metric is None:
        print("Available metrics:")
        for metric in get_available_metrics(df):
            print(f"  • {metric}")
        return

    # Verify the metric exists
    if args.metric not in df["metric_name"].unique():
        print(f"Error: Metric '{args.metric}' not found. Available metrics:")
        for metric in get_available_metrics(df):
            print(f"  • {metric}")
        return

    # Create and show the plot
    print(
        plot_metric_plotille(
            df,
            args.metric,
            use_time=not args.use_steps,
            width=args.width,
            height=args.height,
            last_n_minutes=args.last_minutes,
        )
    )


if __name__ == "__main__":
    main()
