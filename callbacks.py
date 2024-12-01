from stable_baselines3.common.callbacks import BaseCallback


class EpisodeEndMetricsCallback2(BaseCallback):
    """
    Custom callback for logging episode metrics to TensorBoard during training.
    Tracks metrics separately for each environment in the vectorized setup.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_bumpiness = None  # Will initialize in _on_training_start
        self.current_holes = None  # Will initialize in _on_training_start

    def _on_training_start(self) -> None:
        """Initialize trackers for each environment at start of training."""
        n_envs = len(self.training_env.envs)
        self.current_bumpiness = [[] for _ in range(n_envs)]
        self.current_holes = [[] for _ in range(n_envs)]

    def _on_step(self) -> bool:
        """
        Called at each step of the environment.
        Tracks metrics per environment and logs to TensorBoard at episode end.
        """
        # Track bumpiness and holes for each environment when pieces are placed
        for i, info in enumerate(self.locals["infos"]):
            if info.get("pieces_placed", 0) > len(self.current_bumpiness[i]):
                # A new piece was placed in environment i
                self.current_bumpiness[i].append(info["bumpiness"])
                self.current_holes[i].append(info["holes"])

        # Check if any environment is done (end of episode)
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                score = info.get("score", 0)
                lines_cleared = info.get("lines_cleared", 0)
                pieces_placed = info.get("pieces_placed", 0)
                bumpiness = info.get("bumpiness", 0)
                holes = info.get("holes", 0)

                # Log standard metrics
                self.logger.record_mean("episode/score", score)
                self.logger.record_mean("episode/lines_cleared", lines_cleared)
                self.logger.record_mean("episode/pieces_placed", pieces_placed)
                self.logger.record_mean("episode/bumpiness", bumpiness)
                self.logger.record_mean("episode/holes", holes)

                # Calculate and log average metrics if we have any recorded for this environment
                if self.current_bumpiness[i]:
                    avg_bumpiness = sum(self.current_bumpiness[i]) / len(self.current_bumpiness[i])
                    self.logger.record_mean("episode/avg_bumpiness", avg_bumpiness)
                    max_bumpiness = max(self.current_bumpiness[i])
                    self.logger.record_mean("episode/max_bumpiness", max_bumpiness)

                if self.current_holes[i]:
                    avg_holes = sum(self.current_holes[i]) / len(self.current_holes[i])
                    self.logger.record_mean("episode/avg_holes", avg_holes)
                    max_holes = max(self.current_holes[i])
                    self.logger.record_mean("episode/max_holes", max_holes)

                # Reset trackers for this environment
                self.current_bumpiness[i] = []
                self.current_holes[i] = []

        return True
