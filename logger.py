from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log_epoch(
        self,
        curr: int,
        max: int,
        loss: float,
        epsilon: float,
        avg_score: float,
        avg_steps: float,
        avg_reward: float,
        time: float
    ):
        pass
