from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray | None
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100000, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")

        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray | None,
            done: bool
    ) -> None:
        transition = Transition(
            state=np.array(state, dtype=np.float32, copy=True),
            action=int(action),
            reward=float(reward),
            next_state=(
                None if next_state is None
                else np.array(next_state, dtype=np.float32, copy=True)),
            done=bool(done)
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        return self.rng.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
