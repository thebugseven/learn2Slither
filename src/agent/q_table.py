import random
import numpy as np

from src.env.direction import Direction


class QTableAgent:
    ACTIONS = (
        Direction.UP,
        Direction.LEFT,
        Direction.DOWN,
        Direction.RIGHT,
    )

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = {}

        self.rng = random.Random(seed)

    @staticmethod
    def _state_key(state: np.ndarray) -> tuple:
        bins = (0.15, 0.35, 0.75)

        return tuple(
            0 if value == 0 else
            int(np.searchsorted(bins, value, side="left"))
            for value in state
        )

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        key = self._state_key(state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(self.ACTIONS), dtype=np.float64)

        return self.q_table[key]

    def choose_action(self, state: np.ndarray, training: bool = True) -> Direction:
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(self.ACTIONS)

        q_values = self._get_q_values(state)
        max_q = np.max(q_values)

        best_indices = np.flatnonzero(q_values == max_q)
        index = self.rng.choice(best_indices.tolist())

        return self.ACTIONS[index]

    def learn(
            self,
            state: np.ndarray,
            action: Direction,
            reward: float,
            next_state: np.ndarray | None,
            done: bool
    ) -> None:
        q_values = self._get_q_values(state)

        action_index = self.ACTIONS.index(action)
        current_q = q_values[action_index]

        target = reward

        if not done:
            if next_state is None:
                raise ValueError("next_state cannot be None when done is False.")
            next_q_values = self._get_q_values(next_state)
            target += self.gamma * np.max(next_q_values)

        q_values[action_index] += self.alpha * (target - current_q)

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
