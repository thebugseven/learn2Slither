import random
import numpy as np

from src.env.direction import Direction
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer
from .model_io import save_model, load_model


class DQNAgent:
    ACTIONS = (
        Direction.UP,
        Direction.LEFT,
        Direction.DOWN,
        Direction.RIGHT,
    )

    def __init__(
            self,
            lr: float = 0.001,
            gamma: float = 0.95,
            epsilon: float = 1.0,
            epsilon_min: float = 0.05,
            epsilon_decay: float = 0.9995,
            batch_size: int = 64,
            buffer_capacity: int = 100000,
            target_update: int = 1000,
            seed: int | None = None,
    ):
        if batch_size <= 0 or target_update <= 0 or buffer_capacity <= 0:
            raise ValueError(
                "Batch size, target update, and buffer capacity"
                "must be positive integers.")

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_network = QNetwork(lr=lr)
        self.target_network = QNetwork(lr=lr)

        self.target_network.copy_from(self.policy_network)

        self.memory = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        self.rng = random.Random(seed)
        self.train_steps = 0

    def choose_action(
                self,
                state: np.ndarray,
                training: bool = True
            ) -> Direction:
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(self.ACTIONS)

        q_values = self.policy_network.predict(state)[0]

        max_q = np.max(q_values)
        best_indices = np.flatnonzero(q_values == max_q)

        index = self.rng.choice(best_indices.tolist())

        return self.ACTIONS[index]

    def remember(
            self,
            state: np.ndarray,
            action: Direction,
            reward: float,
            next_state: np.ndarray | None,
            done: bool
    ) -> None:
        action_index = self.ACTIONS.index(action)

        self.memory.push(
            state=state,
            action=action_index,
            reward=reward,
            next_state=next_state,
            done=done
        )

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states = np.stack([transition.state for transition in batch])

        actions = np.array(
                        [transition.action for transition in batch],
                        dtype=np.int64
                    )
        rewards = np.array(
                        [transition.reward for transition in batch],
                        dtype=np.float64
                    )
        dones = np.array(
                        [transition.done for transition in batch],
                        dtype=np.bool
                    )

        targets = self.policy_network.predict(states).copy()

        future_values = np.zeros(self.batch_size)
        non_terminal_indices = np.flatnonzero(~dones)

        if len(non_terminal_indices) > 0:
            next_states = np.stack(
                [batch[i].next_state for i in non_terminal_indices]
                )
            next_q_values = self.target_network.predict(next_states)
            future_values[non_terminal_indices] = np.max(next_q_values, axis=1)

        target_values = (rewards + self.gamma * future_values) * (~dones)

        targets[np.arange(self.batch_size), actions] = target_values

        loss = self.policy_network.train_batch(states, targets)

        self.train_steps += 1

        if self.train_steps % self.target_update == 0:
            self.target_network.copy_from(self.policy_network)

        return loss

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str, metadata: dict | None = None) -> None:
        save_model(self.policy_network, path, metadata=metadata)

    def load(self, path: str) -> dict | None:
        metadata = load_model(self.policy_network, path)
        self.target_network.copy_from(self.policy_network)

        self.train_steps = 0

        return metadata
