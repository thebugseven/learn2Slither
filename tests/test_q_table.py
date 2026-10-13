import numpy as np

from src.agent.q_table import QTableAgent
from src.env.direction import Direction


def test_initial_q_values():
    agent = QTableAgent()

    state = np.zeros(16, dtype=np.float32)
    values = agent._get_q_values(state)

    assert values.shape == (4,)
    assert np.all(values == 0)


def test_terminal_update():
    agent = QTableAgent(alpha=0.5, gamma=0.9)

    state = np.zeros(16, dtype=np.float32)

    agent.learn(
        state=state,
        action=Direction.UP,
        reward=10.0,
        next_state=None,
        done=True,
    )

    values = agent._get_q_values(state)
    assert np.isclose(values[0], 5.0)


def test_choose_best_action():
    agent = QTableAgent(epsilon=0.0)

    state = np.zeros(16, dtype=np.float32)
    values = agent._get_q_values(state)

    values[:] = [1.0, 2.0, 3.0, 10.0]

    action = agent.choose_action(state, training=False)

    assert action == Direction.RIGHT


def test_epsilon_decay():
    agent = QTableAgent(
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.5,
    )

    agent.end_episode()

    assert np.isclose(agent.epsilon, 0.5)

    for _ in range(100):
        agent.end_episode()

    assert np.isclose(agent.epsilon, 0.1)