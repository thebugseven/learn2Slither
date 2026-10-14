import numpy as np

from src.agent.q_network import QNetwork


def test_q_network_training():
    np.random.seed(42)

    network = QNetwork(lr=0.001)

    states = np.array([
        [1, 0, 0, 0] + [0] * 12,
        [0, 1, 0, 0] + [0] * 12,
        [0, 0, 1, 0] + [0] * 12,
        [0, 0, 0, 1] + [0] * 12,
    ], dtype=float)

    targets = np.array([
        [10, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 10, 0],
        [0, 0, 0, 10],
    ], dtype=float)

    initial_predictions = network.predict(states)
    initial_loss = np.mean(
        (initial_predictions - targets) ** 2
    )

    for _ in range(2000):
        network.train_batch(states, targets)

    final_predictions = network.predict(states)
    final_loss = np.mean(
        (final_predictions - targets) ** 2
    )

    print("Initial loss:", initial_loss)
    print("Final loss:", final_loss)

    assert final_loss < initial_loss
    assert final_loss < 0.1