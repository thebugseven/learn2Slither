import numpy as np
import pytest

from src.agent.dqn_agent import DQNAgent


def test_save_load(tmp_path):
    agent = DQNAgent()

    rng = np.random.default_rng(42)
    state = rng.random(16, dtype=np.float32)

    original_q_values = agent.policy_network.predict(
        state
    ).copy()

    path = tmp_path / "test_model.npz"

    agent.save(str(path))

    restored_agent = DQNAgent()
    metadata = restored_agent.load(str(path))

    restored_q_values = (
        restored_agent.policy_network.predict(state)
    )

    assert np.array_equal(
        original_q_values,
        restored_q_values,
    )

    assert metadata["version"] == 1
    assert metadata["architecture"] == [16, 64, 64, 4]


def test_model_is_independent(tmp_path):
    agent = DQNAgent()

    path = tmp_path / "model.npz"
    agent.save(str(path))

    restored = DQNAgent()
    restored.load(str(path))

    original_weights = (
        agent.policy_network.model.layers[0].W.copy()
    )

    restored.policy_network.model.layers[0].W += 1.0

    assert np.array_equal(
        agent.policy_network.model.layers[0].W,
        original_weights,
    )


def test_incompatible_model(tmp_path):
    agent = DQNAgent()
    path = tmp_path / "invalid.npz"

    np.savez(
        path,
        metadata=np.array('{"version": 999}')
    )

    with pytest.raises(ValueError):
        agent.load(str(path))