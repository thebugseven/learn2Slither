import numpy as np

from src.agent.replay_buffer import ReplayBuffer


def test_replay_buffer():
    buffer = ReplayBuffer(capacity=3, seed=42)

    for i in range(5):
        state = np.full(16, i, dtype=np.float32)

        buffer.push(
            state=state,
            action=i % 4,
            reward=float(i),
            next_state=state + 1,
            done=False,
        )

    assert len(buffer) == 3

    batch = buffer.sample(2)

    assert len(batch) == 2

    for transition in batch:
        assert transition.state.shape == (16,)
        assert transition.next_state.shape == (16,)
        assert transition.reward >= 2