import numpy as np

from src.env.direction import Direction
from src.interpreter.vision import RawVision
from src.interpreter.state_encoder import StateEncoder


def test_state_encoder():
    vision = RawVision(
        up=("0", "G", "W"),
        left=("S", "0", "W"),
        down=("0", "0", "R", "W"),
        right=("0", "W"),
    )

    encoder = StateEncoder()
    state = encoder.encode(vision)

    expected = np.array([
        1/3, 0,   1/2, 0,    # UP
        1/3, 1,   0,   0,    # LEFT
        1/4, 0,   0,   1/3,  # DOWN
        1/2, 0,   0,   0,    # RIGHT
    ], dtype=np.float32)

    assert state.shape == (16,)
    assert np.allclose(state, expected)