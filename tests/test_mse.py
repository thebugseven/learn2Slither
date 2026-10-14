import numpy as np

from AI_module import MSELoss


def test_mse_forward():
    loss = MSELoss()

    y_pred = np.array([[1.0, 2.0]])
    y_true = np.array([[2.0, 4.0]])

    result = loss.forward(y_pred, y_true)

    assert np.isclose(result, 2.5)


def test_mse_backward():
    loss = MSELoss()

    y_pred = np.array([[1.0, 2.0]])
    y_true = np.array([[2.0, 4.0]])

    loss.forward(y_pred, y_true)
    grad = loss.backward()

    expected = np.array([[-1.0, -2.0]])

    assert np.allclose(grad, expected)