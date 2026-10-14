import numpy as np
import AI_module as ai


class QNetwork:
    def __init__(self, lr: float = 0.001):
        self.model = ai.Sequential(optimizer="adam", lr=lr)
        self.model.add(
            ai.Dense(
                input_size=16,
                output_size=64,
                weights_init="he",
                activation="ReLu"
            )
        )
        self.model.add(
            ai.Dense(
                input_size=64,
                output_size=64,
                weights_init="he",
                activation="ReLu"
            )
        )
        self.model.add(
            ai.Dense(
                input_size=64,
                output_size=4,
                weights_init="xavier",
                activation=None
            )
        )

        self.loss = ai.HuberLoss(delta=1.0)

    def predict(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)

        return self.model.forward(states)

    def train_batch(self, states: np.ndarray, targets: np.ndarray) -> float:
        prediction = self.model.forward(states)

        loss_value = self.loss.forward(prediction, targets)
        gradients = self.loss.backward()

        self.model.backward(gradients)
        self.model.update()

        return float(loss_value)

    def copy_from(self, other: "QNetwork") -> None:
        if len(self.model.layers) != len(other.model.layers):
            raise ValueError("Network architectures do not match")

        for target_layer, source_layer in zip(
            self.model.layers,
            other.model.layers,
        ):
            if target_layer.W.shape != source_layer.W.shape:
                raise ValueError("Layer dimensions do not match")

            target_layer.W = source_layer.W.copy()
            target_layer.b = source_layer.b.copy()

    def get_state(self) -> dict:
        return {
            "weights": [layer.W.copy() for layer in self.model.layers],
            "biases": [layer.b.copy() for layer in self.model.layers],
        }

    def set_state(self, state: dict) -> None:
        weights = state["weights"]
        biases = state["biases"]

        if (len(weights) != len(self.model.layers)) \
                or (len(biases) != len(self.model.layers)):
            raise ValueError("State does not match the network architecture")

        for layer, W, b in zip(self.model.layers, weights, biases):
            if W.shape != layer.W.shape or b.shape != layer.b.shape:
                raise ValueError("Layer dimensions do not match the state")

            if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
                raise ValueError("Weights and biases must be finite numbers")

        for layer, W, b in zip(self.model.layers, weights, biases):
            layer.W = np.array(W, dtype=float, copy=True)
            layer.b = np.array(b, dtype=float, copy=True)
