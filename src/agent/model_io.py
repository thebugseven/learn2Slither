import json
import numpy as np

from pathlib import Path
from .q_network import QNetwork


MODEL_VERSION = 1

ACTIONS = ("UP", "LEFT", "DOWN", "RIGHT")
DIRECTIONS = ("UP", "LEFT", "DOWN", "RIGHT")
SYMBOLS = ("W", "S", "G", "R")


def save_model(
            network: QNetwork,
            path: str,
            metadata: dict | None = None
        ) -> None:
    state = network.get_state()

    info = {
        "version": MODEL_VERSION,
        "architecture": [16, 64, 64, 4],
        "actions": list(ACTIONS),
        "directions": list(DIRECTIONS),
        "symbols": list(SYMBOLS),
        "encoding": "inverse_nearest_distance",
    }

    if metadata:
        if set(metadata) & set(info):
            raise ValueError(
                "Metadata keys conflict with reserved keys in the model info."
                )
        info.update(metadata)

    data = {
        "metadata": np.array(json.dumps(info)),
    }

    for index, (W, b) in enumerate(zip(state["weights"], state["biases"])):
        data[f"W{index}"] = W
        data[f"b{index}"] = b

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(filepath, **data)


def load_model(network: QNetwork, path: str) -> dict:
    filepath = Path(path)

    with np.load(filepath, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))

        expected = {
            "version": MODEL_VERSION,
            "architecture": [16, 64, 64, 4],
            "actions": list(ACTIONS),
            "directions": list(DIRECTIONS),
            "symbols": list(SYMBOLS),
            "encoding": "inverse_nearest_distance",
        }

        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Model {key} mismatch. Expected {value}, "
                    "got {metadata.get(key)}."
                    )

        weights = []
        biases = []

        for index in range(len(network.model.layers)):
            weights.append(data[f"W{index}"].copy())
            biases.append(data[f"b{index}"].copy())

    network.set_state({"weights": weights, "biases": biases})

    return metadata
