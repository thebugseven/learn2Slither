import numpy as np

from src.env.direction import Direction
from .vision import RawVision


class StateEncoder:
    DIRECTIONS = (
        Direction.UP,
        Direction.LEFT,
        Direction.DOWN,
        Direction.RIGHT,
    )
    SYMBOLS = ("W", "S", "G", "R")

    @staticmethod
    def _nearest_distance(cells: tuple[str, ...], symbol: str) -> float:
        for distance, cell in enumerate(cells, start=1):
            if cell == symbol:
                return 1.0 / distance

        return 0.0

    def encode(self, vision: RawVision) -> np.ndarray:
        features = []

        for direction in self.DIRECTIONS:
            cells = vision.get(direction)

            for symbol in self.SYMBOLS:
                features.append(
                    self._nearest_distance(cells, symbol)
                )

        return np.asarray(features, dtype=np.float32)