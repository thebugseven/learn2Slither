from dataclasses import dataclass

from src.env.board import Board
from src.env.direction import Direction


@dataclass(frozen=True)
class RawVision:
    up: tuple[str, ...]
    left: tuple[str, ...]
    down: tuple[str, ...]
    right: tuple[str, ...]

    def get(self, direction: Direction) -> tuple[str, ...]:
        return {
            Direction.UP: self.up,
            Direction.LEFT: self.left,
            Direction.DOWN: self.down,
            Direction.RIGHT: self.right,
        }[direction]


class VisionInterpreter:
    def __init__(self, board: Board):
        self.board = board

    def _cell_at(self, x: int, y: int) -> str:
        if not 0 <= x < self.board.width or not 0 <= y < self.board.height:
            return "W"

        if (x, y) == self.board.snake.head:
            return "H"

        if (x, y) in self.board.snake.body:
            return "S"

        for apple in self.board.apples:
            if apple.position == (x, y):
                return "G" if apple.type.value == "green" else "R"

        return "0"

    def _cast_ray(self, direction: Direction) -> tuple[str, ...]:
        x, y = self.board.snake.head
        dx, dy = direction.value

        cells = []
        while True:
            x += dx
            y += dy

            symbol = self._cell_at(x, y)
            cells.append(symbol)

            if symbol == "W":
                break

        return tuple(cells)

    def observe(self) -> RawVision:
        return RawVision(
            up=self._cast_ray(Direction.UP),
            left=self._cast_ray(Direction.LEFT),
            down=self._cast_ray(Direction.DOWN),
            right=self._cast_ray(Direction.RIGHT),
        )

def display_vision(vision: RawVision, action: Direction | None = None) -> None:
    center = len(vision.left)

    for symbol in reversed(vision.up):
        print(" " * center + symbol)

    horizontal = (
        "".join(reversed(vision.left)) + "H" + "".join(vision.right)
    )

    print(horizontal)

    for symbol in vision.down:
        print(" " * center + symbol)

    if action is not None:
        print(f"Action: {action.name}")
