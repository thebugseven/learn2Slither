import random
from dataclasses import dataclass
from enum import Enum

from .snake import Snake
from .direction import Direction
from .apple import Apple, AppleType


class StepEvent(Enum):
    NONE = "none"
    GREEN_APPLE = "green_apple"
    RED_APPLE = "red_apple"
    WALL_COLLISION = "wall_collision"
    BODY_COLLISION = "body_collision"
    ZERO_LENGTH = "zero_length"


@dataclass
class StepResult:
    event: StepEvent
    done: bool
    length: int
    steps: int


class Board:
    def __init__(self, width: int = 10, height: int = 10, seed: int | None = None):
        if width < 3 or height < 3:
            raise ValueError("Board dimensions must be at least 3x3.")

        self.width = width
        self.height = height
        self.rng = random.Random(seed)

        self.snake = None
        self.apples = []
        self.game_over = False
        self.steps = 0

        self.reset()

    def reset(self):
        self.game_over = False
        self.steps = 0

        self.snake = self._spawn_snake()
        self.apples = []

        for _ in range(2):
            self._spawn_apple(AppleType.GREEN)

        self._spawn_apple(AppleType.RED)

    def _spawn_snake(self) -> Snake:
        directions = list(Direction)

        while True:
            head = (self.rng.randrange(self.width), self.rng.randrange(self.height))

            direction = self.rng.choice(directions)
            dx, dy = direction.value

            body = [(head[0] - i * dx, head[1] - i * dy) for i in range(3)]

            if all(self._is_inside(pos) for pos in body):
                return Snake(body)

    def _is_inside(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def _occupied_positions(self) -> set:
        occupied = set(self.snake.body)

        for apple in self.apples:
            occupied.add(apple.position)

        return occupied

    def _spawn_apple(self, apple_type: AppleType):
        occupied = self._occupied_positions()

        free_cells = [
            (x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in occupied
        ]

        if not free_cells:
            return False

        position = self.rng.choice(free_cells)

        self.apples.append(Apple(position=position, type=apple_type))
        return True

    def _get_apple(self, position):
        for apple in self.apples:
            if apple.position == position:
                return apple
        return None

    def _is_collision(self, new_head, grow=False):
        if not self._is_inside(new_head):
            return StepEvent.WALL_COLLISION

        body = list(self.snake.body)

        if not grow:
            body = body[:-1]

        if new_head in body:
            return StepEvent.BODY_COLLISION

        return None

    def step(self, direction: Direction) -> StepResult:
        if self.game_over:
            raise RuntimeError("Game is over. Please reset the board.")

        new_head = self.snake.next_head(direction)
        apple = self._get_apple(new_head)

        grow = apple is not None and apple.type == AppleType.GREEN

        collision = self._is_collision(new_head, grow=grow)
        self.steps += 1

        if collision is not None:
            self.game_over = True

            return StepResult(event=collision, done=True, length=self.snake.length, steps=self.steps)

        self.snake.move(new_head, grow=grow)

        event = StepEvent.NONE

        if apple is not None:
            self.apples.remove(apple)

            if apple.type == AppleType.GREEN:
                event = StepEvent.GREEN_APPLE
            elif apple.type == AppleType.RED:
                event = StepEvent.RED_APPLE
                self.snake.shrink()

                if self.snake.length == 0:
                    self.game_over = True
                    return StepResult(event=StepEvent.ZERO_LENGTH, done=True, length=0, steps=self.steps)

            if not self.game_over:
                self._spawn_apple(apple.type)

        return StepResult(event=event, done=self.game_over, length=self.snake.length, steps=self.steps)
