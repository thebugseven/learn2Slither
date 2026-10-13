from src.env.board import Board, StepEvent
from src.env.direction import Direction
from src.env.snake import Snake
from src.env.apple import Apple, AppleType


def test_initialization():
    board = Board()

    assert board.width == 10
    assert board.height == 10
    assert board.snake.length == 3
    assert len(board.apples) == 3

    assert sum(
        apple.type == AppleType.GREEN
        for apple in board.apples
    ) == 2

    assert sum(
        apple.type == AppleType.RED
        for apple in board.apples
    ) == 1

    positions = list(board.snake.body)
    positions += [apple.position for apple in board.apples]

    assert len(positions) == len(set(positions))


def test_wall_collision():
    board = Board()

    board.snake = Snake([
        (0, 5),
        (1, 5),
        (2, 5),
    ])

    result = board.step(Direction.LEFT)

    assert result.done
    assert result.event == StepEvent.WALL_COLLISION


def test_green_apple():
    board = Board()

    board.snake = Snake([
        (5, 5),
        (4, 5),
        (3, 5),
    ])

    board.apples = [
        Apple((6, 5), AppleType.GREEN)
    ]

    result = board.step(Direction.RIGHT)

    assert result.event == StepEvent.GREEN_APPLE
    assert board.snake.length == 4
    assert not result.done


def test_red_apple():
    board = Board()

    board.snake = Snake([
        (5, 5),
        (4, 5),
        (3, 5),
    ])

    board.apples = [
        Apple((6, 5), AppleType.RED)
    ]

    result = board.step(Direction.RIGHT)

    assert result.event == StepEvent.RED_APPLE
    assert board.snake.length == 2
    assert not result.done


def test_body_collision():
    board = Board()

    board.snake = Snake([
        (5, 5),
        (5, 4),
        (4, 4),
        (4, 5),
        (4, 6),
    ])

    board.apples = []

    result = board.step(Direction.UP)

    assert result.done
    assert result.event == StepEvent.BODY_COLLISION


def test_reset():
    board = Board()

    board.game_over = True
    board.steps = 150

    board.reset()

    assert not board.game_over
    assert board.steps == 0
    assert board.snake.length == 3
    assert len(board.apples) == 3