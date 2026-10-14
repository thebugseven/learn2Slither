from collections import Counter

from src.env.board import Board, StepEvent
from src.interpreter.vision import VisionInterpreter
from src.interpreter.state_encoder import StateEncoder


THRESHOLDS = (10, 15, 20, 25, 30, 35)


def evaluate(
        agent,
        sessions=100,
        max_steps=500,
        verbose=True,
        seed=42
):
    if sessions <= 0:
        raise ValueError("sessions must be a positive integer.")

    board = Board(seed=seed)
    interpreter = VisionInterpreter(board)
    encoder = StateEncoder()

    statistics = Counter()
    reached = Counter()
    finished = Counter()

    for _ in range(sessions):
        board.reset()

        state = encoder.encode(interpreter.observe())
        max_length = board.snake.length

        for _ in range(max_steps):
            action = agent.choose_action(
                state,
                training=False,
            )

            result = board.step(action)

            max_length = max(max_length, result.length)

            if result.event == StepEvent.GREEN_APPLE:
                statistics["green"] += 1
            elif result.event == StepEvent.RED_APPLE:
                statistics["red"] += 1

            if result.done:
                break

            state = encoder.encode(interpreter.observe())

        statistics["steps"] += board.steps
        statistics["max_length"] += max_length
        statistics["final_length"] += board.snake.length

        if board.game_over:
            statistics["deaths"] += 1
        else:
            statistics["timeouts"] += 1

        for threshold in THRESHOLDS:
            if max_length >= threshold:
                reached[threshold] += 1
            if board.snake.length >= threshold:
                finished[threshold] += 1

    stats = {
        "avg_steps": statistics["steps"] / sessions,
        "avg_max_length": statistics["max_length"] / sessions,
        "avg_final_length": statistics["final_length"] / sessions,
        "green_per_game": statistics["green"] / sessions,
        "red_per_game": statistics["red"] / sessions,
        "timeout_rate": statistics["timeouts"] / sessions,
        "death_rate": statistics["deaths"] / sessions,
        "reached": {
            n: reached[n] / sessions
            for n in THRESHOLDS
        },
        "finished": {
            n: finished[n] / sessions
            for n in THRESHOLDS
        },
    }

    stats["score"] = (
        stats["finished"][10],
        stats["avg_final_length"],
        stats["avg_steps"],
    )

    if verbose:
        print(f"\nEvaluation over {sessions} episodes")
        print(f"Average steps: {stats['avg_steps']:.2f}")
        print(f"Average max length: {stats['avg_max_length']:.2f}")
        print(f"Average final length: {stats['avg_final_length']:.2f}")
        print(f"Timeout rate: {stats['timeout_rate']:.2f}%")

        for threshold in THRESHOLDS:
            print(
                f"Length >= {threshold}: "
                f"reached {stats['reached'][threshold]:.2f}% | "
                f"final {stats['finished'][threshold]:.2f}%"
            )

    return stats
