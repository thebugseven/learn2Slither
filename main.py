from src.env.board import Board, StepEvent
from src.interpreter.vision import VisionInterpreter
from src.interpreter.state_encoder import StateEncoder
from src.agent.q_table import QTableAgent
from src.agent.rewards import RewardSystem

from collections import Counter


def evaluate(agent, sessions=100):
    board = Board()
    interpreter = VisionInterpreter(board)
    encoder = StateEncoder()

    total_steps = 0
    total_max_length = 0
    successes = 0

    for _ in range(sessions):
        board.reset()

        state = encoder.encode(interpreter.observe())
        max_length = board.snake.length

        for _ in range(500):
            action = agent.choose_action(
                state,
                training=False,
            )

            result = board.step(action)

            max_length = max(max_length, result.length)

            if result.done:
                break

            state = encoder.encode(interpreter.observe())

        total_steps += board.steps
        total_max_length += max_length

        if max_length >= 10:
            successes += 1

    print(f"Evaluation over {sessions} episodes")
    print(f"Average steps: {total_steps / sessions:.2f}")
    print(
        f"Average max length: "
        f"{total_max_length / sessions:.2f}"
    )
    print(f"Success rate: {100 * successes / sessions:.2f}%")


def train(sessions: int = 1000):
    board = Board()
    interpreter = VisionInterpreter(board)
    encoder = StateEncoder()
    agent = QTableAgent(
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
    )

    max_steps = 500
    history = []

    for episode in range(sessions):
        board.reset()

        state = encoder.encode(interpreter.observe())

        total_reward = 0.0
        max_length = board.snake.length
        events = Counter()

        for _ in range(max_steps):
            action = agent.choose_action(state)

            result = board.step(action)
            events[result.event] += 1

            reward = RewardSystem.calculate(result.event)

            next_state = None

            if not result.done:
                next_state = encoder.encode(interpreter.observe())

            agent.learn(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=result.done
            )

            total_reward += reward
            max_length = max(max_length, board.snake.length)

            if result.done:
                break

            state = next_state

        agent.end_episode()

        history.append({
            "reward": total_reward,
            "steps": board.steps,
            "length": max_length,
            "green": events[StepEvent.GREEN_APPLE],
            "red": events[StepEvent.RED_APPLE],
            "wall": events[StepEvent.WALL_COLLISION],
            "body": events[StepEvent.BODY_COLLISION],
            
        })

        if (episode + 1) % 1000 == 0:
            recent = history[-1000:]

            avg_reward = sum(h["reward"] for h in recent) / len(recent)
            avg_steps = sum(h["steps"] for h in recent) / len(recent)
            avg_length = sum(h["length"] for h in recent) / len(recent)
            green = sum(h["green"] for h in recent)
            red = sum(h["red"] for h in recent)

            print(
                f"Episode {episode + 1}/{sessions} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Steps: {avg_steps:.1f} | "
                f"Avg Max Length: {avg_length:.2f} | "
                f"Green: {green} | "
                f"Red: {red} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Q-States: {len(agent.q_table)}"
            )

            
    return agent, history

if __name__ == "__main__":
    agent, history = train(10000)
    evaluate(agent, sessions=1000)


