import numpy as np
from collections import Counter
from pathlib import Path

from src.interpreter.vision import VisionInterpreter
from src.interpreter.state_encoder import StateEncoder
from src.env.board import Board, StepEvent
from src.agent.rewards import RewardSystem

from .evaluator import evaluate


def train_dqn(
        sessions: int = 1000,
        agent=None,
        max_steps: int = 500,
        evaluation_interval: int = 100,
):
    
    board = Board()
    interpreter = VisionInterpreter(board)
    encoder = StateEncoder()

    if agent is None:
        raise ValueError("An agent must be provided for training.")

    max_steps = 500
    history = []
    best_score = (-1.0, -1.0, -1.0)

    for episode in range(sessions):
        board.reset()

        state = encoder.encode(interpreter.observe())

        total_reward = 0.0
        max_length = board.snake.length
        events = Counter()

        losses = []

        for _ in range(max_steps):
            action = agent.choose_action(state)

            result = board.step(action)
            events[result.event] += 1

            reward = RewardSystem.calculate(result.event)

            next_state = None

            if not result.done:
                next_state = encoder.encode(interpreter.observe())

            agent.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=result.done
            )

            total_reward += reward
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            max_length = max(max_length, result.length)

            if result.done:
                break

            state = next_state

        agent.end_episode()

        episode_number = episode + 1

        if episode_number in (1, 10, 100, 1000):
            agent.save(f"models/dqn_{episode_number}.npz", metadata={"episode": episode_number})

        history.append({
            "reward": total_reward,
            "steps": board.steps,
            "length": max_length,
            "green": events[StepEvent.GREEN_APPLE],
            "red": events[StepEvent.RED_APPLE],
            "loss": float(np.mean(losses)) if losses else 0.0,
            
        })

        if (episode + 1) % 100 == 0:
            recent = history[-100:]

            print(
                f"Episode {episode + 1}/{sessions} | "
                f"Reward: {np.mean([h['reward'] for h in recent]):.2f} | "
                f"Steps: {np.mean([h['steps'] for h in recent]):.1f} | "
                f"Length: {np.mean([h['length'] for h in recent]):.2f} | "
                f"Loss: {np.mean([h['loss'] for h in recent]):.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

            stats = evaluate(agent, sessions=100, verbose=False)

            score = (
                stats["final_sucess_rate"],
                stats["avg_max_length"],
                stats["avg_steps"]
            )

            if score > best_score:
                best_score = score
                agent.save("models/dqn_best.npz", metadata={
                    "episode": episode_number,
                    "score": list(score),
                })
            
        agent.save("models/dqn_last.npz", metadata={
            "episode": episode_number,
            "score": list(score),
        })

    return agent, history
