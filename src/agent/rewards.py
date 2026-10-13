from src.env.board import StepEvent


class RewardSystem:
    REWARDS = {
        StepEvent.WALL_COLLISION: -100.0,
        StepEvent.BODY_COLLISION: -100.0,
        StepEvent.GREEN_APPLE: 10.0,
        StepEvent.RED_APPLE: -10.0,
        StepEvent.ZERO_LENGTH: -100.0,
        StepEvent.NONE: -0.05,
    }

    @classmethod
    def calculate(cls, event: StepEvent) -> float:
        return cls.REWARDS[event]