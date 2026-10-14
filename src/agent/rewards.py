from src.env.board import StepEvent


class RewardSystem:
    REWARDS = {
        StepEvent.WALL_COLLISION: -10.0,
        StepEvent.BODY_COLLISION: -10.0,
        StepEvent.GREEN_APPLE: 1.0,
        StepEvent.RED_APPLE: -1.0,
        StepEvent.ZERO_LENGTH: -10.0,
        StepEvent.NONE: -0.005,
    }

    @classmethod
    def calculate(cls, event: StepEvent) -> float:
        return cls.REWARDS[event]
