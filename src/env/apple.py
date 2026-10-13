from dataclasses import dataclass
from enum import Enum


class AppleType(Enum):
    GREEN = "green"
    RED = "red"


@dataclass
class Apple:
    position: tuple[int, int]
    type: AppleType