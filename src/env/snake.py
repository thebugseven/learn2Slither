from collections import deque
from .direction import Direction


class Snake:
    def __init__(self, body: list[tuple[int, int]]):
        if not body:
            raise ValueError("Snake body cannot be empty.")

        self.body = deque(body)

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)

    def next_head(self, direction: Direction) -> tuple[int, int]:
        dx, dy = direction.value
        x, y = self.head

        return (x + dx, y + dy)

    def move(self, new_head: tuple[int, int], grow: bool = False):
        self.body.appendleft(new_head)
        
        if not grow:
            self.body.pop()

    def shrink(self):
        if self.body:
            self.body.pop()
