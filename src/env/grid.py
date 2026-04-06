import random

class Grid:
    EMPTY = 0

    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive")

        self.width = width
        self.height = height
        self.grid = [[self.EMPTY for _ in range(width)] for _ in range(height)]

    def is_valid_position(self, x, y) -> bool:
        return 0 <= x < self.height and 0 <= y < self.width

    def is_empty(self, x, y) -> bool:
        if not self.is_valid_position(x, y):
            return False
        return self.grid[x][y] == self.EMPTY

    def place_entity(self, x, y, entity_id) -> bool:
        if not self.is_valid_position(x, y):
            return False
        if not self.is_empty(x, y):
            return False

        self.grid[x][y] = entity_id
        return True

    def get_entity(self, x, y):
        if not self.is_valid_position(x, y):
            return None
        return self.grid[x][y]

    def remove_entity(self, x, y):
        if self.is_valid_position(x, y):
            self.grid[x][y] = self.EMPTY

    def random_empty_position(self):
        empty_positions = [
            (x, y)
            for x in range(self.height)
            for y in range(self.width)
            if self.is_empty(x, y)
        ]

        return random.choice(empty_positions) if empty_positions else None
