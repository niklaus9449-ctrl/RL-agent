"""
grid.py — Core grid engine for the Drone Search & Rescue environment.
Optimized for fast NumPy operations and clean entity management.
"""

import random
import numpy as np


class Grid:
    # Entity codes
    EMPTY    = 0
    DRONE    = 1
    VICTIM   = 2
    OBSTACLE = 3

    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError(f"Grid dimensions must be positive, got ({width}, {height})")

        self.width  = width
        self.height = height
        # NumPy array for O(1) lookups and fast flat copies
        self._grid: np.ndarray = np.zeros((height, width), dtype=np.int8)

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.height and 0 <= y < self.width

    def is_empty(self, x: int, y: int) -> bool:
        return self.is_valid(x, y) and self._grid[x, y] == self.EMPTY

    def get(self, x: int, y: int) -> int | None:
        if not self.is_valid(x, y):
            return None
        return int(self._grid[x, y])

    def place(self, x: int, y: int, entity_id: int) -> bool:
        """Place entity only on empty valid cell. Returns success flag."""
        if not self.is_empty(x, y):
            return False
        self._grid[x, y] = entity_id
        return True

    def force_place(self, x: int, y: int, entity_id: int) -> None:
        """Place entity unconditionally (e.g. moving drone)."""
        if self.is_valid(x, y):
            self._grid[x, y] = entity_id

    def remove(self, x: int, y: int) -> None:
        if self.is_valid(x, y):
            self._grid[x, y] = self.EMPTY

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def random_empty(self) -> tuple[int, int] | None:
        """Return a uniformly random empty cell, or None if grid is full."""
        empties = list(zip(*np.where(self._grid == self.EMPTY)))
        if not empties:
            return None
        idx = random.randrange(len(empties))
        r, c = empties[idx]
        return int(r), int(c)

    def positions_of(self, entity_id: int) -> list[tuple[int, int]]:
        """Return all positions of a given entity type."""
        rows, cols = np.where(self._grid == entity_id)
        return list(zip(rows.tolist(), cols.tolist()))

    def flat_view(self) -> np.ndarray:
        """Return a flat float32 copy — ready for neural network input."""
        return self._grid.flatten().astype(np.float32)

    def reset(self) -> None:
        self._grid[:] = self.EMPTY

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    SYMBOLS = {EMPTY: ".", DRONE: "D", VICTIM: "V", OBSTACLE: "#"}

    def render(self) -> str:
        rows = []
        for row in self._grid:
            rows.append(" ".join(self.SYMBOLS.get(int(c), "?") for c in row))
        return "\n".join(rows)
