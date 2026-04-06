"""
Entities used inside the Drone Search & Rescue environment.
These are simple data containers. The environment controls behavior.
"""

from dataclasses import dataclass
# ---------- Base Entity ----------

@dataclass
class Entity:
    x: int
    y: int

# ---------- Drone Entity ----------

class Drone(Entity):
    """
    Drone used to explore the grid.
    The environment controls movement logic.
    """

    def __init__(self, x: int, y: int, battery_life: int = 100):
        super().__init__(x, y)
        self.battery_life = battery_life

    def move(self, dx: int, dy: int):
        """Update drone position (environment validates movement)."""
        if self.battery_life <= 0:
            raise RuntimeError("Drone battery depleted")

        self.x += dx
        self.y += dy
        self.battery_life -= 1

# ---------- Victim Entity ----------

class Victim(Entity):
    """
    Victim located somewhere in the grid.
    """

    def __init__(self, x: int, y: int, health: int = 100):
        super().__init__(x, y)
        self.health = health
        self.rescued = False

    def rescue(self):
        """Mark victim as rescued."""
        self.rescued = True

    def is_rescued(self) -> bool:
        return self.rescued


# ---------- Obstacle Entity (optional but useful later) ----------

class Obstacle(Entity):
    """
    Represents blocked cells in the grid.
    """

    def __init__(self, x: int, y: int):
        super().__init__(x, y)
