"""
entities.py — Typed data containers for the Drone S&R environment.
Pure data classes; all movement/collision logic lives in DroneEnv.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    x: int
    y: int

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------

@dataclass
class Drone(Entity):
    battery: int = 100
    rescues: int = 0

    def consume(self) -> None:
        """Drain one battery unit per step."""
        if self.battery <= 0:
            raise RuntimeError("Drone battery is depleted — episode should have ended.")
        self.battery -= 1

    @property
    def is_alive(self) -> bool:
        return self.battery > 0


# ---------------------------------------------------------------------------
# Victim
# ---------------------------------------------------------------------------

@dataclass
class Victim(Entity):
    health: int   = 100
    rescued: bool = False

    def rescue(self) -> None:
        self.rescued = True

    @property
    def is_rescued(self) -> bool:
        return self.rescued


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------

@dataclass
class Obstacle(Entity):
    """Impassable static cell."""
    pass
