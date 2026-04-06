from src.env.grid import Grid
import random

class DroneEnv:
    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1)    # right
    }

    def __init__(self, width=5, height=5, num_victims=3, max_steps=50):

        self.width = width
        self.height = height
        self.num_victims = num_victims
        self.max_steps = max_steps

        self.grid = Grid(width, height)

        self.drone_pos = None
        self.steps = 0
        self.victims_remaining = 0

    # ---------------------------
    # Reset Environment
    # ---------------------------
    def reset(self):

        self.grid = Grid(self.width, self.height)

        self.steps = 0
        self.victims_remaining = self.num_victims

        # place drone
        self.drone_pos = self.grid.random_empty_position()
        x, y = self.drone_pos
        self.grid.place_entity(x, y, 1)

        # place victims
        for _ in range(self.num_victims):
            victim_pos = self.grid.random_empty_position()
            if victim_pos:
                vx, vy = victim_pos
                self.grid.place_entity(vx, vy, 2)

        return self._get_state()

    # ---------------------------
    # Step function
    # ---------------------------

    def step(self, action):

        dx, dy = self.ACTIONS[action]

        x, y = self.drone_pos
        new_x = x + dx
        new_y = y + dy

        reward = -1
        done = False

        if not self.grid.is_valid_position(new_x, new_y):
            reward = -5
            return self._get_state(), reward, done, {}

        # remove drone from current cell
        self.grid.remove_entity(x, y)

        entity = self.grid.get_entity(new_x, new_y)

        # victim found
        if entity == 2:
            reward = 20
            self.victims_remaining -= 1
            self.grid.remove_entity(new_x, new_y)

        # move drone
        self.drone_pos = (new_x, new_y)
        self.grid.place_entity(new_x, new_y, 1)

        self.steps += 1

        # termination condition
        if self.victims_remaining == 0 or self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}

    # ---------------------------
    # Get state
    # ---------------------------
    def _get_state(self):

        return {
            "drone_position": self.drone_pos,
            "grid": self.grid.grid
        }

    # ---------------------------
    # Render (debugging)
    # ---------------------------
    def render(self):
      
        symbols = {
            0: ".",
            1: "D",
            2: "V"
        }

        for row in self.grid.grid:
            print(" ".join(symbols[cell] for cell in row))

        print()
