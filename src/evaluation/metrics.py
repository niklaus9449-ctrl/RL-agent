# metrics.py - Defines evaluation metrics for the drone environment.
def victims_found(env):
    return getattr(env.drone, "found_victims", 0)
def steps_taken(env):
    return getattr(env, "steps", 0)

def steps_taken(env):
    return getattr(env, "steps", 0)

def exploration_efficiency(env):
    # assumes env.grid.visited is a 2D array of 0/1
    visited = getattr(env.grid, "visited", None)

    if visited is None:
        return 0.0

    explored = visited.sum()
    total = visited.size

    return explored / total if total > 0 else 0.0
