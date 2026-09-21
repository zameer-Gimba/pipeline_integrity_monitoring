import numpy as np


def seed_history(n_points=90, start_pressure=50.0, noise=1.3, rng=None):
    """Seed the rolling live-demo buffer with near-normal readings."""
    rng = rng if rng is not None else np.random.default_rng()
    return list(start_pressure + rng.normal(0, noise, size=n_points))
