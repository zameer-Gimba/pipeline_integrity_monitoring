import numpy as np
import pandas as pd

from src.config import DEMO_CONFIG, PipelineConfig


def simulate_pipeline_data(
    n_points=200,
    save_path=None,
    config: PipelineConfig = DEMO_CONFIG,
    leak_start=80,
    leak_end=100,
):
    """Generate Digital Twin pressure data for the prototype.

    The default 40–60 PSI envelope is a demonstration parameter only.
    It must be replaced by asset-specific engineering/operating limits
    before any real deployment.
    """
    rng = np.random.default_rng(42)
    time = np.arange(n_points)

    # Keep normal operation inside the demonstration envelope.
    pressure = rng.normal(loc=50.0, scale=2.0, size=n_points)

    # Simulated leak / pressure-loss event.
    pressure[leak_start:leak_end] = rng.normal(
        loc=32.0,
        scale=2.0,
        size=leak_end - leak_start,
    )

    # Gradual degradation after the simulated event.
    for i in range(120, n_points):
        pressure[i] -= (i - 120) * 0.12

    data = pd.DataFrame({"time": time, "pressure": pressure})

    if save_path:
        data.to_csv(save_path, index=False)

    return data
