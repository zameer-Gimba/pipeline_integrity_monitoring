import numpy as np
import pandas as pd

from src.config import DEMO_CONFIG, PipelineConfig
from src.live_history import seed_history


def simulate_pipeline_data(
    n_points=200,
    save_path=None,
    config: PipelineConfig = DEMO_CONFIG,
    leak_start=80,
    leak_end=100,
):
    """Generate Digital Twin pressure data for the prototype.

    The default pressure envelope is a demonstration parameter only.
    It must be replaced by asset-specific engineering/operating limits
    before any real deployment.
    """
    rng = np.random.default_rng(42)
    time = np.arange(n_points)

    pressure = rng.normal(
        loc=config.normal_pressure_psi,
        scale=config.normal_pressure_std_psi,
        size=n_points,
    )

    pressure[leak_start:leak_end] = rng.normal(
        loc=32.0,
        scale=2.0,
        size=leak_end - leak_start,
    )

    for i in range(120, n_points):
        pressure[i] -= (i - 120) * 0.12

    data = pd.DataFrame({"time": time, "pressure": pressure})

    if save_path:
        data.to_csv(save_path, index=False)

    return data
