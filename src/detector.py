import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import DEMO_CONFIG, PipelineConfig


def reference_z_score(value, config: PipelineConfig = DEMO_CONFIG):
    """Calculate a current reading's z-score against a stable normal reference."""
    std = config.normal_pressure_std_psi
    if std <= 0:
        return 0.0
    return (float(value) - config.normal_pressure_psi) / std


def detect_anomalies_zscore(
    data,
    threshold=None,
    config: PipelineConfig = DEMO_CONFIG,
):
    """Detect readings that deviate from the stable normal reference.

    Unlike the old implementation, this does not recalculate mean/std from
    the rolling live window. That prevents the alert boundary from moving
    simply because older abnormal readings entered or left the window.
    """
    threshold = config.z_score_alert if threshold is None else threshold
    values = data["pressure"].to_numpy(dtype=float)
    z_scores = (values - config.normal_pressure_psi) / config.normal_pressure_std_psi
    return np.where(np.abs(z_scores) > threshold)[0]


def build_ai_model(config: PipelineConfig = DEMO_CONFIG):
    """Build a deterministic Isolation Forest from a normal reference population."""
    rng = np.random.default_rng(config.ai_training_seed)
    normal_reference = rng.normal(
        loc=config.normal_pressure_psi,
        scale=config.normal_pressure_std_psi,
        size=(2000, 1),
    )

    model = IsolationForest(
        contamination=config.ai_contamination,
        random_state=config.ai_training_seed,
    )
    model.fit(normal_reference)
    return model


def detect_anomalies_ai(data, config: PipelineConfig = DEMO_CONFIG):
    """Detect deviations using a model trained only on the normal reference."""
    model = build_ai_model(config)
    X = data[["pressure"]].to_numpy(dtype=float)
    preds = model.predict(X)
    return np.where(preds == -1)[0]


def detect_current_ai(value, config: PipelineConfig = DEMO_CONFIG):
    """Return the current AI classification and score for dashboard explanation."""
    model = build_ai_model(config)
    value_array = np.array([[float(value)]])
    prediction = int(model.predict(value_array)[0])
    score = float(model.score_samples(value_array)[0])
    return prediction == -1, score


def detect_anomalies_threshold(
    data,
    upper=None,
    lower=None,
    config: PipelineConfig = DEMO_CONFIG,
):
    """Detect values outside the configured demonstration envelope.

    Real deployment values must come from the specific pipeline asset's
    engineering and operating data.
    """
    upper = config.pressure_upper_psi if upper is None else upper
    lower = config.pressure_lower_psi if lower is None else lower

    pressure = data["pressure"].to_numpy(dtype=float)
    return np.where((pressure > upper) | (pressure < lower))[0]
