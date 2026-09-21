import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import DEMO_CONFIG, PipelineConfig


# 1. Z-Score Detection
def z_score(series):
    mean = np.mean(series)
    std = np.std(series)

    if std == 0:
        return np.zeros_like(series, dtype=float)

    return (series - mean) / std


def detect_anomalies_zscore(
    data,
    threshold=None,
):
    threshold = (
        DEMO_CONFIG.z_score_alert if threshold is None else threshold
    )

    z_scores = z_score(data["pressure"])
    return np.where(np.abs(z_scores) > threshold)[0]


# 2. AI Detection (Isolation Forest)
def detect_anomalies_ai(data):
    model = IsolationForest(contamination=0.05, random_state=42)
    X = data[["pressure"]].values
    model.fit(X)
    preds = model.predict(X)
    return np.where(preds == -1)[0]


# 3. ENGINEERING THRESHOLD DETECTION
def detect_anomalies_threshold(
    data,
    upper=None,
    lower=None,
    config: PipelineConfig = DEMO_CONFIG,
):
    """Detect values outside the configured operating envelope.

    DEMO_CONFIG contains demonstration values only. Real deployment values
    must come from the specific pipeline asset's engineering/operating data.
    """
    upper = config.pressure_upper_psi if upper is None else upper
    lower = config.pressure_lower_psi if lower is None else lower

    pressure = data["pressure"].to_numpy()
    return np.where((pressure > upper) | (pressure < lower))[0]
