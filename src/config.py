from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    # Demonstration values only. Replace with asset-specific NNPC/NPSC
    # engineering and operating parameters during deployment.
    pressure_lower_psi: float = 40.0
    pressure_upper_psi: float = 60.0
    z_score_alert: float = 3.0
    heartbeat_interval_seconds: int = 5
    heartbeat_timeout_seconds: int = 30

DEMO_CONFIG = PipelineConfig()
