from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the controlled prototype demonstration.

    Pressure limits are demonstration parameters only. A real deployment
    must load the limits defined by the specific asset's engineering and
    operating data.
    """

    # Demonstration engineering envelope.
    pressure_lower_psi: float = 40.0
    pressure_upper_psi: float = 60.0

    # Stable normal reference used by the statistical and AI layers.
    # Keeping the reference fixed prevents live-window composition from
    # making the top indicators appear to change randomly.
    normal_pressure_psi: float = 50.0
    normal_pressure_std_psi: float = 1.3

    # Alert when the current reading is more than 3 reference standard
    # deviations from the normal reference.
    z_score_alert: float = 3.0

    # Isolation Forest is trained once conceptually on a deterministic
    # normal reference population rather than on the rolling live window.
    ai_contamination: float = 0.01
    ai_training_seed: int = 42

    heartbeat_interval_seconds: int = 5
    heartbeat_timeout_seconds: int = 30


DEMO_CONFIG = PipelineConfig()
