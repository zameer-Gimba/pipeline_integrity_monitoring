"""Scripted demo timeline for recorded walkthroughs.

Real field telemetry is random and can't be trusted to produce a clean
Normal -> Abnormality -> Alert -> ... -> Recovery sequence on cue. For a
recorded demo we want the *opposite* of randomness: a deterministic,
repeatable timeline the presenter can start with one click and narrate
over, knowing exactly what will appear on screen and when.

Each ScenarioStage owns:
- a time window (seconds since "Start Demo" was pressed)
- a pressure target the live reading eases toward during that window
- whether the node is actively reporting (drives heartbeat/offline logic)
- whether the tamper signal should be raised
- a severity used purely for UI coloring (normal/warning/critical)

Nothing here is faked at the detection layer: the Z-score, Isolation
Forest, and threshold detectors in `detector.py` still run for real on
whatever pressure values this module produces. Only the *pressure
generation itself* is scripted, exactly like a real HIL/SCADA test rig
replays a canned scenario into otherwise-live detection logic.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class ScenarioStage:
    key: str
    label: str
    start: float             # seconds since demo start
    end: float                # seconds since demo start
    pressure_target: float    # PSI the reading eases toward in this window
    pressure_noise: float     # stddev of noise layered on top of the ease
    heartbeat_online: bool    # is the node actively reporting right now
    tamper_active: bool       # should the tamper signal be raised
    severity: str             # "normal" | "warning" | "critical"


# Tuned so the whole cycle finishes comfortably inside a 1-3 minute video,
# with "Recovery" holding steady at the end for as long as you keep talking.
SCENARIO: List[ScenarioStage] = [
    ScenarioStage("normal", "Normal Operation",
                  0, 12, pressure_target=50.0, pressure_noise=1.3,
                  heartbeat_online=True, tamper_active=False, severity="normal"),
    ScenarioStage("pressure_abnormality", "Pressure Abnormality Detected",
                  12, 26, pressure_target=34.0, pressure_noise=2.0,
                  heartbeat_online=True, tamper_active=False, severity="warning"),
    ScenarioStage("anomaly_alert", "Anomaly Alert",
                  26, 38, pressure_target=30.0, pressure_noise=2.5,
                  heartbeat_online=True, tamper_active=False, severity="critical"),
    ScenarioStage("heartbeat_interruption", "Node Heartbeat Interruption",
                  38, 50, pressure_target=30.0, pressure_noise=2.5,
                  heartbeat_online=False, tamper_active=False, severity="warning"),
    ScenarioStage("offline_alert", "Node Offline",
                  50, 62, pressure_target=30.0, pressure_noise=2.5,
                  heartbeat_online=False, tamper_active=False, severity="critical"),
    ScenarioStage("tamper_event", "Tamper Event Detected",
                  62, 74, pressure_target=45.0, pressure_noise=2.0,
                  heartbeat_online=True, tamper_active=True, severity="warning"),
    ScenarioStage("tamper_alert", "Tamper Alert",
                  74, 88, pressure_target=45.0, pressure_noise=2.0,
                  heartbeat_online=True, tamper_active=True, severity="critical"),
    ScenarioStage("recovery", "Recovery — Systems Nominal",
                  88, 108, pressure_target=50.0, pressure_noise=1.3,
                  heartbeat_online=True, tamper_active=False, severity="normal"),
]

# A calm "nothing started yet" stage shown before the presenter clicks
# Start Demo, so the dashboard is visibly alive from the moment it loads
# instead of sitting frozen on an empty chart.
IDLE_STAGE = ScenarioStage("idle", "Normal Operation (live, demo not started)",
                            start=0, end=float("inf"),
                            pressure_target=50.0, pressure_noise=1.3,
                            heartbeat_online=True, tamper_active=False,
                            severity="normal")

TOTAL_DURATION = SCENARIO[-1].end

# Heartbeat thresholds used ONLY for the live-demo instance. These are
# intentionally much shorter than a real field deployment's timeout
# (which would be tens of seconds to minutes) so the scripted
# heartbeat_interruption/offline_alert stages above resolve within a
# few seconds on screen instead of requiring a long real-time wait.
LIVE_HEARTBEAT_WARNING_SECONDS = 4.0
LIVE_HEARTBEAT_OFFLINE_SECONDS = 10.0


def stage_at(elapsed_seconds: float) -> ScenarioStage:
    """Return the scenario stage active at `elapsed_seconds` since start.

    Holds on the final stage once the timeline completes, so the demo
    settles into "Recovery" and stays there rather than resetting or
    erroring out while the presenter keeps talking.
    """
    if elapsed_seconds >= TOTAL_DURATION:
        return SCENARIO[-1]
    for stage in SCENARIO:
        if stage.start <= elapsed_seconds < stage.end:
            return stage
    return SCENARIO[0]


def next_pressure(prev_pressure: float, stage: ScenarioStage,
                   rng: np.random.Generator, pull: float = 0.25) -> float:
    """One live tick of pressure: ease toward the stage's target + noise.

    `pull` controls how quickly the reading approaches the target each
    tick (0-1). Produces a smooth, organic-looking line instead of
    jumping instantly to a new value at every stage boundary.
    """
    drift = (stage.pressure_target - prev_pressure) * pull
    noise = rng.normal(0, stage.pressure_noise)
    return prev_pressure + drift + noise
