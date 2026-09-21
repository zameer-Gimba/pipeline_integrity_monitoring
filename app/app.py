import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DEMO_CONFIG
from src.detector import (
    detect_anomalies_ai,
    detect_anomalies_threshold,
    detect_anomalies_zscore,
    detect_current_ai,
    reference_z_score,
)
from src.heartbeat import HeartbeatMonitor
from src.mqtt_client import MQTTClient
from src.scenario import (
    IDLE_STAGE,
    LIVE_HEARTBEAT_OFFLINE_SECONDS,
    LIVE_HEARTBEAT_WARNING_SECONDS,
    TOTAL_DURATION,
    next_pressure,
    stage_at,
)
from src.live_history import seed_history
from src.tamper import TamperMonitor
from src.visualization import plot_live_window

BUFFER_SIZE = 90
TICK_SECONDS = 1.0

st.set_page_config(page_title="Pipeline Integrity Monitoring", layout="wide")
st.title("Pipeline Integrity Monitoring System")
st.caption(
    "Prototype demonstration. Pressure thresholds shown here are demonstration "
    "parameters and must be replaced by the specific asset's engineering and "
    "operating data for deployment."
)

if "rng" not in st.session_state:
    st.session_state.rng = np.random.default_rng()
if "history" not in st.session_state:
    st.session_state.history = deque(
        seed_history(BUFFER_SIZE, rng=st.session_state.rng), maxlen=BUFFER_SIZE
    )
if "heartbeat" not in st.session_state:
    st.session_state.heartbeat = HeartbeatMonitor(LIVE_HEARTBEAT_OFFLINE_SECONDS)
    st.session_state.heartbeat.receive()
if "tamper" not in st.session_state:
    st.session_state.tamper = TamperMonitor()
if "mqtt" not in st.session_state:
    st.session_state.mqtt = None
if "demo_running" not in st.session_state:
    st.session_state.demo_running = False
if "demo_start_ts" not in st.session_state:
    st.session_state.demo_start_ts = None
if "last_tick_ts" not in st.session_state:
    st.session_state.last_tick_ts = 0.0

st_autorefresh(interval=int(TICK_SECONDS * 1000), key="tick")

with st.sidebar:
    st.header("Scripted Demo")
    st.caption(
        f"Runs the full Normal → Abnormality → Alert → Heartbeat Loss "
        f"→ Offline → Tamper → Recovery sequence in ~{int(TOTAL_DURATION)}s."
    )

    if not st.session_state.demo_running:
        if st.button(
            "▶ Start Scripted Demo",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.demo_running = True
            st.session_state.demo_start_ts = time.time()
    else:
        _elapsed_preview = min(
            time.time() - st.session_state.demo_start_ts, TOTAL_DURATION
        )
        st.write(f"Running — {_elapsed_preview:0.0f}s / {int(TOTAL_DURATION)}s")
        if st.button("⏹ Reset Demo", use_container_width=True):
            st.session_state.demo_running = False
            st.session_state.demo_start_ts = None
            st.session_state.history = deque(
                seed_history(BUFFER_SIZE, rng=st.session_state.rng),
                maxlen=BUFFER_SIZE,
            )
            st.session_state.heartbeat = HeartbeatMonitor(
                LIVE_HEARTBEAT_OFFLINE_SECONDS
            )
            st.session_state.heartbeat.receive()
            st.session_state.tamper.set_state(False)

    st.caption(
        "The walkthrough is driven by a deterministic test scenario. "
        "MQTT remains available through the application integration layer "
        "for external telemetry when configured."
    )

if st.session_state.demo_running:
    elapsed = min(time.time() - st.session_state.demo_start_ts, TOTAL_DURATION)
    stage = stage_at(elapsed)
    effective_online = stage.heartbeat_online
else:
    elapsed = None
    stage = IDLE_STAGE
    effective_online = True

if st.session_state.mqtt is None and os.getenv("MQTT_BROKER"):
    try:
        st.session_state.mqtt = MQTTClient(
            os.getenv("MQTT_BROKER"),
            int(os.getenv("MQTT_PORT", "1883")),
            os.getenv("MQTT_TOPIC", "pipeline/prototype/telemetry"),
        )
        st.session_state.mqtt.connect()
    except Exception:
        st.session_state.mqtt = None

now = time.time()
if now - st.session_state.last_tick_ts >= TICK_SECONDS:
    st.session_state.last_tick_ts = now
    prev_pressure = st.session_state.history[-1]

    if effective_online:
        new_pressure = next_pressure(prev_pressure, stage, st.session_state.rng)
        st.session_state.heartbeat.receive()
    else:
        new_pressure = prev_pressure

    st.session_state.history.append(new_pressure)

    if st.session_state.demo_running:
        st.session_state.tamper.set_state(stage.tamper_active)

history_list = list(st.session_state.history)
window = pd.DataFrame(
    {"time": np.arange(len(history_list)), "pressure": history_list}
)

pressure = float(window["pressure"].iloc[-1])

# Every status below is evaluated from the same current reading/reference
# system. Historical counts are kept separate and never drive the badges.
z_anomalies = detect_anomalies_zscore(window, config=DEMO_CONFIG)
ai_anomalies = detect_anomalies_ai(window, config=DEMO_CONFIG)
threshold_anomalies = detect_anomalies_threshold(
    window,
    upper=DEMO_CONFIG.pressure_upper_psi,
    lower=DEMO_CONFIG.pressure_lower_psi,
    config=DEMO_CONFIG,
)

z_score_now = reference_z_score(pressure, DEMO_CONFIG)
ai_alert_now, ai_score_now = detect_current_ai(pressure, DEMO_CONFIG)
threshold_alert = (
    pressure < DEMO_CONFIG.pressure_lower_psi
    or pressure > DEMO_CONFIG.pressure_upper_psi
)

if st.session_state.mqtt is not None:
    st.session_state.mqtt.publish(
        {
            "node_id": "DEMO-NODE-01",
            "pressure_psi": pressure,
            "heartbeat": effective_online,
            "tamper": st.session_state.tamper.tamper_detected,
        }
    )

seconds_since = st.session_state.heartbeat.seconds_since_last_seen()
if seconds_since is None or seconds_since <= LIVE_HEARTBEAT_WARNING_SECONDS:
    node_status = "ONLINE"
elif seconds_since <= LIVE_HEARTBEAT_OFFLINE_SECONDS:
    node_status = "DEGRADED"
else:
    node_status = "OFFLINE"

z_alert_now = abs(z_score_now) > DEMO_CONFIG.z_score_alert

# Transparent interpretation of disagreements between layers. An AI-only
# signal is intentionally not relabeled as an engineering failure; it is
# surfaced so the operator can investigate a possible statistical false
# positive or an issue that engineering limits have not captured.
if threshold_alert and z_alert_now and ai_alert_now:
    layer_interpretation = "All detection layers agree: current reading is anomalous."
    layer_interpretation_type = "error"
elif threshold_alert and not ai_alert_now:
    layer_interpretation = (
        "Engineering alert: the reading is outside the configured envelope, "
        "while the AI layer does not currently classify it as anomalous."
    )
    layer_interpretation_type = "warning"
elif ai_alert_now and not threshold_alert and not z_alert_now:
    layer_interpretation = (
        "AI-only signal: engineering limits and Z-score remain normal. "
        "This may represent a statistical false positive or an emerging pattern "
        "not captured by the engineering envelope."
    )
    layer_interpretation_type = "warning"
elif ai_alert_now and not threshold_alert:
    layer_interpretation = (
        "AI signal detected while the reading remains inside the engineering "
        "envelope; investigate the statistical deviation."
    )
    layer_interpretation_type = "warning"
else:
    layer_interpretation = "Detection layers agree: current reading is within normal conditions."
    layer_interpretation_type = "success"

banner_fn = {"normal": st.success, "warning": st.warning, "critical": st.error}[
    stage.severity
]
banner_fn(f"**{stage.label}**")

if st.session_state.demo_running:
    frac = min(elapsed / TOTAL_DURATION, 1.0)
    st.progress(frac, text=f"{elapsed:0.0f}s / {int(TOTAL_DURATION)}s")

cols = st.columns(5)
cols[0].metric("Pressure", f"{pressure:.1f} PSI")
cols[1].metric("Engineering", "ALERT" if threshold_alert else "SAFE")
cols[2].metric("Z-Score", "ALERT" if z_alert_now else "NORMAL")
cols[3].metric("AI Layer", "ANOMALOUS" if ai_alert_now else "NORMAL")
cols[4].metric("Node", node_status)

st.caption(
    f"All live indicators are evaluated from the current reading ({pressure:.1f} PSI). "
    f"Z-score = {z_score_now:+.2f}; AI score = {ai_score_now:+.3f}. "
    "Historical event counts are shown separately."
)

if layer_interpretation_type == "error":
    st.error(layer_interpretation)
elif layer_interpretation_type == "warning":
    st.warning(layer_interpretation)
else:
    st.success(layer_interpretation)

st.divider()

status_cols = st.columns(2)
with status_cols[0]:
    if st.session_state.tamper.tamper_detected:
        st.error("TAMPER ALERT — physical interference signal detected.")
    else:
        st.success("TAMPER STATUS — SECURE")

with status_cols[1]:
    if node_status == "ONLINE":
        st.success("HEARTBEAT — NODE ONLINE")
    elif node_status == "DEGRADED":
        st.warning(
            f"HEARTBEAT DEGRADED — last signal {seconds_since:.0f}s ago. "
            "Investigate before this escalates to an offline alert."
        )
    else:
        st.error(
            "NODE OFFLINE — investigate power, communications, equipment failure "
            "or possible physical interference."
        )

fig = plot_live_window(
    window,
    threshold_anomalies,
    upper=DEMO_CONFIG.pressure_upper_psi,
    lower=DEMO_CONFIG.pressure_lower_psi,
    stale=not effective_online,
    stage_label=stage.label,
)
st.pyplot(fig)

st.subheader("Detection Summary — historical, across the window shown above")
st.caption(
    "These counts describe the whole rolling buffer plotted on the chart, "
    "not the current instant. Use the badges above for the live/current status."
)
st.write(f"Engineering threshold events: {len(threshold_anomalies)}")
st.write(
    f"Z-score events (|Z| > {DEMO_CONFIG.z_score_alert}): {len(z_anomalies)}"
)
st.write(f"Isolation Forest events: {len(ai_anomalies)}")
