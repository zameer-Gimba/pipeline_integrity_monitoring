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
from src.simulator import seed_history
from src.tamper import TamperMonitor
from src.visualization import plot_live_window

BUFFER_SIZE = 90     # rolling window length shown on the chart
TICK_SECONDS = 1.0   # how often a new live reading is generated

st.set_page_config(page_title="Pipeline Integrity Monitoring", layout="wide")
st.title("Pipeline Integrity Monitoring System")
st.caption(
    "Prototype demonstration. Pressure thresholds are demonstration parameters "
    "and must be replaced by asset-specific NNPC/NPSC engineering data for deployment."
)

# ---------------------------------------------------------------------------
# Session state — everything here persists across the autorefresh reruns
# that make the dashboard feel live.
# ---------------------------------------------------------------------------
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

# Tick the whole app once a second even with no user interaction — this is
# what makes the chart scroll and the badges update on their own.
st_autorefresh(interval=int(TICK_SECONDS * 1000), key="tick")

# ---------------------------------------------------------------------------
# Sidebar — scripted demo controls + manual/interactive controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Scripted Demo")
    st.caption(
        f"Runs the full Normal \u2192 Abnormality \u2192 Alert \u2192 Heartbeat Loss "
        f"\u2192 Offline \u2192 Tamper \u2192 Recovery sequence in "
        f"~{int(TOTAL_DURATION)}s, for recording."
    )

    if not st.session_state.demo_running:
        if st.button("\u25b6 Start Scripted Demo", use_container_width=True, type="primary"):
            st.session_state.demo_running = True
            st.session_state.demo_start_ts = time.time()
    else:
        _elapsed_preview = min(
            time.time() - st.session_state.demo_start_ts, TOTAL_DURATION
        )
        st.write(f"Running \u2014 {_elapsed_preview:0.0f}s / {int(TOTAL_DURATION)}s")
        if st.button("\u23f9 Reset Demo", use_container_width=True):
            st.session_state.demo_running = False
            st.session_state.demo_start_ts = None
            st.session_state.history = deque(
                seed_history(BUFFER_SIZE, rng=st.session_state.rng), maxlen=BUFFER_SIZE
            )
            st.session_state.heartbeat = HeartbeatMonitor(LIVE_HEARTBEAT_OFFLINE_SECONDS)
            st.session_state.heartbeat.receive()
            st.session_state.tamper.set_state(False)

    st.divider()
    st.subheader("MQTT (for future Raspberry Pi / hardware integration)")
    manual_disabled = st.session_state.demo_running
    if manual_disabled:
        st.caption("Disabled while the scripted demo is running.")

    mqtt_enabled = st.checkbox("Enable MQTT", value=False, disabled=manual_disabled)
    broker = st.text_input(
        "MQTT broker", value=os.getenv("MQTT_BROKER", ""), disabled=manual_disabled
    )
    port = st.number_input(
        "MQTT port", min_value=1, max_value=65535, value=1883, disabled=manual_disabled
    )
    topic = st.text_input(
        "MQTT topic", value="pipeline/prototype/telemetry", disabled=manual_disabled
    )

    if mqtt_enabled and st.session_state.mqtt is None and not manual_disabled:
        if broker:
            try:
                client = MQTTClient(broker, int(port), topic)
                client.connect()
                st.session_state.mqtt = client
                st.success("MQTT connected")
            except Exception as exc:
                st.error(f"MQTT connection failed: {exc}")
        else:
            st.info("Enter a broker address to enable MQTT.")

    if not mqtt_enabled and st.session_state.mqtt is not None:
        st.session_state.mqtt.disconnect()
        st.session_state.mqtt = None

# ---------------------------------------------------------------------------
# Which scenario stage are we in right now?
# ---------------------------------------------------------------------------
if st.session_state.demo_running:
    elapsed = min(time.time() - st.session_state.demo_start_ts, TOTAL_DURATION)
    stage = stage_at(elapsed)
    effective_online = stage.heartbeat_online
else:
    elapsed = None
    stage = IDLE_STAGE
    effective_online = True

# ---------------------------------------------------------------------------
# Live tick: at most once per TICK_SECONDS, generate/hold a pressure
# reading and update heartbeat/tamper accordingly.
# ---------------------------------------------------------------------------
now = time.time()
if now - st.session_state.last_tick_ts >= TICK_SECONDS:
    st.session_state.last_tick_ts = now
    prev_pressure = st.session_state.history[-1]

    if effective_online:
        new_pressure = next_pressure(prev_pressure, stage, st.session_state.rng)
        st.session_state.heartbeat.receive()
    else:
        # Comms are down — no fresh telemetry arrives, so the trend line
        # holds at the last known value instead of magically continuing.
        new_pressure = prev_pressure

    st.session_state.history.append(new_pressure)

    if st.session_state.demo_running:
        st.session_state.tamper.set_state(stage.tamper_active)
    # else: tamper state is already controlled by the manual buttons above.

# ---------------------------------------------------------------------------
# Build the current window + run detection on it
# ---------------------------------------------------------------------------
history_list = list(st.session_state.history)
window = pd.DataFrame(
    {"time": np.arange(len(history_list)), "pressure": history_list}
)

pressure = float(window["pressure"].iloc[-1])
z_anomalies = detect_anomalies_zscore(window, threshold=DEMO_CONFIG.z_score_alert)
ai_anomalies = detect_anomalies_ai(window)
threshold_anomalies = detect_anomalies_threshold(
    window, upper=DEMO_CONFIG.pressure_upper_psi, lower=DEMO_CONFIG.pressure_lower_psi
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

# Three-tier node status: ONLINE -> DEGRADED (heartbeat interruption
# window) -> OFFLINE. Uses the real HeartbeatMonitor's elapsed-time math,
# just tuned to demo-appropriate thresholds instead of real deployment
# timeouts (which would be far longer).
seconds_since = st.session_state.heartbeat.seconds_since_last_seen()
if seconds_since is None or seconds_since <= LIVE_HEARTBEAT_WARNING_SECONDS:
    node_status = "ONLINE"
elif seconds_since <= LIVE_HEARTBEAT_OFFLINE_SECONDS:
    node_status = "DEGRADED"
else:
    node_status = "OFFLINE"

threshold_alert = (
    pressure < DEMO_CONFIG.pressure_lower_psi
    or pressure > DEMO_CONFIG.pressure_upper_psi
)

# Both detection badges answer the SAME question the Pressure/Engineering
# badges answer — "is the CURRENT reading anomalous?" — not "does an
# anomaly exist anywhere in this window?". That mismatch was the original
# bug where SAFE/ALERT/ANOMALOUS could contradict each other.
current_index = len(window) - 1
z_alert_now = current_index in set(z_anomalies)
ai_alert_now = current_index in set(ai_anomalies)

# ---------------------------------------------------------------------------
# Scenario banner + progress bar
# ---------------------------------------------------------------------------
banner_fn = {"normal": st.success, "warning": st.warning, "critical": st.error}[
    stage.severity
]
banner_fn(f"**{stage.label}**")

if st.session_state.demo_running:
    frac = min(elapsed / TOTAL_DURATION, 1.0)
    st.progress(frac, text=f"{elapsed:0.0f}s / {int(TOTAL_DURATION)}s")

# ---------------------------------------------------------------------------
# Status row
# ---------------------------------------------------------------------------
cols = st.columns(5)
cols[0].metric("Pressure", f"{pressure:.1f} PSI")
cols[1].metric("Engineering", "ALERT" if threshold_alert else "SAFE")
cols[2].metric("Z-Score", "ALERT" if z_alert_now else "NORMAL")
cols[3].metric("AI Layer", "ANOMALOUS" if ai_alert_now else "NORMAL")
cols[4].metric("Node", node_status)

st.caption(
    f"Live status reflects the current reading only ({pressure:.1f} PSI, "
    "most recent tick). Historical events in the displayed window are "
    "listed separately below."
)

st.divider()

status_cols = st.columns(2)
with status_cols[0]:
    if st.session_state.tamper.tamper_detected:
        st.error("TAMPER ALERT \u2014 physical interference signal detected.")
    else:
        st.success("TAMPER STATUS \u2014 SECURE")

with status_cols[1]:
    if node_status == "ONLINE":
        st.success("HEARTBEAT \u2014 NODE ONLINE")
    elif node_status == "DEGRADED":
        st.warning(
            f"HEARTBEAT DEGRADED \u2014 last signal {seconds_since:.0f}s ago. "
            "Investigate before this escalates to an offline alert."
        )
    else:
        st.error(
            "NODE OFFLINE \u2014 investigate power, communications, equipment failure "
            "or possible physical interference."
        )

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
fig = plot_live_window(
    window,
    threshold_anomalies,
    upper=DEMO_CONFIG.pressure_upper_psi,
    lower=DEMO_CONFIG.pressure_lower_psi,
    stale=not effective_online,
    stage_label=stage.label,
)
st.pyplot(fig)

st.subheader("Detection Summary \u2014 historical, across the window shown above")
st.caption(
    "These counts describe the whole rolling buffer plotted on the chart, "
    "not the current instant. Use the badges above for the live/current status."
)
st.write(f"Engineering threshold events: {len(threshold_anomalies)}")
st.write(f"Z-score events (|Z| \u2265 {DEMO_CONFIG.z_score_alert}): {len(z_anomalies)}")
st.write(f"Isolation Forest events: {len(ai_anomalies)}")

st.info(
    "Prototype scope: Digital Twin + anomaly detection + MQTT-ready telemetry + "
    "heartbeat + tamper monitoring, with a scripted demo mode for recorded "
    "walkthroughs. Additional field capabilities can be added after "
    "site-specific requirements are defined."
)
