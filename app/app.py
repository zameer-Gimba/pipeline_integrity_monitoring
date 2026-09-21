import os
import sys
import time

import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DEMO_CONFIG
from src.detector import (
    detect_anomalies_ai,
    detect_anomalies_threshold,
    detect_anomalies_zscore,
)
from src.heartbeat import HeartbeatMonitor
from src.mqtt_client import MQTTClient
from src.simulator import simulate_pipeline_data
from src.tamper import TamperMonitor

st.set_page_config(page_title="Pipeline Integrity Monitoring", layout="wide")
st.title("Pipeline Integrity Monitoring System")
st.caption(
    "Prototype demonstration. Pressure thresholds are demonstration parameters "
    "and must be replaced by asset-specific NNPC/NPSC engineering data for deployment."
)

if "heartbeat" not in st.session_state:
    st.session_state.heartbeat = HeartbeatMonitor(
        DEMO_CONFIG.heartbeat_timeout_seconds
    )
if "tamper" not in st.session_state:
    st.session_state.tamper = TamperMonitor()
if "mqtt" not in st.session_state:
    st.session_state.mqtt = None
if "latest" not in st.session_state:
    st.session_state.latest = None

with st.sidebar:
    st.header("Prototype Controls")
    mqtt_enabled = st.checkbox("Enable MQTT", value=False)
    broker = st.text_input("MQTT broker", value=os.getenv("MQTT_BROKER", ""))
    port = st.number_input("MQTT port", min_value=1, max_value=65535, value=1883)
    topic = st.text_input(
        "MQTT topic",
        value="pipeline/prototype/telemetry",
    )

    if mqtt_enabled and st.session_state.mqtt is None:
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

    if st.button("Trigger Tamper Event"):
        st.session_state.tamper.set_state(True)

    if st.button("Clear Tamper"):
        st.session_state.tamper.set_state(False)

    simulate_offline = st.checkbox("Simulate node offline", value=False)

data = simulate_pipeline_data(n_points=200)

# Use the last part of the Digital Twin for the dashboard view.
pressure = float(data["pressure"].iloc[-1])
z_anomalies = detect_anomalies_zscore(data, threshold=DEMO_CONFIG.z_score_alert)
ai_anomalies = detect_anomalies_ai(data)
threshold_anomalies = detect_anomalies_threshold(
    data,
    upper=DEMO_CONFIG.pressure_upper_psi,
    lower=DEMO_CONFIG.pressure_lower_psi,
)

if not simulate_offline:
    st.session_state.heartbeat.receive()

if st.session_state.mqtt is not None:
    st.session_state.mqtt.publish(
        {
            "node_id": "DEMO-NODE-01",
            "pressure_psi": pressure,
            "heartbeat": not simulate_offline,
            "tamper": st.session_state.tamper.tamper_detected,
        }
    )

online = st.session_state.heartbeat.is_online()
threshold_alert = (
    pressure < DEMO_CONFIG.pressure_lower_psi
    or pressure > DEMO_CONFIG.pressure_upper_psi
)

cols = st.columns(5)
cols[0].metric("Pressure", f"{pressure:.1f} PSI")
cols[1].metric("Engineering", "ALERT" if threshold_alert else "SAFE")
cols[2].metric(
    "Z-Score",
    "ALERT" if len(z_anomalies) else "NORMAL",
)
cols[3].metric(
    "AI Layer",
    "ANOMALOUS" if len(ai_anomalies) else "NORMAL",
)
cols[4].metric("Node", "ONLINE" if online else "OFFLINE")

st.divider()

status_cols = st.columns(2)
with status_cols[0]:
    if st.session_state.tamper.tamper_detected:
        st.error("TAMPER ALERT — physical interference signal detected.")
    else:
        st.success("TAMPER STATUS — SECURE")

with status_cols[1]:
    if online:
        st.success("HEARTBEAT — NODE ONLINE")
    else:
        st.error(
            "NODE OFFLINE — investigate power, communications, equipment failure "
            "or possible physical interference."
        )

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data["time"], data["pressure"], label="Live Pressure")
ax.axhline(
    DEMO_CONFIG.pressure_upper_psi,
    linestyle="--",
    label=f"Upper demonstration limit ({DEMO_CONFIG.pressure_upper_psi} PSI)",
)
ax.axhline(
    DEMO_CONFIG.pressure_lower_psi,
    linestyle="--",
    label=f"Lower demonstration limit ({DEMO_CONFIG.pressure_lower_psi} PSI)",
)
ax.fill_between(
    data["time"],
    DEMO_CONFIG.pressure_lower_psi,
    DEMO_CONFIG.pressure_upper_psi,
    alpha=0.1,
    label="Demonstration operating envelope",
)

if len(threshold_anomalies):
    ax.scatter(
        data["time"].iloc[threshold_anomalies],
        data["pressure"].iloc[threshold_anomalies],
        label="Engineering anomalies",
        s=45,
        zorder=4,
    )

ax.set_xlabel("Time")
ax.set_ylabel("Pressure (PSI)")
ax.set_title("Pipeline Monitoring — Prototype Digital Twin")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
st.pyplot(fig)

st.subheader("Detection Summary")
st.write(f"Engineering threshold events: {len(threshold_anomalies)}")
st.write(f"Z-score events (|Z| ≥ {DEMO_CONFIG.z_score_alert}): {len(z_anomalies)}")
st.write(f"Isolation Forest events: {len(ai_anomalies)}")

st.info(
    "Prototype scope: Digital Twin + anomaly detection + MQTT-ready telemetry + "
    "heartbeat + tamper monitoring. Additional field capabilities can be added "
    "after site-specific requirements are defined."
)
