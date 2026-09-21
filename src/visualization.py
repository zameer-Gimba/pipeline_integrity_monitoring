import matplotlib.pyplot as plt
import pandas as pd


def plot_live_window(data, threshold_anomalies, upper, lower, stale=False,
                      stage_label=""):
    """Build (but do not show) a matplotlib Figure for the live/scripted
    dashboard's rolling window. Returns the Figure so the caller (the
    Streamlit app, or a test script) decides how to display or save it.

    `stale=True` means the node isn't currently reporting fresh telemetry
    (heartbeat interruption / offline) — the tail of the line is drawn
    dashed to visually distinguish "no new data" from a genuine flat
    real reading.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    if stale and len(data) > 1:
        # Draw everything up to the last fresh point solid, then the
        # frozen tail dashed, so a viewer can see comms dropped.
        split = max(len(data) - 1, 1)
        ax.plot(data["time"].iloc[:split + 1], data["pressure"].iloc[:split + 1],
                color="#2c3e50", linewidth=1.6, label="Live Pressure")
        ax.plot(data["time"].iloc[split:], data["pressure"].iloc[split:],
                color="#7f8c8d", linewidth=1.6, linestyle="--",
                label="No new telemetry (last known value)")
    else:
        ax.plot(data["time"], data["pressure"], color="#2c3e50",
                linewidth=1.6, label="Live Pressure")

    ax.axhline(upper, linestyle="--", color="#c0392b",
               label=f"Upper demonstration limit ({upper} PSI)")
    ax.axhline(lower, linestyle="--", color="#c0392b",
               label=f"Lower demonstration limit ({lower} PSI)")
    ax.fill_between(data["time"], lower, upper, alpha=0.1, color="#27ae60",
                     label="Demonstration operating envelope")

    if len(threshold_anomalies):
        ax.scatter(
            data["time"].iloc[threshold_anomalies],
            data["pressure"].iloc[threshold_anomalies],
            label="Engineering anomalies",
            color="#e74c3c",
            edgecolor="black",
            s=45,
            zorder=4,
        )

    ax.set_xlabel("Recent readings (most recent \u2192 right)")
    ax.set_ylabel("Pressure (PSI)")
    title = "Pipeline Monitoring — Live Digital Twin"
    if stage_label:
        title += f"  |  {stage_label}"
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig

def plot_results_final(data, anomalies, threshold_upper=60, threshold_lower=40):
    if data.empty:
        print("Warning: Data is empty.")
        return

    plt.figure(figsize=(12, 6))

    # 1. Plot the Main Pressure Line
    plt.plot(data["time"], data["pressure"], 
             label="Live Pressure", 
             color="#2c3e50", 
             linewidth=1.5, 
             alpha=0.7, 
             zorder=1)

    # 2. Add Threshold Lines (Horizontal)
    plt.axhline(y=threshold_upper, color="#c0392b", linestyle="--", 
                linewidth=1.5, label=f"Upper Limit ({threshold_upper})", zorder=2)
    plt.axhline(y=threshold_lower, color="#c0392b", linestyle="--", 
                linewidth=1.5, label=f"Lower Limit ({threshold_lower})", zorder=2)

    # 3. Fill the "Safe Operating Zone" with a light green tint
    plt.fill_between(data["time"], threshold_lower, threshold_upper, 
                     color="#27ae60", alpha=0.1, label="Safe Operating Range")

    # 4. Highlight Anomalies
    if anomalies is not None and len(anomalies) > 0:
        plt.scatter(
            data["time"].iloc[anomalies],
            data["pressure"].iloc[anomalies],
            label="Anomalies Detected",
            color="#e74c3c",
            edgecolor="black",
            s=80,
            zorder=4
        )

    # Professional Formatting
    plt.gcf().autofmt_xdate() # Auto-rotate date labels
    plt.xlabel("Timestamp")
    plt.ylabel("Pressure (PSI)")
    plt.title("Pipeline Monitoring: Pressure & Safety Thresholds", fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Place legend outside or in a clear corner
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()
