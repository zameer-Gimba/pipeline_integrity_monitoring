from src.config import DEMO_CONFIG
from src.detector import (
    detect_anomalies_ai,
    detect_anomalies_threshold,
    detect_anomalies_zscore,
)
from src.simulator import simulate_pipeline_data
from src.visualization import plot_results_final


def main():
    data = simulate_pipeline_data(
        n_points=200,
        save_path="data/simulated_pipeline_data.csv",
    )

    anomalies = detect_anomalies_threshold(data, config=DEMO_CONFIG)

    plot_results_final(
        data,
        anomalies,
        threshold_upper=DEMO_CONFIG.pressure_upper_psi,
        threshold_lower=DEMO_CONFIG.pressure_lower_psi,
    )

    print(f"Engineering anomalies detected: {len(anomalies)}")
    print(
        f"Z-score anomalies detected: "
        f"{len(detect_anomalies_zscore(data, threshold=DEMO_CONFIG.z_score_alert))}"
    )
    print(f"AI anomalies detected: {len(detect_anomalies_ai(data))}")


if __name__ == "__main__":
    main()
