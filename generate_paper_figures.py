from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Figure 4: model accuracy across datasets
# -----------------------------

accuracy_data = pd.DataFrame({
    "Model": [
        "Isolation Forest", "Autoencoder", "LSTM Autoencoder", "XGBoost",
        "Isolation Forest", "Autoencoder", "LSTM Autoencoder", "XGBoost",
        "Isolation Forest", "Autoencoder", "LSTM Autoencoder", "XGBoost",
    ],
    "Dataset": [
        "Prometheus", "Prometheus", "Prometheus", "Prometheus",
        "LO2", "LO2", "LO2", "LO2",
        "RS-Anomic 60:40", "RS-Anomic 60:40", "RS-Anomic 60:40", "RS-Anomic 60:40",
    ],
    "Accuracy": [
        70.49, 92.35, 70.33, 92.35,
        81.67, 95.00, 49.15, 96.67,
        78.58, 80.22, 23.53, 100.00,
    ]
})

pivot = accuracy_data.pivot(index="Model", columns="Dataset", values="Accuracy")
ax = pivot.plot(kind="bar", figsize=(11, 6))
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Model Accuracy Across Internal and External Datasets")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 110)
plt.legend(title="Dataset")
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_4_model_accuracy_across_datasets.png", dpi=300)
plt.close()

# -----------------------------
# Figure 5: RS-Anomic strict unseen anomaly-family accuracy
# -----------------------------

strict_data = pd.DataFrame({
    "Anomaly Family": [
        "high-cpu-dispatch",
        "high-fileIO-payment",
        "high-latency-user",
        "high-load-1500",
        "low-bandwidth-user",
        "memory-leak-cart",
        "out-of-order-packets-user",
        "packetloss-user",
        "rt-delay-catalogue",
        "service-down-payment",
    ],
    "Accuracy": [
        100.00,
        95.8234,
        98.0848,
        99.9535,
        100.00,
        100.00,
        85.3986,
        96.8579,
        99.5227,
        100.00,
    ]
})

strict_data = strict_data.sort_values("Accuracy")
plt.figure(figsize=(10, 6))
plt.barh(strict_data["Anomaly Family"], strict_data["Accuracy"])
plt.xlabel("Accuracy (%)")
plt.title("Strict Unseen Anomaly-family Validation on RS-Anomic")
plt.xlim(0, 105)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_5_rs_anomic_unseen_family_accuracy.png", dpi=300)
plt.close()

# -----------------------------
# Figure 6: repeated live detection latency
# -----------------------------

latency_path = Path("results/latency/detection_latency_3run_summary.csv")

if latency_path.exists():
    latency_df = pd.read_csv(latency_path)
else:
    latency_df = pd.DataFrame({
        "label": [
            "cpu_stress",
            "error_rate",
            "high_traffic",
            "inter_service_error",
            "inter_service_slow",
            "memory_stress",
            "slow_response",
        ],
        "mean_delay_seconds": [12.411, 4.843, 7.4067, 2.807, 9.2507, 4.3803, 7.2993],
        "std_delay_seconds": [0.4583, 0.3537, 6.335, 1.6925, 2.8059, 2.3562, 1.1657],
    })

latency_df = latency_df.sort_values("mean_delay_seconds")

plt.figure(figsize=(10, 6))
plt.barh(latency_df["label"], latency_df["mean_delay_seconds"], xerr=latency_df["std_delay_seconds"])
plt.xlabel("Mean detection delay (seconds)")
plt.title("Repeated Live Detection-latency Results")
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_6_detection_latency.png", dpi=300)
plt.close()

# -----------------------------
# Figure 7: comparison with targeted papers
# -----------------------------

comparison_data = pd.DataFrame({
    "Study": [
        "Nobre et al. app-level",
        "Nobre et al. service-level",
        "GAL-MAD RS-Anomic 60:40",
        "CMAnomaly",
        "Your Prometheus XGBoost",
        "Your LO2 XGBoost",
        "Your RS-Anomic strict XGBoost",
    ],
    "Score": [
        71.00,
        97.00,
        95.00,
        94.94,
        92.35,
        96.67,
        97.56,
    ]
})

comparison_data = comparison_data.sort_values("Score")
plt.figure(figsize=(10, 6))
plt.barh(comparison_data["Study"], comparison_data["Score"])
plt.xlabel("Reported score (%)")
plt.title("Comparison with Targeted Published Studies")
plt.xlim(0, 105)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_7_comparison_with_published_studies.png", dpi=300)
plt.close()

# -----------------------------
# Figure 8: detection success rate
# -----------------------------

success_data = pd.DataFrame({
    "Anomaly Type": [
        "CPU stress",
        "Error rate",
        "High traffic",
        "Inter-service error",
        "Inter-service slow",
        "Memory stress",
        "Slow response",
    ],
    "Success Rate": [100, 100, 100, 100, 100, 100, 100]
})

plt.figure(figsize=(9, 5))
plt.bar(success_data["Anomaly Type"], success_data["Success Rate"])
plt.ylabel("Detection success rate (%)")
plt.title("Detection Success Rate Across Repeated Live Anomaly Runs")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_8_detection_success_rate.png", dpi=300)
plt.close()

print("Paper figures generated successfully.")
print(f"Saved in: {OUT_DIR}")
for file in sorted(OUT_DIR.glob("*.png")):
    print(file)
