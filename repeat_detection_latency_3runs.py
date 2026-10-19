import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from measure_detection_latency import collect_features

MODEL_DIR = Path("results/latency")
MODEL_PATH = MODEL_DIR / "xgboost_latency_model.pkl"
SCALER_PATH = MODEL_DIR / "xgboost_latency_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

RAW_OUTPUT = MODEL_DIR / "detection_latency_3run_raw.csv"
SUMMARY_OUTPUT = MODEL_DIR / "detection_latency_3run_summary.csv"
TRACE_DIR = MODEL_DIR / "repeated_traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "high_traffic",
    "slow_response",
    "cpu_stress",
    "memory_stress",
    "error_rate",
    "inter_service_error",
    "inter_service_slow",
]


def safe_request(url):
    try:
        requests.get(url, timeout=10)
    except Exception:
        pass


def generate_anomaly(label, stop_event):
    print(f"Anomaly generator started for: {label}")

    while not stop_event.is_set():
        if label == "high_traffic":
            safe_request("http://localhost:5001")
            safe_request("http://localhost:5002")
            safe_request("http://localhost:5003")
            time.sleep(0.10)

        elif label == "slow_response":
            safe_request("http://localhost:5001/slow")
            safe_request("http://localhost:5002/slow")
            safe_request("http://localhost:5003/slow")

        elif label == "cpu_stress":
            safe_request("http://localhost:5001/cpu")
            safe_request("http://localhost:5002/cpu")
            safe_request("http://localhost:5003/cpu")

        elif label == "memory_stress":
            safe_request("http://localhost:5001/memory")
            safe_request("http://localhost:5002/memory")
            safe_request("http://localhost:5003/memory")
            time.sleep(5)

        elif label == "error_rate":
            safe_request("http://localhost:5001/error")
            safe_request("http://localhost:5002/error")
            safe_request("http://localhost:5003/error")
            time.sleep(0.50)

        elif label == "inter_service_error":
            safe_request("http://localhost:5000/aggregate?mode=error")
            time.sleep(0.50)

        elif label == "inter_service_slow":
            safe_request("http://localhost:5000/aggregate?mode=slow")

        else:
            raise ValueError(f"Unknown label: {label}")

    print(f"Anomaly generator stopped for: {label}")


def clear_memory():
    safe_request("http://localhost:5001/memory-clear")
    safe_request("http://localhost:5002/memory-clear")
    safe_request("http://localhost:5003/memory-clear")


def run_one_experiment(label, run_id, model, scaler, feature_columns, warmup, timeout, poll_interval):
    print("\n=================================================")
    print(f"Running latency experiment: {label}, run {run_id}")
    print("=================================================")

    print(f"Warmup for {warmup} seconds...")
    time.sleep(warmup)

    stop_event = threading.Event()
    anomaly_thread = threading.Thread(
        target=generate_anomaly,
        args=(label, stop_event),
        daemon=True
    )

    start_time = time.time()
    start_datetime = datetime.now().isoformat(timespec="seconds")

    anomaly_thread.start()

    detected = False
    detection_time = None
    detection_datetime = None
    detection_probability = None

    trace_rows = []

    while True:
        elapsed = time.time() - start_time

        features = collect_features(feature_columns)
        X = pd.DataFrame([features], columns=feature_columns)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        X_scaled = scaler.transform(X.values)

        prediction = int(model.predict(X_scaled)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X_scaled)[0][1])
        else:
            probability = float(prediction)

        current_datetime = datetime.now().isoformat(timespec="seconds")

        trace_row = {
            "label": label,
            "run_id": run_id,
            "timestamp": current_datetime,
            "elapsed_seconds": round(elapsed, 3),
            "prediction": prediction,
            "anomaly_probability": round(probability, 4),
        }
        trace_row.update(features)
        trace_rows.append(trace_row)

        print(
            f"[{current_datetime}] label={label} run={run_id} "
            f"elapsed={elapsed:.2f}s prediction={prediction} probability={probability:.4f}"
        )

        if prediction == 1:
            detected = True
            detection_time = time.time()
            detection_datetime = datetime.now().isoformat(timespec="seconds")
            detection_probability = probability
            break

        if elapsed >= timeout:
            break

        time.sleep(poll_interval)

    stop_event.set()
    anomaly_thread.join(timeout=3)

    if label == "memory_stress":
        clear_memory()

    trace_file = TRACE_DIR / f"{label}_run_{run_id}_trace.csv"
    pd.DataFrame(trace_rows).to_csv(trace_file, index=False)

    if detected:
        delay = detection_time - start_time
        status = "detected"
    else:
        delay = np.nan
        status = "not_detected"

    result = {
        "model": "XGBoost Binary",
        "label": label,
        "run_id": run_id,
        "status": status,
        "start_time": start_datetime,
        "first_detection_time": detection_datetime,
        "detection_delay_seconds": round(delay, 3) if detected else np.nan,
        "first_detection_probability": round(detection_probability, 4) if detected else np.nan,
        "trace_file": str(trace_file),
    }

    print("\nRun finished.")
    print(result)

    return result


def summarize(raw_df):
    detected_df = raw_df[raw_df["status"] == "detected"].copy()

    summary = detected_df.groupby("label").agg(
        runs=("run_id", "count"),
        mean_delay_seconds=("detection_delay_seconds", "mean"),
        min_delay_seconds=("detection_delay_seconds", "min"),
        max_delay_seconds=("detection_delay_seconds", "max"),
        std_delay_seconds=("detection_delay_seconds", "std"),
        mean_probability=("first_detection_probability", "mean"),
    ).reset_index()

    total_runs = raw_df.groupby("label")["run_id"].count().reset_index(name="total_runs")
    detected_runs = detected_df.groupby("label")["run_id"].count().reset_index(name="detected_runs")

    summary = summary.merge(total_runs, on="label", how="left")
    summary = summary.merge(detected_runs, on="label", how="left")

    summary["detection_success_rate_percent"] = (
        summary["detected_runs"] / summary["total_runs"] * 100
    )

    numeric_cols = [
        "mean_delay_seconds",
        "min_delay_seconds",
        "max_delay_seconds",
        "std_delay_seconds",
        "mean_probability",
        "detection_success_rate_percent",
    ]

    for col in numeric_cols:
        summary[col] = summary[col].round(4)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--cooldown", type=int, default=65)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading latency detection model...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    all_results = []

    for label in LABELS:
        for run_id in range(1, args.repeats + 1):
            result = run_one_experiment(
                label=label,
                run_id=run_id,
                model=model,
                scaler=scaler,
                feature_columns=feature_columns,
                warmup=args.warmup,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )

            all_results.append(result)

            print(f"\nCooldown for {args.cooldown} seconds before next run...")
            print("This is important because Prometheus queries use a 1-minute window.")
            time.sleep(args.cooldown)

    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv(RAW_OUTPUT, index=False)

    summary_df = summarize(raw_df)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    print("\n=================================================")
    print("All repeated detection-latency experiments finished.")
    print("=================================================")

    print(f"\nRaw results saved: {RAW_OUTPUT}")
    print(f"Summary saved: {SUMMARY_OUTPUT}")

    print("\nFinal summary:")
    print(summary_df)


if __name__ == "__main__":
    main()