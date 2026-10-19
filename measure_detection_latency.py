import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

PROMETHEUS_URL = "http://localhost:9090"

MODEL_DIR = Path("results/latency")
MODEL_PATH = MODEL_DIR / "xgboost_latency_model.pkl"
SCALER_PATH = MODEL_DIR / "xgboost_latency_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

SUMMARY_FILE = MODEL_DIR / "detection_latency_results.csv"

FEATURE_QUERIES = {
    "user_up": 'up{job="user-service"}',
    "product_up": 'up{job="product-service"}',
    "order_up": 'up{job="order-service"}',
    "gateway_up": 'up{job="gateway-service"}',

    "user_request_rate": 'increase(user_service_requests_total[1m]) / 60',
    "product_request_rate": 'increase(product_service_requests_total[1m]) / 60',
    "order_request_rate": 'increase(order_service_requests_total[1m]) / 60',
    "gateway_request_rate": 'increase(gateway_requests_total[1m]) / 60',

    "user_avg_latency": 'sum(increase(user_service_request_latency_seconds_sum[1m])) / sum(increase(user_service_request_latency_seconds_count[1m]))',
    "product_avg_latency": 'sum(increase(product_service_request_latency_seconds_sum[1m])) / sum(increase(product_service_request_latency_seconds_count[1m]))',
    "order_avg_latency": 'sum(increase(order_service_request_latency_seconds_sum[1m])) / sum(increase(order_service_request_latency_seconds_count[1m]))',

    "user_cpu": 'rate(process_cpu_seconds_total{job="user-service"}[1m]) * 100',
    "product_cpu": 'rate(process_cpu_seconds_total{job="product-service"}[1m]) * 100',
    "order_cpu": 'rate(process_cpu_seconds_total{job="order-service"}[1m]) * 100',
    "gateway_cpu": 'rate(process_cpu_seconds_total{job="gateway-service"}[1m]) * 100',

    "user_memory_mb": 'process_resident_memory_bytes{job="user-service"} / 1024 / 1024',
    "product_memory_mb": 'process_resident_memory_bytes{job="product-service"} / 1024 / 1024',
    "order_memory_mb": 'process_resident_memory_bytes{job="order-service"} / 1024 / 1024',
    "gateway_memory_mb": 'process_resident_memory_bytes{job="gateway-service"} / 1024 / 1024',

    "user_error_rate": 'increase(user_service_errors_total[1m]) / 60',
    "product_error_rate": 'increase(product_service_errors_total[1m]) / 60',
    "order_error_rate": 'increase(order_service_errors_total[1m]) / 60',

    "inter_user_request_rate": 'sum(increase(inter_service_requests_total{target_service="user-service"}[1m])) / 60',
    "inter_product_request_rate": 'sum(increase(inter_service_requests_total{target_service="product-service"}[1m])) / 60',
    "inter_order_request_rate": 'sum(increase(inter_service_requests_total{target_service="order-service"}[1m])) / 60',

    "inter_user_error_rate": 'sum(increase(inter_service_errors_total{target_service="user-service"}[1m])) / 60',
    "inter_product_error_rate": 'sum(increase(inter_service_errors_total{target_service="product-service"}[1m])) / 60',
    "inter_order_error_rate": 'sum(increase(inter_service_errors_total{target_service="order-service"}[1m])) / 60',

    "inter_user_avg_latency": 'sum(increase(inter_service_request_latency_seconds_sum{target_service="user-service"}[1m])) / sum(increase(inter_service_request_latency_seconds_count{target_service="user-service"}[1m]))',
    "inter_product_avg_latency": 'sum(increase(inter_service_request_latency_seconds_sum{target_service="product-service"}[1m])) / sum(increase(inter_service_request_latency_seconds_count{target_service="product-service"}[1m]))',
    "inter_order_avg_latency": 'sum(increase(inter_service_request_latency_seconds_sum{target_service="order-service"}[1m])) / sum(increase(inter_service_request_latency_seconds_count{target_service="order-service"}[1m]))',

    "gateway_aggregate_latency": 'sum(increase(gateway_aggregate_latency_seconds_sum[1m])) / sum(increase(gateway_aggregate_latency_seconds_count[1m]))',
}


def query_prometheus(promql):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": promql},
        timeout=20
    )
    response.raise_for_status()
    data = response.json()

    result = data.get("data", {}).get("result", [])
    if not result:
        return 0.0

    value = result[0]["value"][1]

    try:
        return float(value)
    except Exception:
        return 0.0


def collect_features(feature_columns):
    row = {}

    for feature in feature_columns:
        promql = FEATURE_QUERIES.get(feature)

        if promql is None:
            row[feature] = 0.0
        else:
            row[feature] = query_prometheus(promql)

    return row


def append_summary(row):
    file_exists = SUMMARY_FILE.exists()

    with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Anomaly name, e.g. high_traffic")
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    trace_rows = []

    print("\nLive detection latency measurement started.")
    print(f"Anomaly label: {args.label}")
    print(f"Warmup seconds: {args.warmup}")
    print(f"Polling interval: {args.poll_interval} seconds")
    print(f"Timeout: {args.timeout} seconds")

    print("\nPlease keep system NORMAL during warmup...")
    time.sleep(args.warmup)

    print("\n====================================================")
    print("START THE ANOMALY NOW IN ANOTHER POWERSHELL WINDOW")
    print("====================================================\n")

    start_time = time.time()
    start_datetime = datetime.now().isoformat(timespec="seconds")

    first_detection_time = None
    first_detection_datetime = None
    first_detection_probability = None

    while True:
        now = time.time()
        elapsed = now - start_time

        features = collect_features(feature_columns)
        X = pd.DataFrame([features], columns=feature_columns).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(X.values)

        prediction = int(model.predict(X_scaled)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X_scaled)[0][1])
        else:
            probability = float(prediction)

        current_datetime = datetime.now().isoformat(timespec="seconds")

        trace_row = {
            "label": args.label,
            "timestamp": current_datetime,
            "elapsed_seconds": round(elapsed, 2),
            "prediction": prediction,
            "anomaly_probability": round(probability, 4),
        }

        trace_row.update(features)
        trace_rows.append(trace_row)

        print(
            f"[{current_datetime}] elapsed={elapsed:.2f}s "
            f"prediction={prediction} probability={probability:.4f}"
        )

        if prediction == 1:
            first_detection_time = time.time()
            first_detection_datetime = datetime.now().isoformat(timespec="seconds")
            first_detection_probability = probability
            break

        if elapsed >= args.timeout:
            break

        time.sleep(args.poll_interval)

    trace_file = MODEL_DIR / f"{args.label}_detection_trace.csv"
    pd.DataFrame(trace_rows).to_csv(trace_file, index=False)

    if first_detection_time is not None:
        delay = first_detection_time - start_time
        status = "detected"
    else:
        delay = None
        status = "not_detected"

    summary_row = {
        "model": "XGBoost Binary",
        "label": args.label,
        "status": status,
        "start_time": start_datetime,
        "first_detection_time": first_detection_datetime,
        "detection_delay_seconds": round(delay, 2) if delay is not None else "",
        "poll_interval_seconds": args.poll_interval,
        "first_detection_probability": round(first_detection_probability, 4) if first_detection_probability is not None else "",
        "trace_file": str(trace_file),
    }

    append_summary(summary_row)

    print("\nDetection latency measurement finished.")
    print(f"Status: {status}")
    print(f"Detection delay: {summary_row['detection_delay_seconds']} seconds")
    print(f"Trace saved: {trace_file}")
    print(f"Summary saved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()