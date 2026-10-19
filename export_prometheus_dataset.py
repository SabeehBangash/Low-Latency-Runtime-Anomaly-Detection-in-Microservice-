import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROMETHEUS_URL = "http://localhost:9090"

LABEL_MAP = {
    "normal": 0,
    "high_traffic": 1,
    "service_failure": 2,
    "slow_response": 3,
    "cpu_stress": 4,
    "memory_stress": 5,
    "error_rate": 6,
    "inter_service_normal": 7,
    "inter_service_slow": 8,
    "inter_service_error": 9,
}

FEATURE_QUERIES = {
    # Availability
    "user_up": 'up{job="user-service"}',
    "product_up": 'up{job="product-service"}',
    "order_up": 'up{job="order-service"}',
    "gateway_up": 'up{job="gateway-service"}',

    # Request rate
    "user_request_rate": 'increase(user_service_requests_total[1m]) / 60',
    "product_request_rate": 'increase(product_service_requests_total[1m]) / 60',
    "order_request_rate": 'increase(order_service_requests_total[1m]) / 60',
    "gateway_request_rate": 'increase(gateway_requests_total[1m]) / 60',

    # Average latency
    "user_avg_latency": 'sum(increase(user_service_request_latency_seconds_sum[1m])) / sum(increase(user_service_request_latency_seconds_count[1m]))',
    "product_avg_latency": 'sum(increase(product_service_request_latency_seconds_sum[1m])) / sum(increase(product_service_request_latency_seconds_count[1m]))',
    "order_avg_latency": 'sum(increase(order_service_request_latency_seconds_sum[1m])) / sum(increase(order_service_request_latency_seconds_count[1m]))',

    # CPU
    "user_cpu": 'rate(process_cpu_seconds_total{job="user-service"}[1m]) * 100',
    "product_cpu": 'rate(process_cpu_seconds_total{job="product-service"}[1m]) * 100',
    "order_cpu": 'rate(process_cpu_seconds_total{job="order-service"}[1m]) * 100',
    "gateway_cpu": 'rate(process_cpu_seconds_total{job="gateway-service"}[1m]) * 100',

    # Memory
    "user_memory_mb": 'process_resident_memory_bytes{job="user-service"} / 1024 / 1024',
    "product_memory_mb": 'process_resident_memory_bytes{job="product-service"} / 1024 / 1024',
    "order_memory_mb": 'process_resident_memory_bytes{job="order-service"} / 1024 / 1024',
    "gateway_memory_mb": 'process_resident_memory_bytes{job="gateway-service"} / 1024 / 1024',

    # Error rate
    "user_error_rate": 'increase(user_service_errors_total[1m]) / 60',
    "product_error_rate": 'increase(product_service_errors_total[1m]) / 60',
    "order_error_rate": 'increase(order_service_errors_total[1m]) / 60',

    # Inter-service request rate by target service
    "inter_user_request_rate": 'sum(increase(inter_service_requests_total{target_service="user-service"}[1m])) / 60',
    "inter_product_request_rate": 'sum(increase(inter_service_requests_total{target_service="product-service"}[1m])) / 60',
    "inter_order_request_rate": 'sum(increase(inter_service_requests_total{target_service="order-service"}[1m])) / 60',

    # Inter-service error rate by target service
    "inter_user_error_rate": 'sum(increase(inter_service_errors_total{target_service="user-service"}[1m])) / 60',
    "inter_product_error_rate": 'sum(increase(inter_service_errors_total{target_service="product-service"}[1m])) / 60',
    "inter_order_error_rate": 'sum(increase(inter_service_errors_total{target_service="order-service"}[1m])) / 60',

    # Inter-service average latency by target service
    "inter_user_avg_latency": 'sum(increase(inter_service_request_latency_seconds_sum{target_service="user-service"}[1m])) / sum(increase(inter_service_request_latency_seconds_count{target_service="user-service"}[1m]))',
    "inter_product_avg_latency": 'sum(increase(inter_service_request_latency_seconds_sum{target_service="product-service"}[1m])) / sum(increase(inter_service_request_latency_seconds_count{target_service="product-service"}[1m]))',
    "inter_order_avg_latency": 'sum(increase(inter_service_request_latency_seconds_sum{target_service="order-service"}[1m])) / sum(increase(inter_service_request_latency_seconds_count{target_service="order-service"}[1m]))',

    # Gateway aggregate latency
    "gateway_aggregate_latency": 'sum(increase(gateway_aggregate_latency_seconds_sum[1m])) / sum(increase(gateway_aggregate_latency_seconds_count[1m]))',
}


def query_range(promql, start, end, step):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {
        "query": promql,
        "start": start,
        "end": end,
        "step": step
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")

    result = data["data"]["result"]

    if not result:
        return {}

    values = result[0]["values"]
    output = {}

    for ts, value in values:
        timestamp = int(float(ts))

        try:
            output[timestamp] = float(value)
        except ValueError:
            output[timestamp] = 0.0

    return output


def export_dataset(label, minutes, output_path, step):
    end = int(time.time())
    start = end - (minutes * 60)

    rows = {}

    for feature_name, promql in FEATURE_QUERIES.items():
        print(f"Exporting feature: {feature_name}")

        try:
            values = query_range(promql, start, end, step)
        except Exception as error:
            print(f"Warning: failed to export {feature_name}: {error}")
            values = {}

        for timestamp in range(start, end + 1, step):
            if timestamp not in rows:
                rows[timestamp] = {
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp).isoformat()
                }

            rows[timestamp][feature_name] = values.get(timestamp, 0.0)

    df = pd.DataFrame(rows.values()).sort_values("timestamp")

    for feature_name in FEATURE_QUERIES:
        if feature_name not in df.columns:
            df[feature_name] = 0.0

        df[feature_name] = df[feature_name].fillna(0.0)

    df["label"] = label
    df["label_id"] = LABEL_MAP.get(label, -1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nDataset exported successfully")
    print(f"Output file: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Label: {label}")
    print(f"Time window: last {minutes} minutes")
    print(f"Step: {step} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        required=True,
        help="normal, high_traffic, service_failure, slow_response, cpu_stress, memory_stress, error_rate, inter_service_normal, inter_service_slow, inter_service_error"
    )
    parser.add_argument("--minutes", type=int, default=5)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    export_dataset(
        label=args.label,
        minutes=args.minutes,
        output_path=args.output,
        step=args.step
    )