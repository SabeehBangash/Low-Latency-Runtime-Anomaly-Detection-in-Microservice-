import argparse
import csv
import time
import threading
from datetime import datetime
from pathlib import Path

import requests

OUT_DIR = Path("results/overhead")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUT_DIR / "monitoring_overhead_summary.csv"
RAW_FILE = OUT_DIR / "monitoring_overhead_raw.csv"

SERVICE_URLS = [
    "http://localhost:5001",  # user-service
    "http://localhost:5002",  # product-service
    "http://localhost:5003",  # order-service
]

PROMETHEUS_URL = "http://localhost:9090"


def safe_request(url):
    try:
        t0 = time.perf_counter()
        r = requests.get(url, timeout=5)
        t1 = time.perf_counter()
        return r.status_code == 200, (t1 - t0) * 1000  # ms
    except Exception:
        return False, None


def workload_worker(duration, stop_event, counters, latencies, lock):
    """Continuously hits all three services as fast as possible for `duration` seconds."""
    end_time = time.time() + duration
    while time.time() < end_time and not stop_event.is_set():
        for url in SERVICE_URLS:
            ok, latency_ms = safe_request(url)
            with lock:
                counters["attempted"] += 1
                if ok:
                    counters["succeeded"] += 1
                    latencies.append(latency_ms)


def prometheus_is_scraping():
    """Checks whether Prometheus is currently reachable (best-effort check)."""
    try:
        r = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def run_condition(condition_name, duration, threads, warmup):
    print(f"\n=================================================")
    print(f"Condition: {condition_name}")
    print(f"=================================================")

    prom_status = "reachable" if prometheus_is_scraping() else "NOT reachable"
    print(f"Prometheus status check: {prom_status}")

    print(f"Warmup for {warmup} seconds (let the system settle)...")
    time.sleep(warmup)

    counters = {"attempted": 0, "succeeded": 0}
    latencies = []
    lock = threading.Lock()
    stop_event = threading.Event()

    workers = []
    start_time = time.time()
    start_datetime = datetime.now().isoformat(timespec="seconds")

    for _ in range(threads):
        t = threading.Thread(
            target=workload_worker,
            args=(duration, stop_event, counters, latencies, lock),
            daemon=True,
        )
        workers.append(t)
        t.start()

    for t in workers:
        t.join()

    elapsed = time.time() - start_time

    throughput = counters["succeeded"] / elapsed if elapsed > 0 else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else None

    result = {
        "condition": condition_name,
        "prometheus_status_at_start": prom_status,
        "start_time": start_datetime,
        "duration_seconds": round(elapsed, 2),
        "threads": threads,
        "requests_attempted": counters["attempted"],
        "requests_succeeded": counters["succeeded"],
        "throughput_req_per_sec": round(throughput, 3),
        "mean_latency_ms": round(mean_latency, 3) if mean_latency is not None else "",
    }

    print("\nCondition finished.")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


def append_csv(path, row):
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Measure monitoring (Prometheus scraping) overhead by comparing "
                     "throughput with monitoring ON vs OFF, same workload, same duration."
    )
    parser.add_argument("--duration", type=int, default=120, help="Seconds per condition")
    parser.add_argument("--threads", type=int, default=4, help="Concurrent request workers")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup seconds before measuring")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per condition for averaging")
    args = parser.parse_args()

    print("Monitoring Overhead Measurement")
    print("================================")
    print("This script measures throughput under two conditions:")
    print("  1) monitoring_on  : Prometheus container/service running and scraping normally")
    print("  2) monitoring_off : Prometheus container/service STOPPED")
    print("\nThe four microservices (gateway/user/product/order) must stay running")
    print("identically in both conditions. Only Prometheus itself is stopped/started.\n")

    all_results = []

    for run_id in range(1, args.repeats + 1):
        input(f"\n[Run {run_id}/{args.repeats}] Make sure Prometheus IS RUNNING, then press Enter...")
        result_on = run_condition("monitoring_on", args.duration, args.threads, args.warmup)
        result_on["run_id"] = run_id
        all_results.append(result_on)
        append_csv(RAW_FILE, result_on)

        input(f"\n[Run {run_id}/{args.repeats}] Now STOP Prometheus "
              f"(e.g. `docker stop prometheus` or Ctrl+C the process), then press Enter...")
        result_off = run_condition("monitoring_off", args.duration, args.threads, args.warmup)
        result_off["run_id"] = run_id
        all_results.append(result_off)
        append_csv(RAW_FILE, result_off)

        input(f"\n[Run {run_id}/{args.repeats}] Restart Prometheus before the next run "
              f"(e.g. `docker start prometheus`), then press Enter to continue...")

    # Summarize
    on_runs = [r for r in all_results if r["condition"] == "monitoring_on"]
    off_runs = [r for r in all_results if r["condition"] == "monitoring_off"]

    mean_on = sum(r["throughput_req_per_sec"] for r in on_runs) / len(on_runs)
    mean_off = sum(r["throughput_req_per_sec"] for r in off_runs) / len(off_runs)
    overhead_percent = ((mean_off - mean_on) / mean_off) * 100 if mean_off > 0 else None

    print("\n=================================================")
    print("FINAL SUMMARY")
    print("=================================================")
    print(f"Mean throughput WITH monitoring:    {mean_on:.3f} req/s")
    print(f"Mean throughput WITHOUT monitoring: {mean_off:.3f} req/s")
    if overhead_percent is not None:
        print(f"Monitoring throughput overhead:     {overhead_percent:.2f}%")

    summary_row = {
        "repeats": args.repeats,
        "duration_seconds": args.duration,
        "threads": args.threads,
        "mean_throughput_on_req_per_sec": round(mean_on, 3),
        "mean_throughput_off_req_per_sec": round(mean_off, 3),
        "monitoring_overhead_percent": round(overhead_percent, 3) if overhead_percent is not None else "",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    append_csv(SUMMARY_FILE, summary_row)

    print(f"\nRaw results saved to:     {RAW_FILE}")
    print(f"Summary results saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()