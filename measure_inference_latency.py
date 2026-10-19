from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

DATASET_PATH = Path("datasets/processed/microservices_anomaly_dataset.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "user_up", "product_up", "order_up", "gateway_up",
    "user_request_rate", "product_request_rate", "order_request_rate", "gateway_request_rate",
    "user_avg_latency", "product_avg_latency", "order_avg_latency",
    "user_cpu", "product_cpu", "order_cpu", "gateway_cpu",
    "user_memory_mb", "product_memory_mb", "order_memory_mb", "gateway_memory_mb",
    "user_error_rate", "product_error_rate", "order_error_rate",
    "inter_user_request_rate", "inter_product_request_rate", "inter_order_request_rate",
    "inter_user_error_rate", "inter_product_error_rate", "inter_order_error_rate",
    "inter_user_avg_latency", "inter_product_avg_latency", "inter_order_avg_latency",
    "gateway_aggregate_latency",
]


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24), nn.ReLU(),
            nn.Linear(24, 12), nn.ReLU(),
            nn.Linear(12, 6), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 12), nn.ReLU(),
            nn.Linear(12, 24), nn.ReLU(),
            nn.Linear(24, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=input_dim, batch_first=True)

    def forward(self, x):
        encoded_output, _ = self.encoder(x)
        decoded_output, _ = self.decoder(encoded_output)
        return decoded_output


def time_single_predictions(predict_fn, sample, n_runs=200, warmup=20):
    """Times n_runs single-sample predictions; returns list of latencies in ms."""
    for _ in range(warmup):
        predict_fn(sample)

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(sample)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # ms
    return timings


def main():
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = df[FEATURE_COLUMNS].values
    y_binary = df["binary_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.30, random_state=42, stratify=y_binary
    )
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    X_train_normal = X_train_sub[y_train_sub == 0]

    single_sample = X_test[0:1]  # shape (1, 32) — one "live" sample

    results = {}

    # --- Isolation Forest ---
    iso = IsolationForest(n_estimators=200, contamination=0.20, random_state=42)
    iso.fit(X_train_normal)
    timings = time_single_predictions(lambda s: iso.predict(s), single_sample)
    results["Isolation Forest"] = timings

    # --- Autoencoder ---
    device = torch.device("cpu")  # live deployment is CPU; keep comparable
    ae = Autoencoder(X_scaled.shape[1]).to(device)
    ae.eval()
    sample_tensor = torch.tensor(single_sample, dtype=torch.float32).to(device)

    def ae_predict(s):
        with torch.no_grad():
            return ae(torch.tensor(s, dtype=torch.float32).to(device)).cpu().numpy()

    timings = time_single_predictions(ae_predict, single_sample)
    results["Autoencoder"] = timings

    # --- LSTM Autoencoder ---
    lstm = LSTMAutoencoder(input_dim=X_scaled.shape[1], hidden_dim=32).to(device)
    lstm.eval()
    seq_sample = np.repeat(single_sample, 5, axis=0).reshape(1, 5, X_scaled.shape[1])

    def lstm_predict(s):
        with torch.no_grad():
            return lstm(torch.tensor(s, dtype=torch.float32).to(device)).cpu().numpy()

    timings = time_single_predictions(lstm_predict, seq_sample)
    results["LSTM Autoencoder"] = timings

    # --- XGBoost Binary ---
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )
    xgb.fit(X_train, y_train)
    timings = time_single_predictions(lambda s: xgb.predict(s), single_sample)
    results["XGBoost Binary"] = timings

    # --- Summarize ---
    summary_rows = []
    for model_name, timings in results.items():
        timings = np.array(timings)
        summary_rows.append({
            "model": model_name,
            "mean_ms": round(timings.mean(), 4),
            "median_ms": round(np.median(timings), 4),
            "min_ms": round(timings.min(), 4),
            "max_ms": round(timings.max(), 4),
            "std_ms": round(timings.std(), 4),
            "n_runs": len(timings),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n==============================")
    print("Per-Model Single-Sample Inference Latency (CPU, 200 runs each)")
    print("==============================")
    print(summary_df.to_string(index=False))

    out_csv = RESULTS_DIR / "per_model_inference_latency.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()