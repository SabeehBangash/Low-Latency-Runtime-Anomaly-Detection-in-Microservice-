from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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


def train_autoencoder_model(X_train_normal, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train_tensor), batch_size=16, shuffle=True)

    model.train()
    for epoch in range(80):
        for batch in loader:
            batch_x = batch[0].to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_x)
            loss.backward()
            optimizer.step()
    return model, device


def main():
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = df[FEATURE_COLUMNS].values
    y_binary = df["binary_label"].values
    y_label = df["label"].values  # original anomaly-type label, kept for per-type scoring

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Single split, carrying the original label alongside binary label and features
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=y_binary
    )

    X_train, X_test = X_scaled[idx_train], X_scaled[idx_test]
    y_train_bin, y_test_bin = y_binary[idx_train], y_binary[idx_test]
    y_test_label = y_label[idx_test]  # anomaly-type ground truth for the test set only

    X_train_sub, X_val, y_train_sub_bin, y_val_bin = train_test_split(
        X_train, y_train_bin, test_size=0.20, random_state=42, stratify=y_train_bin
    )
    X_train_normal = X_train_sub[y_train_sub_bin == 0]
    X_val_normal = X_val[y_val_bin == 0]

    predictions = {}

    # --- Isolation Forest ---
    iso = IsolationForest(n_estimators=200, contamination=0.20, random_state=42)
    iso.fit(X_train_normal)
    iso_pred_raw = iso.predict(X_test)
    predictions["Isolation Forest"] = np.where(iso_pred_raw == -1, 1, 0)

    # --- Autoencoder (validation-calibrated threshold) ---
    ae_model, device = train_autoencoder_model(X_train_normal, X_scaled.shape[1])
    ae_model.eval()
    with torch.no_grad():
        val_recon = ae_model(torch.tensor(X_val_normal, dtype=torch.float32).to(device)).cpu().numpy()
    val_errors = np.mean((X_val_normal - val_recon) ** 2, axis=1)
    ae_threshold = np.percentile(val_errors, 95)
    with torch.no_grad():
        test_recon = ae_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    test_errors = np.mean((X_test - test_recon) ** 2, axis=1)
    predictions["Autoencoder"] = (test_errors > ae_threshold).astype(int)

    # --- XGBoost Binary ---
    xgb_binary = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )
    xgb_binary.fit(X_train, y_train_bin)
    predictions["XGBoost Binary"] = xgb_binary.predict(X_test)

    # --- Build per-anomaly-type recall table ---
    anomaly_types = sorted(set(y_test_label) - {"normal"})
    rows = []
    for anomaly_type in anomaly_types:
        mask = (y_test_label == anomaly_type)
        n_samples = mask.sum()
        row = {"anomaly_type": anomaly_type, "n_test_samples": int(n_samples)}
        for model_name, y_pred in predictions.items():
            if n_samples > 0:
                recall = (y_pred[mask] == 1).mean()
            else:
                recall = np.nan
            row[model_name] = round(float(recall), 4)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    print("\n==============================")
    print("Per-Anomaly-Type Recall (test set, by model)")
    print("==============================")
    print(result_df.to_string(index=False))

    out_csv = RESULTS_DIR / "per_anomaly_type_recall.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print("\nNote: LSTM Autoencoder omitted from this table because its sequence-based")
    print("windowing changes the sample-to-label alignment; it would need separate handling.")


if __name__ == "__main__":
    main()