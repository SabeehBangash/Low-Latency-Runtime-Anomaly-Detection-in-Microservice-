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


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=input_dim, batch_first=True)

    def forward(self, x):
        encoded_output, _ = self.encoder(x)
        decoded_output, _ = self.decoder(encoded_output)
        return decoded_output


def create_sequences_with_type(X, y_binary, y_type, sequence_length=5):
    """
    Same windowing logic as train_models.py's create_sequences(), but also
    tracks the original anomaly-type label (y_type) for each window, taken
    from the same final-row-in-window position used for the binary label.
    """
    X_sequences = []
    y_binary_sequences = []
    y_type_sequences = []

    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:i + sequence_length])
        y_binary_sequences.append(y_binary[i + sequence_length - 1])
        y_type_sequences.append(y_type[i + sequence_length - 1])

    return (
        np.array(X_sequences),
        np.array(y_binary_sequences),
        np.array(y_type_sequences),
    )


def train_autoencoder_model(X_train_normal, input_dim, epochs=80, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            batch_x = batch[0].to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_x)
            loss.backward()
            optimizer.step()
    return model, device


def train_lstm_autoencoder_model(X_train_normal_seq, input_dim, epochs=80, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(X_train_normal_seq, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
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
    y_label = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Row-level split (Isolation Forest, Autoencoder, XGBoost Binary)
    # ------------------------------------------------------------------
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=y_binary
    )

    X_train, X_test = X_scaled[idx_train], X_scaled[idx_test]
    y_train_bin, y_test_bin = y_binary[idx_train], y_binary[idx_test]
    y_test_label = y_label[idx_test]

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

    # --- Autoencoder ---
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

    # Build the row-level results table first (3 models)
    anomaly_types = sorted(set(y_test_label) - {"normal"})
    rows = {a: {"anomaly_type": a, "n_test_samples_rowlevel": int((y_test_label == a).sum())}
            for a in anomaly_types}

    for model_name, y_pred in predictions.items():
        for a in anomaly_types:
            mask = (y_test_label == a)
            recall = (y_pred[mask] == 1).mean() if mask.sum() > 0 else np.nan
            rows[a][model_name] = round(float(recall), 4)

    # ------------------------------------------------------------------
    # Sequence-level split (LSTM Autoencoder) — mirrors train_models.py
    # create_sequences(), but also carries the anomaly-type label per window
    # ------------------------------------------------------------------
    X_seq, y_seq_bin, y_seq_type = create_sequences_with_type(
        X_scaled, y_binary, y_label, sequence_length=5
    )

    X_train_seq, X_test_seq, y_train_seq_bin, y_test_seq_bin, y_train_seq_type, y_test_seq_type = \
        train_test_split(
            X_seq, y_seq_bin, y_seq_type,
            test_size=0.30, random_state=42, stratify=y_seq_bin
        )

    X_train_sub_seq, X_val_seq, y_train_sub_seq_bin, y_val_seq_bin = train_test_split(
        X_train_seq, y_train_seq_bin, test_size=0.20, random_state=42, stratify=y_train_seq_bin
    )
    X_train_normal_seq = X_train_sub_seq[y_train_sub_seq_bin == 0]
    X_val_normal_seq = X_val_seq[y_val_seq_bin == 0]

    lstm_model, device2 = train_lstm_autoencoder_model(X_train_normal_seq, X_scaled.shape[1])
    lstm_model.eval()
    with torch.no_grad():
        val_recon_seq = lstm_model(
            torch.tensor(X_val_normal_seq, dtype=torch.float32).to(device2)
        ).cpu().numpy()
    val_errors_seq = np.mean((X_val_normal_seq - val_recon_seq) ** 2, axis=(1, 2))
    lstm_threshold = np.percentile(val_errors_seq, 95)

    with torch.no_grad():
        test_recon_seq = lstm_model(
            torch.tensor(X_test_seq, dtype=torch.float32).to(device2)
        ).cpu().numpy()
    test_errors_seq = np.mean((X_test_seq - test_recon_seq) ** 2, axis=(1, 2))
    lstm_pred = (test_errors_seq > lstm_threshold).astype(int)

    # Add LSTM recall per anomaly type, using sequence-level labels
    for a in anomaly_types:
        mask_seq = (y_test_seq_type == a)
        n_seq = int(mask_seq.sum())
        recall_seq = (lstm_pred[mask_seq] == 1).mean() if n_seq > 0 else np.nan
        if a not in rows:
            rows[a] = {"anomaly_type": a, "n_test_samples_rowlevel": 0}
        rows[a]["n_test_samples_seqlevel"] = n_seq
        rows[a]["LSTM Autoencoder"] = round(float(recall_seq), 4) if n_seq > 0 else np.nan

    result_df = pd.DataFrame(list(rows.values()))
    col_order = [
        "anomaly_type", "n_test_samples_rowlevel", "n_test_samples_seqlevel",
        "Isolation Forest", "Autoencoder", "XGBoost Binary", "LSTM Autoencoder"
    ]
    col_order = [c for c in col_order if c in result_df.columns]
    result_df = result_df[col_order]

    print("\n==============================")
    print("Per-Anomaly-Type Recall (all four models)")
    print("==============================")
    print(result_df.to_string(index=False))

    out_csv = RESULTS_DIR / "per_anomaly_type_recall_v2.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print("\nNote: n_test_samples_rowlevel applies to Isolation Forest, Autoencoder, and")
    print("XGBoost Binary (row-level evaluation). n_test_samples_seqlevel applies to LSTM")
    print("Autoencoder, whose 5-row sequence windows are labelled by their final row and")
    print("therefore do not align one-to-one with the row-level test set.")


if __name__ == "__main__":
    main()