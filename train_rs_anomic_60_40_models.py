from pathlib import Path
import argparse
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DATASET_PATH = Path("datasets/processed/rs_anomic_60_40_processed.csv")
RESULT_DIR = Path("results/rs_anomic_60_40")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def save_text(filename, text):
    path = RESULT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {path}")


def make_report(title, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["normal", "anomaly"],
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    text = f"""
{title}

Accuracy:
{acc:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}
"""
    return acc, text


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            batch_first=True
        )

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        output, _ = self.decoder(repeated)
        return output


def train_autoencoder(X_train_normal, X_test, y_test, input_dim):
    print("\n==============================")
    print("2. Autoencoder on RS-Anomic 60:40")
    print("==============================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=512,
        shuffle=True
    )

    epochs = 50

    model.train()
    for epoch in range(epochs):
        losses = []

        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Autoencoder epoch {epoch + 1}/{epochs}, loss: {np.mean(losses):.6f}")

    model.eval()

    with torch.no_grad():
        train_x = torch.tensor(X_train_normal, dtype=torch.float32).to(device)
        train_recon = model(train_x)
        train_loss = torch.mean((train_recon - train_x) ** 2, dim=1).cpu().numpy()

        test_x = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_recon = model(test_x)
        test_loss = torch.mean((test_recon - test_x) ** 2, dim=1).cpu().numpy()

    threshold = np.percentile(train_loss, 95)
    y_pred = (test_loss > threshold).astype(int)

    acc, text = make_report("Autoencoder on RS-Anomic 60:40 Dataset", y_test, y_pred)
    text += f"\nThreshold: {threshold:.6f}\n"

    print(text)
    save_text("autoencoder_rs_anomic_60_40_results.txt", text)

    torch.save(model.state_dict(), RESULT_DIR / "autoencoder_rs_anomic_60_40.pt")

    return acc


def create_sequences(X, y, seq_len=5):
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(1 if np.any(y[i:i + seq_len] == 1) else 0)

    return np.array(X_seq), np.array(y_seq)


def train_lstm_autoencoder(X_train, y_train, X_test, y_test, input_dim):
    print("\n==============================")
    print("3. LSTM Autoencoder on RS-Anomic 60:40")
    print("==============================")

    seq_len = 5

    X_train_normal = X_train[y_train == 0]
    y_train_normal = y_train[y_train == 0]

    X_train_seq, y_train_seq = create_sequences(X_train_normal, y_train_normal, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

    max_train_sequences = 20000

    if len(X_train_seq) > max_train_sequences:
        np.random.seed(RANDOM_STATE)
        idx = np.random.choice(len(X_train_seq), max_train_sequences, replace=False)
        X_train_seq = X_train_seq[idx]
        y_train_seq = y_train_seq[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=256,
        shuffle=True
    )

    epochs = 30

    model.train()
    for epoch in range(epochs):
        losses = []

        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"LSTM Autoencoder epoch {epoch + 1}/{epochs}, loss: {np.mean(losses):.6f}")

    model.eval()

    with torch.no_grad():
        train_x = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
        train_recon = model(train_x)
        train_loss = torch.mean((train_recon - train_x) ** 2, dim=(1, 2)).cpu().numpy()

        test_x = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        test_recon = model(test_x)
        test_loss = torch.mean((test_recon - test_x) ** 2, dim=(1, 2)).cpu().numpy()

    threshold = np.percentile(train_loss, 95)
    y_pred = (test_loss > threshold).astype(int)

    acc, text = make_report("LSTM Autoencoder on RS-Anomic 60:40 Dataset", y_test_seq, y_pred)
    text += f"\nThreshold: {threshold:.6f}\n"

    print(text)
    save_text("lstm_autoencoder_rs_anomic_60_40_results.txt", text)

    torch.save(model.state_dict(), RESULT_DIR / "lstm_autoencoder_rs_anomic_60_40.pt")

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM if laptop is slow")
    parser.add_argument("--max-rows", type=int, default=0, help="Use smaller sample if needed. 0 means full dataset.")
    args = parser.parse_args()

    print("Loading RS-Anomic 60:40 dataset...")
    df = pd.read_csv(DATASET_PATH)

    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    if args.max_rows and args.max_rows > 0 and args.max_rows < len(df):
        print(f"Using sample of {args.max_rows} rows for memory-safe test...")
        df = df.sample(n=args.max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    ignore_columns = {
        "label",
        "binary_label"
    }

    feature_columns = [
        col for col in df.columns
        if col not in ignore_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"Feature columns used: {len(feature_columns)}")

    X = df[feature_columns].values
    y = df["binary_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, RESULT_DIR / "rs_anomic_60_40_scaler.pkl")

    summary = []

    print("\n==============================")
    print("1. Isolation Forest on RS-Anomic 60:40")
    print("==============================")

    start = time.time()

    X_train_normal = X_train_scaled[y_train == 0]

    iso = IsolationForest(
        n_estimators=150,
        max_samples=min(10000, len(X_train_normal)),
        contamination=0.05,
        random_state=RANDOM_STATE,
        n_jobs=2
    )

    iso.fit(X_train_normal)

    train_scores = -iso.decision_function(X_train_normal)
    test_scores = -iso.decision_function(X_test_scaled)

    threshold = np.percentile(train_scores, 95)
    y_pred_iso = (test_scores > threshold).astype(int)

    acc_iso, text_iso = make_report(
        "Isolation Forest on RS-Anomic 60:40 Dataset",
        y_test,
        y_pred_iso
    )

    text_iso += f"\nThreshold: {threshold:.6f}\n"
    text_iso += f"Runtime seconds: {time.time() - start:.2f}\n"

    print(text_iso)
    save_text("isolation_forest_rs_anomic_60_40_results.txt", text_iso)

    joblib.dump(iso, RESULT_DIR / "isolation_forest_rs_anomic_60_40.pkl")

    summary.append({
        "dataset": "RS-Anomic 60:40",
        "model": "Isolation Forest",
        "task": "binary anomaly detection",
        "accuracy": acc_iso
    })

    acc_ae = train_autoencoder(
        X_train_normal=X_train_scaled[y_train == 0],
        X_test=X_test_scaled,
        y_test=y_test,
        input_dim=len(feature_columns)
    )

    summary.append({
        "dataset": "RS-Anomic 60:40",
        "model": "Autoencoder",
        "task": "binary anomaly detection",
        "accuracy": acc_ae
    })

    if not args.skip_lstm:
        acc_lstm = train_lstm_autoencoder(
            X_train=X_train_scaled,
            y_train=y_train,
            X_test=X_test_scaled,
            y_test=y_test,
            input_dim=len(feature_columns)
        )

        summary.append({
            "dataset": "RS-Anomic 60:40",
            "model": "LSTM Autoencoder",
            "task": "binary anomaly detection",
            "accuracy": acc_lstm
        })
    else:
        print("\nSkipping LSTM Autoencoder because --skip-lstm was used.")

    print("\n==============================")
    print("4. XGBoost Binary on RS-Anomic 60:40")
    print("==============================")

    start = time.time()

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2,
        random_state=RANDOM_STATE
    )

    xgb.fit(X_train_scaled, y_train)

    y_pred_xgb = xgb.predict(X_test_scaled)

    acc_xgb, text_xgb = make_report(
        "XGBoost Binary on RS-Anomic 60:40 Dataset",
        y_test,
        y_pred_xgb
    )

    text_xgb += f"\nRuntime seconds: {time.time() - start:.2f}\n"

    print(text_xgb)
    save_text("xgboost_binary_rs_anomic_60_40_results.txt", text_xgb)

    joblib.dump(xgb, RESULT_DIR / "xgboost_binary_rs_anomic_60_40.pkl")

    summary.append({
        "dataset": "RS-Anomic 60:40",
        "model": "XGBoost Binary",
        "task": "binary anomaly detection",
        "accuracy": acc_xgb
    })

    summary_df = pd.DataFrame(summary)

    print("\n==============================")
    print("RS-Anomic 60:40 Model Summary")
    print("==============================")
    print(summary_df)

    summary_path = RESULT_DIR / "rs_anomic_60_40_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved summary: {summary_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()