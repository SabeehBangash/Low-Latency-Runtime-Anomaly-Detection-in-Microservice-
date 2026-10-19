from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest

from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


DATASET_PATH = Path("datasets/processed/lo2_external_log_dataset_balanced.csv")
RESULTS_DIR = Path("results/lo2_external")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 12),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
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
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


def save_text(file_name, text):
    path = RESULTS_DIR / file_name
    path.write_text(text, encoding="utf-8")
    print(f"Saved: {path}")


def evaluate_binary_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["normal", "anomaly"],
        zero_division=0
    )

    matrix = confusion_matrix(y_true, y_pred)

    text = f"""
{model_name} on LO2 External Dataset

Accuracy:
{accuracy:.4f}

Classification Report:
{report}

Confusion Matrix:
{matrix}
"""

    print(text)
    save_text(f"{model_name.lower().replace(' ', '_')}_lo2_results.txt", text)

    return {
        "dataset": "LO2 external log dataset",
        "model": model_name,
        "task": "binary anomaly detection",
        "accuracy": accuracy
    }


def create_sequences(X, y, sequence_length=5):
    X_seq = []
    y_seq = []

    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])

    return np.array(X_seq), np.array(y_seq)


def train_autoencoder(X_train_normal, X_test, y_test, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=32,
        shuffle=True
    )

    model.train()

    for epoch in range(80):
        total_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Autoencoder epoch {epoch + 1}/80, loss: {total_loss:.6f}")

    model.eval()

    with torch.no_grad():
        train_tensor = torch.tensor(X_train_normal, dtype=torch.float32).to(device)
        train_reconstructed = model(train_tensor).cpu().numpy()

    train_errors = np.mean((X_train_normal - train_reconstructed) ** 2, axis=1)
    threshold = np.percentile(train_errors, 95)

    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_reconstructed = model(test_tensor).cpu().numpy()

    test_errors = np.mean((X_test - test_reconstructed) ** 2, axis=1)
    y_pred = (test_errors > threshold).astype(int)

    return evaluate_binary_model("Autoencoder", y_test, y_pred)


def train_lstm_autoencoder(X_scaled, y_binary, sequence_length=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_seq, y_seq = create_sequences(X_scaled, y_binary, sequence_length)

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq,
        y_seq,
        test_size=0.30,
        random_state=42,
        stratify=y_seq
    )

    X_train_normal_seq = X_train_seq[y_train_seq == 0]

    input_dim = X_scaled.shape[1]

    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(X_train_normal_seq, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=16,
        shuffle=True
    )

    model.train()

    for epoch in range(80):
        total_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"LSTM Autoencoder epoch {epoch + 1}/80, loss: {total_loss:.6f}")

    model.eval()

    with torch.no_grad():
        train_tensor = torch.tensor(X_train_normal_seq, dtype=torch.float32).to(device)
        train_reconstructed = model(train_tensor).cpu().numpy()

    train_errors = np.mean((X_train_normal_seq - train_reconstructed) ** 2, axis=(1, 2))
    threshold = np.percentile(train_errors, 95)

    with torch.no_grad():
        test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        test_reconstructed = model(test_tensor).cpu().numpy()

    test_errors = np.mean((X_test_seq - test_reconstructed) ** 2, axis=(1, 2))
    y_pred = (test_errors > threshold).astype(int)

    return evaluate_binary_model("LSTM Autoencoder", y_test_seq, y_pred)


def main():
    print("Loading LO2 external dataset...")
    df = pd.read_csv(DATASET_PATH)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    ignore_columns = {
    "run_id",
    "scenario",
    "label",
    "binary_label",
    "status_code_from_name",
}

    feature_columns = [
        col for col in df.columns
        if col not in ignore_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"Feature columns used: {len(feature_columns)}")

    X = df[feature_columns].values
    y_binary = df["binary_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_binary,
        test_size=0.30,
        random_state=42,
        stratify=y_binary
    )

    X_train_normal = X_train[y_train == 0]

    all_results = []

    print("\n==============================")
    print("1. Isolation Forest on LO2")
    print("==============================")

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.20,
        random_state=42
    )

    isolation_forest.fit(X_train_normal)
    raw_pred = isolation_forest.predict(X_test)
    y_pred = np.where(raw_pred == -1, 1, 0)

    all_results.append(
        evaluate_binary_model("Isolation Forest", y_test, y_pred)
    )

    print("\n==============================")
    print("2. Autoencoder on LO2")
    print("==============================")

    all_results.append(
        train_autoencoder(
            X_train_normal=X_train_normal,
            X_test=X_test,
            y_test=y_test,
            input_dim=X_scaled.shape[1]
        )
    )

    print("\n==============================")
    print("3. LSTM Autoencoder on LO2")
    print("==============================")

    all_results.append(
        train_lstm_autoencoder(
            X_scaled=X_scaled,
            y_binary=y_binary,
            sequence_length=5
        )
    )

    print("\n==============================")
    print("4. XGBoost Binary on LO2")
    print("==============================")

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    all_results.append(
        evaluate_binary_model("XGBoost Binary", y_test, xgb_pred)
    )

    summary_df = pd.DataFrame(all_results)
    summary_path = RESULTS_DIR / "lo2_external_proposal_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n==============================")
    print("LO2 External Proposal Model Summary")
    print("==============================")
    print(summary_df)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()