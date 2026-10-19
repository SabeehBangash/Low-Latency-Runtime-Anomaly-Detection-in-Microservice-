from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    "user_up",
    "product_up",
    "order_up",
    "gateway_up",

    "user_request_rate",
    "product_request_rate",
    "order_request_rate",
    "gateway_request_rate",

    "user_avg_latency",
    "product_avg_latency",
    "order_avg_latency",

    "user_cpu",
    "product_cpu",
    "order_cpu",
    "gateway_cpu",

    "user_memory_mb",
    "product_memory_mb",
    "order_memory_mb",
    "gateway_memory_mb",

    "user_error_rate",
    "product_error_rate",
    "order_error_rate",

    "inter_user_request_rate",
    "inter_product_request_rate",
    "inter_order_request_rate",

    "inter_user_error_rate",
    "inter_product_error_rate",
    "inter_order_error_rate",

    "inter_user_avg_latency",
    "inter_product_avg_latency",
    "inter_order_avg_latency",

    "gateway_aggregate_latency",
]


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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
        encoded_output, _ = self.encoder(x)
        decoded_output, _ = self.decoder(encoded_output)
        return decoded_output


def save_text(file_name, text):
    output_path = RESULTS_DIR / file_name
    output_path.write_text(text, encoding="utf-8")
    print(f"Saved: {output_path}")


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
{model_name}

Accuracy:
{accuracy:.4f}

Classification Report:
{report}

Confusion Matrix:
{matrix}
"""

    print(text)
    save_text(f"{model_name.lower().replace(' ', '_')}_binary_results.txt", text)

    return {
        "model": model_name,
        "task": "binary anomaly detection",
        "accuracy": accuracy
    }


def create_sequences(X, y, sequence_length=5):
    X_sequences = []
    y_sequences = []

    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length - 1])

    return np.array(X_sequences), np.array(y_sequences)


def train_autoencoder(X_train_normal, X_test, y_test, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor),
        batch_size=16,
        shuffle=True
    )

    model.train()
    for epoch in range(80):
        total_loss = 0.0

        for batch in train_loader:
            batch_x = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
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
        batch_size=8,
        shuffle=True
    )

    model.train()
    for epoch in range(80):
        total_loss = 0.0

        for batch in train_loader:
            batch_x = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"LSTM Autoencoder epoch {epoch + 1}/80, loss: {total_loss:.6f}")

    model.eval()

    with torch.no_grad():
        train_normal_tensor = torch.tensor(X_train_normal_seq, dtype=torch.float32).to(device)
        train_reconstructed = model(train_normal_tensor).cpu().numpy()

    train_errors = np.mean((X_train_normal_seq - train_reconstructed) ** 2, axis=(1, 2))
    threshold = np.percentile(train_errors, 95)

    with torch.no_grad():
        test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        test_reconstructed = model(test_tensor).cpu().numpy()

    test_errors = np.mean((X_test_seq - test_reconstructed) ** 2, axis=(1, 2))
    y_pred = (test_errors > threshold).astype(int)

    return evaluate_binary_model("LSTM Autoencoder", y_test_seq, y_pred)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")

    X = df[FEATURE_COLUMNS].values
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
    print("1. Training Isolation Forest")
    print("==============================")

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.20,
        random_state=42
    )

    isolation_forest.fit(X_train_normal)

    iso_pred_raw = isolation_forest.predict(X_test)
    iso_pred = np.where(iso_pred_raw == -1, 1, 0)

    all_results.append(
        evaluate_binary_model("Isolation Forest", y_test, iso_pred)
    )

    print("\n==============================")
    print("2. Training Autoencoder")
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
    print("3. Training LSTM Autoencoder")
    print("==============================")

    all_results.append(
        train_lstm_autoencoder(
            X_scaled=X_scaled,
            y_binary=y_binary,
            sequence_length=5
        )
    )

    print("\n==============================")
    print("4. Training XGBoost Binary Classifier")
    print("==============================")

    xgb_binary = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    xgb_binary.fit(X_train, y_train)
    xgb_binary_pred = xgb_binary.predict(X_test)

    all_results.append(
        evaluate_binary_model("XGBoost Binary", y_test, xgb_binary_pred)
    )

    print("\n==============================")
    print("5. Training XGBoost Multiclass Classifier")
    print("==============================")

    label_encoder = LabelEncoder()
    y_multiclass = label_encoder.fit_transform(df["label"].values)

    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_scaled,
        y_multiclass,
        test_size=0.30,
        random_state=42,
        stratify=y_multiclass
    )

    xgb_multi = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )

    xgb_multi.fit(X_train_multi, y_train_multi)
    y_pred_multi = xgb_multi.predict(X_test_multi)

    multi_accuracy = accuracy_score(y_test_multi, y_pred_multi)

    multi_report = classification_report(
        y_test_multi,
        y_pred_multi,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    multi_matrix = confusion_matrix(y_test_multi, y_pred_multi)

    multi_text = f"""
XGBoost Multiclass Anomaly Type Classification

Accuracy:
{multi_accuracy:.4f}

Classification Report:
{multi_report}

Confusion Matrix:
{multi_matrix}
"""

    print(multi_text)
    save_text("xgboost_multiclass_anomaly_type_results.txt", multi_text)

    all_results.append({
        "model": "XGBoost Multiclass",
        "task": "anomaly type classification",
        "accuracy": multi_accuracy
    })

    summary_df = pd.DataFrame(all_results)
    summary_path = RESULTS_DIR / "proposal_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n==============================")
    print("Proposal Model Summary")
    print("==============================")
    print(summary_df)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()