from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATASET_PATH = Path("datasets/processed/microservices_anomaly_dataset.csv")
MODEL_DIR = Path("results/latency")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

df = pd.read_csv(DATASET_PATH)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

X = df[FEATURE_COLUMNS].values
y = df["binary_label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_scaled, y)

joblib.dump(model, MODEL_DIR / "xgboost_latency_model.pkl")
joblib.dump(scaler, MODEL_DIR / "xgboost_latency_scaler.pkl")

with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(FEATURE_COLUMNS, f, indent=2)

print("Latency detection model saved successfully.")
print(f"Model: {MODEL_DIR / 'xgboost_latency_model.pkl'}")
print(f"Scaler: {MODEL_DIR / 'xgboost_latency_scaler.pkl'}")
print(f"Features: {MODEL_DIR / 'feature_columns.json'}")