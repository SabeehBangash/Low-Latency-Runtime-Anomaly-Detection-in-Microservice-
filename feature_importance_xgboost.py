from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATASET_PATH = Path("datasets/processed/microservices_anomaly_dataset.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR = Path("results/paper_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

    xgb_binary = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )
    xgb_binary.fit(X_train, y_train)

    importances = xgb_binary.feature_importances_

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n==============================")
    print("XGBoost Binary Feature Importance (sorted, descending)")
    print("==============================")
    print(importance_df.to_string(index=False))

    out_csv = RESULTS_DIR / "xgboost_binary_feature_importance.csv"
    importance_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Plot top 15 features
    top_n = importance_df.head(15)
    plt.figure(figsize=(9, 7))
    plt.barh(top_n["feature"][::-1], top_n["importance"][::-1])
    plt.xlabel("Feature importance (gain-based)")
    plt.title("XGBoost Binary: Top 15 Most Discriminative Features")
    plt.tight_layout()
    out_fig = FIG_DIR / "figure_feature_importance_xgboost_binary.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()