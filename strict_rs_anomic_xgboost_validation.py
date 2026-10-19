from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATASET_PATH = Path("datasets/processed/rs_anomic_60_40_processed.csv")
RESULT_DIR = Path("results/rs_anomic_60_40")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULT_DIR / "strict_unseen_anomaly_family_xgboost_results.csv"
DETAIL_FILE = RESULT_DIR / "strict_unseen_anomaly_family_xgboost_details.txt"

RANDOM_STATE = 42


def anomaly_family(label):
    label = str(label)
    label = re.sub(r"-2$", "", label)
    return label


def main():
    print("Loading RS-Anomic 60:40 dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    df["anomaly_family"] = df["label"].apply(anomaly_family)

    ignore_columns = {
        "label",
        "binary_label",
        "anomaly_family",
    }

    feature_columns = [
        col for col in df.columns
        if col not in ignore_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"Feature columns used: {len(feature_columns)}")

    normal_df = df[df["binary_label"] == 0].copy()
    anomaly_families = sorted(df[df["binary_label"] == 1]["anomaly_family"].unique())

    summary_rows = []
    details = []

    for test_family in anomaly_families:
        print("\n====================================")
        print(f"Testing unseen anomaly family: {test_family}")
        print("====================================")

        test_anomaly = df[df["anomaly_family"] == test_family].copy()
        train_anomaly = df[
            (df["binary_label"] == 1) &
            (df["anomaly_family"] != test_family)
        ].copy()

        test_normal_count = min(len(test_anomaly), int(len(normal_df) * 0.30))
        test_normal = normal_df.sample(n=test_normal_count, random_state=RANDOM_STATE)

        train_normal = normal_df.drop(test_normal.index)

        train_df = pd.concat([train_normal, train_anomaly], ignore_index=True)
        test_df = pd.concat([test_normal, test_anomaly], ignore_index=True)

        train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        X_train = train_df[feature_columns].values
        y_train = train_df["binary_label"].values

        X_test = test_df[feature_columns].values
        y_test = test_df["binary_label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
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

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["normal", "anomaly"],
            zero_division=0
        )

        print(f"Accuracy: {acc:.4f}")
        print(report)
        print(cm)

        summary_rows.append({
            "test_anomaly_family": test_family,
            "test_rows": len(test_df),
            "test_normal_rows": int((y_test == 0).sum()),
            "test_anomaly_rows": int((y_test == 1).sum()),
            "accuracy": acc
        })

        details.append(f"""
====================================
Unseen anomaly family: {test_family}
====================================
Test rows: {len(test_df)}
Normal rows: {(y_test == 0).sum()}
Anomaly rows: {(y_test == 1).sum()}

Accuracy:
{acc:.4f}

Classification report:
{report}

Confusion matrix:
{cm}
""")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_FILE, index=False)

    with open(DETAIL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(details))

    print("\nStrict unseen anomaly-family validation completed.")
    print(f"Summary saved: {OUTPUT_FILE}")
    print(f"Details saved: {DETAIL_FILE}")

    print("\nFinal summary:")
    print(summary_df)

    print("\nAverage strict accuracy:")
    print(summary_df["accuracy"].mean())


if __name__ == "__main__":
    main()