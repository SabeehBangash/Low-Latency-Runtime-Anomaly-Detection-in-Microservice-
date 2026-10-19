from pathlib import Path
import pandas as pd

INPUT_FILE = Path("datasets/processed/lo2_external_log_dataset.csv")
OUTPUT_FILE = Path("datasets/processed/lo2_external_log_dataset_balanced.csv")

df = pd.read_csv(INPUT_FILE)

normal_df = df[df["binary_label"] == 0]
anomaly_df = df[df["binary_label"] == 1]

n_normal = len(normal_df)

balanced_anomaly_df = anomaly_df.sample(
    n=n_normal,
    random_state=42
)

balanced_df = pd.concat(
    [normal_df, balanced_anomaly_df],
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

balanced_df.to_csv(OUTPUT_FILE, index=False)

print("Balanced LO2 dataset created successfully.")
print(f"Output file: {OUTPUT_FILE}")
print(f"Rows: {len(balanced_df)}")
print(f"Columns: {len(balanced_df.columns)}")
print("\nLabel distribution:")
print(balanced_df["label"].value_counts())
print("\nBinary label distribution:")
print(balanced_df["binary_label"].value_counts())