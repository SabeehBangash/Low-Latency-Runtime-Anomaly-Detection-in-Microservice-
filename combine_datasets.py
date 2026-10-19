from pathlib import Path
import pandas as pd


RAW_DIR = Path("datasets/raw")
PROCESSED_DIR = Path("datasets/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "normal_baseline.csv",
    "high_traffic.csv",
    "service_failure.csv",
    "slow_response.csv",
    "cpu_stress.csv",
    "memory_stress.csv",
    "error_rate.csv",
    "inter_service_normal.csv",
    "inter_service_slow.csv",
    "inter_service_error.csv",
]

all_dataframes = []

for file_name in FILES:
    file_path = RAW_DIR / file_name

    if not file_path.exists():
        print(f"Missing file: {file_path}")
        continue

    print(f"Reading: {file_path}")
    df = pd.read_csv(file_path)

    all_dataframes.append(df)

if not all_dataframes:
    raise RuntimeError("No CSV files found. Please check datasets/raw folder.")

combined_df = pd.concat(all_dataframes, ignore_index=True)

# Fill missing values
combined_df = combined_df.fillna(0)

# Create binary anomaly label
# normal = 0, all other labels = 1
combined_df["binary_label"] = combined_df["label"].apply(
    lambda x: 0 if x in ["normal", "inter_service_normal"] else 1
)

# Save combined dataset
output_path = PROCESSED_DIR / "microservices_anomaly_dataset.csv"
combined_df.to_csv(output_path, index=False)

print("\nCombined dataset created successfully.")
print(f"Output file: {output_path}")
print(f"Total rows: {len(combined_df)}")
print(f"Total columns: {len(combined_df.columns)}")

print("\nLabel distribution:")
print(combined_df["label"].value_counts())

print("\nBinary label distribution:")
print(combined_df["binary_label"].value_counts())

print("\nColumns:")
for column in combined_df.columns:
    print(column)