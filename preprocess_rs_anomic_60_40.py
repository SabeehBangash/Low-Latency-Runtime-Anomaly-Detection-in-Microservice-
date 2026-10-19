from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path("datasets/external/rs-anomic")
OUT_DIR = Path("datasets/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "rs_anomic_60_40_processed.csv"
RANDOM_STATE = 42


def clean_name(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def read_numeric_csv(csv_path, prefix):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Could not read {csv_path}: {e}")
        return None

    numeric_cols = []

    for col in df.columns:
        col_lower = str(col).lower()

        if "time" in col_lower or "timestamp" in col_lower or str(col).lower().startswith("unnamed"):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        return None

    numeric_df = df[numeric_cols].copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.interpolate(method="linear", limit_direction="both")
    numeric_df = numeric_df.ffill().bfill().fillna(0)

    # Convert cumulative counters into per-step changes
    for col in numeric_df.columns:
        col_lower = str(col).lower()
        if (
            "total" in col_lower
            or "seconds" in col_lower
            or "bytes" in col_lower
            or "packets" in col_lower
            or "fail" in col_lower
        ):
            numeric_df[col] = numeric_df[col].diff().clip(lower=0)

    numeric_df = numeric_df.drop(index=[0, 1, 2], errors="ignore").reset_index(drop=True)
    numeric_df.columns = [f"{prefix}_{clean_name(col)}" for col in numeric_df.columns]

    return numeric_df


def load_folder_csvs(folder_path, folder_prefix):
    frames = []

    if folder_path is None or not folder_path.exists():
        return frames

    for csv_file in sorted(folder_path.glob("*.csv")):
        file_prefix = f"{folder_prefix}_{clean_name(csv_file.stem)}"
        df = read_numeric_csv(csv_file, file_prefix)

        if df is not None:
            frames.append(df)

    return frames


def find_response_time_folder(base_rt_root, anomaly_name=None):
    if anomaly_name is None:
        folder = base_rt_root
        return folder if folder.exists() else None

    possible_names = [
        anomaly_name,
        f"{anomaly_name}_rt",
        f"{anomaly_name}-rt",
    ]

    for name in possible_names:
        folder = base_rt_root / name
        if folder.exists():
            return folder

    return None


def build_behavior_dataset(label, cadvisor_dir, response_time_dir, binary_label):
    frames = []

    print(f"Reading cAdvisor metrics for: {label}")
    frames.extend(load_folder_csvs(cadvisor_dir, "cadvisor"))

    if response_time_dir is not None and response_time_dir.exists():
        print(f"Reading response-time metrics for: {label}")
        frames.extend(load_folder_csvs(response_time_dir, "response_time"))
    else:
        print(f"No response-time folder found for: {label}")

    if not frames:
        raise RuntimeError(f"No usable CSV files found for {label}")

    min_len = min(len(frame) for frame in frames)
    frames = [frame.iloc[:min_len].reset_index(drop=True) for frame in frames]

    final_df = pd.concat(frames, axis=1)
    final_df["label"] = label
    final_df["binary_label"] = binary_label

    return final_df


def main():
    print("Starting RS-Anomic 60:40 preprocessing...")

    normal_cadvisor = ROOT / "normal_data" / "cAdvisor"
    normal_rt_root = ROOT / "normal_data" / "response_times"

    anomaly_cadvisor_root = ROOT / "anomaly_data" / "cAdvisor"
    anomaly_rt_root = ROOT / "anomaly_data" / "response_times"

    if not normal_cadvisor.exists():
        raise RuntimeError(f"Normal cAdvisor folder not found: {normal_cadvisor}")

    if not anomaly_cadvisor_root.exists():
        raise RuntimeError(f"Anomaly cAdvisor folder not found: {anomaly_cadvisor_root}")

    print("\nLoading normal data...")
    normal_rt = find_response_time_folder(normal_rt_root)
    normal_df = build_behavior_dataset(
        label="normal",
        cadvisor_dir=normal_cadvisor,
        response_time_dir=normal_rt,
        binary_label=0
    )

    anomaly_dfs = []

    print("\nLoading anomaly data...")
    for anomaly_dir in sorted(anomaly_cadvisor_root.iterdir()):
        if not anomaly_dir.is_dir():
            continue

        anomaly_name = anomaly_dir.name
        print(f"\nProcessing anomaly: {anomaly_name}")

        anomaly_rt = find_response_time_folder(anomaly_rt_root, anomaly_name)

        anomaly_df = build_behavior_dataset(
            label=anomaly_name,
            cadvisor_dir=anomaly_dir,
            response_time_dir=anomaly_rt,
            binary_label=1
        )

        anomaly_dfs.append(anomaly_df)

    all_anomaly_df = pd.concat(anomaly_dfs, ignore_index=True)

    normal_df = normal_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    all_anomaly_df = all_anomaly_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    anomaly_count = len(all_anomaly_df)
    required_normal_count = int(anomaly_count * 60 / 40)

    if required_normal_count <= len(normal_df):
        selected_normal = normal_df.sample(n=required_normal_count, random_state=RANDOM_STATE)
        selected_anomaly = all_anomaly_df
    else:
        selected_normal = normal_df
        required_anomaly_count = int(len(selected_normal) * 40 / 60)
        selected_anomaly = all_anomaly_df.sample(n=required_anomaly_count, random_state=RANDOM_STATE)

    final_df = pd.concat([selected_normal, selected_anomaly], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nRS-Anomic 60:40 dataset created successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")

    print("\nLabel distribution:")
    print(final_df["label"].value_counts())

    print("\nBinary label distribution:")
    print(final_df["binary_label"].value_counts())

    print("\nBinary percentage:")
    print(final_df["binary_label"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()