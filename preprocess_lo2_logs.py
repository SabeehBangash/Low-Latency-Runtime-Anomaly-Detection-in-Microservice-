from pathlib import Path
import re
import pandas as pd

LO2_ROOT = Path("datasets/external/lo2/lo2-sample")
LOGS_DIR = LO2_ROOT / "logs"
OUTPUT_DIR = Path("datasets/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "lo2_external_log_dataset.csv"

KEYWORDS = {
    "error": r"\bERROR\b|\bError\b|\berror\b|Exception|exception|failed|Failed|FAILED",
    "warn": r"\bWARN\b|\bWarning\b|\bwarning\b",
    "http_400": r"\b400\b",
    "http_401": r"\b401\b",
    "http_404": r"\b404\b",
    "http_500": r"\b500\b",
    "invalid": r"invalid|Invalid|INVALID",
    "missing": r"missing|Missing|MISSING",
    "denied": r"denied|Denied|DENIED|unauthorized|Unauthorized",
}


def safe_read_text(file_path):
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def clean_service_name(log_file_name):
    name = log_file_name.replace(".log", "")
    name = name.replace("light-oauth2-", "")
    name = name.replace("-", "_")
    return name


def extract_status_code(scenario_name):
    match = re.search(r"_(400|401|404|500)(?:_|$)", scenario_name)
    if match:
        return int(match.group(1))

    if scenario_name == "correct":
        return 200

    return 0


def extract_features_from_scenario(run_id, scenario_dir):
    scenario_name = scenario_dir.name

    row = {
        "run_id": run_id,
        "scenario": scenario_name,
        "status_code_from_name": extract_status_code(scenario_name),
        "label": "normal" if scenario_name == "correct" else "anomaly",
        "binary_label": 0 if scenario_name == "correct" else 1,
    }

    total_size = 0
    total_lines = 0
    total_keyword_hits = 0
    total_log_files = 0

    log_files = list(scenario_dir.glob("*.log"))

    for log_file in log_files:
        total_log_files += 1
        service_name = clean_service_name(log_file.name)

        size_bytes = log_file.stat().st_size
        text = safe_read_text(log_file)
        lines = text.splitlines()

        row[f"{service_name}_log_size_bytes"] = size_bytes
        row[f"{service_name}_line_count"] = len(lines)

        total_size += size_bytes
        total_lines += len(lines)

        for keyword_name, pattern in KEYWORDS.items():
            count = len(re.findall(pattern, text))
            row[f"{service_name}_{keyword_name}_count"] = count
            total_keyword_hits += count

    row["total_log_files"] = total_log_files
    row["total_log_size_bytes"] = total_size
    row["total_line_count"] = total_lines
    row["total_keyword_hits"] = total_keyword_hits

    return row


def main():
    if not LOGS_DIR.exists():
        raise FileNotFoundError(f"Logs folder not found: {LOGS_DIR}")

    rows = []

    run_dirs = [p for p in LOGS_DIR.iterdir() if p.is_dir()]
    print(f"Found LO2 run folders: {len(run_dirs)}")

    for run_dir in run_dirs:
        run_id = run_dir.name
        scenario_dirs = [p for p in run_dir.iterdir() if p.is_dir()]

        for scenario_dir in scenario_dirs:
            row = extract_features_from_scenario(run_id, scenario_dir)
            rows.append(row)

    if not rows:
        raise RuntimeError("No LO2 log scenarios found.")

    df = pd.DataFrame(rows)
    df = df.fillna(0)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nLO2 external log dataset created successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nBinary label distribution:")
    print(df["binary_label"].value_counts())
    print("\nFirst 20 columns:")
    for col in list(df.columns)[:20]:
        print(col)


if __name__ == "__main__":
    main()