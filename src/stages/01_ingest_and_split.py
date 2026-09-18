import argparse
from pathlib import Path
from logs.pipeline_setup_log import log_setup
import pandas as pd
import yaml

logger = log_setup()

def load_params(config_path: str = "config/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ingest_and_split(config_path: str):
    config = load_params(config_path)
    data_cfg = config["data"]

    raw_path = Path(data_cfg["raw_data_path"])
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[+] Loading raw dataset from: {raw_path}")
    
    df = pd.read_csv(raw_path)
    
    # Fixed spelling: df.columns (not df.columms)
    drop_cols = [col for col in data_cfg["drop_cols"] if col in df.columns]
    if drop_cols:
        logger.info(f"[+] Dropping non-predictive columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
        
    split_col = data_cfg["temporal_split_column"]
    test_ratio = data_cfg["test_split_ratio"]
    # Fixed: use target_col key, not isFraud
    target_col = data_cfg["target_col"]

    step_cutoff = df[split_col].quantile(1.0 - test_ratio)
    logger.info(f"[+] Temporal cutoff on '{split_col}': {step_cutoff} (Ratio: {test_ratio*100}%)")        
    
    train_df = df[df[split_col] <= step_cutoff].copy()
    test_df = df[df[split_col] > step_cutoff].copy()
    
    # 3. Sanity metrics & temporal integrity checks
    train_fraud_rate = (train_df[target_col].mean()) * 100
    test_fraud_rate = (test_df[target_col].mean()) * 100

    logger.info("=" * 60)
    logger.info(f"Total Records: {len(df):,}")
    logger.info(
        f"Train Records: {len(train_df):,} | Fraud Rate: {train_fraud_rate:.4f}% | Step Range: [{train_df[split_col].min()} - {train_df[split_col].max()}]"
    )
    logger.info(
        f"Test Records:  {len(test_df):,} | Fraud Rate: {test_fraud_rate:.4f}% | Step Range: [{test_df[split_col].min()} - {test_df[split_col].max()}]"
    )
    logger.info("=" * 60)

    # Defensive check: ensure zero temporal overlap
    assert train_df[split_col].max() <= test_df[split_col].min(), (
        "Temporal integrity error: Train steps overlap with test steps."
    )
    
    # 4. Save to Parquet format (preserves schema & types)
    train_path = Path(data_cfg["train_path"])
    test_path = Path(data_cfg["test_path"])

    train_df.to_parquet(train_path, index=False, engine="pyarrow")
    test_df.to_parquet(test_path, index=False, engine="pyarrow")
    print(f"[+] Saved train dataset to: {train_path}")
    print(f"[+] Saved test dataset to:  {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest and temporal split stage"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to params.yaml",
    )
    args = parser.parse_args()
    ingest_and_split(args.config)
