import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.core.logger import logger
import yaml


def load_params(config_path: str = "config/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering applied identically to batch or streaming data."""
    df = df.copy()
    # Accounting balance discrepancies
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def preprocess(config_path: str):
    config = load_params(config_path)
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]

    artifacts_dir = Path(prep_cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[+] Loading processed splits...")
    train_df = pd.read_parquet(data_cfg["train_path"])
    test_df = pd.read_parquet(data_cfg["test_path"])

    target_col = data_cfg["target_col"]

    # 1. Feature Engineering
    logger.info("[+] Applying financial domain feature engineering...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    cat_cols = prep_cfg["categorical_features"]
    num_cols = prep_cfg["numeric_features"]

    # 2. Build Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    # 3. Fit strictly on train; transform both
    logger.info("[+] Fitting preprocessor exclusively on train split...")
    X_train_trans = preprocessor.fit_transform(train_df)
    X_test_trans = preprocessor.transform(test_df)

    # Retrieve explicit feature names out of the transformer
    feature_names = preprocessor.get_feature_names_out()

    train_features_df = pd.DataFrame(X_train_trans, columns=feature_names)
    train_features_df[target_col] = y_train

    test_features_df = pd.DataFrame(X_test_trans, columns=feature_names)
    test_features_df[target_col] = y_test

    # 4. Save artifacts and preprocessed splits
    preprocessor_path = Path(prep_cfg["preprocessor_path"])
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"[+] Saved fitted transformer to: {preprocessor_path}")

    train_out = Path(prep_cfg["train_features_path"])
    test_out = Path(prep_cfg["test_features_path"])

    train_features_df.to_parquet(train_out, index=False, engine="pyarrow")
    test_features_df.to_parquet(test_out, index=False, engine="pyarrow")

    logger.info(f"[+] Transformed Train Shape: {train_features_df.shape}")
    logger.info(f"[+] Transformed Test Shape:  {test_features_df.shape}")
    logger.info(f"[+] Generated Features: {list(feature_names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and Feature Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to params.yaml",
    )
    args = parser.parse_args()
    preprocess(args.config)