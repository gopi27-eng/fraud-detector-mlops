import argparse
import json
import os
from pathlib import Path
import sys
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.core.logger import logger
import xgboost as xgb
import yaml


def load_params(config_path: str = "config/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(config_path: str):
    
    config = load_params(config_path)
    data_cfg = config["data"]
    prep_cfg= config["preprocessing"]
    train_cfg = config["train"]
    eval_cfg = config["evaluate"]
    
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(train_cfg["experiment_name"])   
    
    # 2. Load Model & Feature Splits
    logger.info("[+] Loading model and feature sets for gate evaluation...")
    model_path = Path(train_cfg["model_path"])
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))

    train_df = pd.read_parquet(prep_cfg["train_features_path"])
    test_df = pd.read_parquet(prep_cfg["test_features_path"])

    target_col = data_cfg["target_col"]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values 
    
    
    # 3. Calculate Threshold-Agnostic Probabilities
    y_train_probs = clf.predict_proba(X_train)[:, 1]
    y_test_probs = clf.predict_proba(X_test)[:, 1]

    train_pr_auc = float(average_precision_score(y_train, y_train_probs))
    test_pr_auc = float(average_precision_score(y_test, y_test_probs))
    train_roc_auc = float(roc_auc_score(y_train, y_train_probs))
    test_roc_auc = float(roc_auc_score(y_test, y_test_probs))

    metrics_payload = {
        "train_roc_auc": round(train_roc_auc, 4),
        "test_roc_auc": round(test_roc_auc, 4),
        "train_pr_auc": round(train_pr_auc, 4),
        "test_pr_auc": round(test_pr_auc, 4),
    }
    
    # Persist metrics file for DVC/CI visibility
    metrics_path = Path(eval_cfg["metrics_output_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    logger.info(f"[+] Evaluation metrics saved to: {metrics_path}")

    logger.info("=" * 60)
    logger.info(f"Metrics Output: {metrics_payload}")
    logger.info("=" * 60)

    # 4. Production Gate Evaluation
    gate_failed = False
    failure_reasons = []

    if test_pr_auc < eval_cfg["min_test_pr_auc"]:
        gate_failed = True
        failure_reasons.append(
            f"Test PR-AUC ({test_pr_auc:.4f}) below minimum gate ({eval_cfg['min_test_pr_auc']})"
        )

    if test_roc_auc < eval_cfg["min_test_roc_auc"]:
        gate_failed = True
        failure_reasons.append(
            f"Test ROC-AUC ({test_roc_auc:.4f}) below minimum gate ({eval_cfg['min_test_roc_auc']})"
        )

    auc_drop = (train_roc_auc - test_roc_auc) / train_roc_auc
    if auc_drop > eval_cfg["max_auc_drop_ratio"]:
        gate_failed = True
        failure_reasons.append(
            f"ROC-AUC drop ({auc_drop:.2%}) exceeds allowed degradation threshold ({eval_cfg['max_auc_drop_ratio']:.2%})"
        )

    if gate_failed:
        logger.warning("[-] PRODUCTION QUALITY GATE: FAILED")
        for reason in failure_reasons:
            logger.info(f"    * {reason}")
        logger.warning("[-] Candidate model rejected. Halting downstream deployment.")
        sys.exit(1)
    else:
        logger.success("[+] PRODUCTION QUALITY GATE: PASSED. Candidate model promoted.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Gate Stage")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to params.yaml",
    )
    args = parser.parse_args()
    evaluate(args.config)