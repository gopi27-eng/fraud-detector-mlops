import argparse
import json
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from xgboost import XGBClassifier


def train_model(config_path: str):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # MLflow Tracking Configuration
    # Uses environment variable if provided (e.g. in GitHub Actions), otherwise local directory
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config.get("train", {}).get(
        "experiment_name", "fraud-detector-training"
    )
    mlflow.set_experiment(experiment_name)

    # Input/Output paths
    processed_train_path = config.get("preprocess", {}).get(
        "processed_train_path", "data/processed/train_features.parquet"
    )
    artifacts_dir = Path(config.get("train", {}).get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_output_path = artifacts_dir / "model.json"

    print(f"[+] Loading processed training data from {processed_train_path}...")
    train_df = pd.read_parquet(processed_train_path)

    target_col = config.get("data", {}).get("target_column", "isFraud")
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # Hyperparameters
    params = config.get("train", {}).get(
        "params",
        {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 10,
            "random_state": 42,
            "eval_metric": "logloss",
        },
    )

    with mlflow.start_run(run_name="retrain_run"):
        print("[+] Training XGBoost Classifier...")
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Save model artifact in JSON format required for deployment
        print(f"[+] Saving model artifact to {model_output_path}...")
        model.save_model(str(model_output_path))

        # Log artifact to MLflow
        mlflow.xgboost.log_model(model, artifact_path="model")

        # Save parameters locally alongside model for auditing
        params_file = artifacts_dir / "train_params.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=4)

        print("[+] Training complete. Artifacts successfully written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 03: Model Training")
    parser.add_argument(
        "--config",
        default="params.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    train_model(args.config)