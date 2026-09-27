import argparse
from pathlib import Path
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.core.logger import logger
import xgboost as xgb
import yaml




def load_params(config_path: str = "config/params.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
    
def train(config_path:str):
    
    config = load_params(config_path)
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    train_cfg = config["train"]
    
    
    logger.info(f"[+] Loading engineered feature sets...")
    
    train_df = pd.read_parquet(prep_cfg["train_features_path"])
    test_df = pd.read_parquet(prep_cfg["test_features_path"])
    
    target_col = data_cfg["target_col"]
    
    
    X_train = train_df.drop(columns = target_col)
    y_train = train_df[target_col].values
    
    
    
    X_test = test_df.drop(columns = target_col)
    y_test = test_df[target_col].values
    
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = float(num_neg / num_pos)
    logger.info(f"[+] Class distribution in train: {num_neg} negative, {num_pos} positive")
    logger.info(f"[+] Computed scale_pos_weight: {scale_pos_weight:.2f}")
    
    
    experiment_name = train_cfg["experiment_name"] 
    mlflow.set_experiment(experiment_name)
    
    
    model_dir = Path(train_cfg["model_dir"])
    model_dir.mkdir(parents= True, exist_ok= True)
    model_path = Path(train_cfg["model_path"])
    
    hyperparams = train_cfg["params"].copy()
    
    
    
    with mlflow.start_run(run_name="xgboost_baseline"):
        # Log hyperparams and class weights
        mlflow.log_params(hyperparams)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        logger.info("[+] Training XGBoost Classifier...")
        clf = xgb.XGBClassifier(
            **hyperparams,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train)
        
    # 4. Predict probabilities for threshold-agnostic metric evaluation
        y_train_probs = clf.predict_proba(X_train)[:, 1]
        y_test_probs = clf.predict_proba(X_test)[:, 1]

        # PR-AUC (Average Precision) and ROC-AUC
        train_pr_auc = average_precision_score(y_train, y_train_probs)
        test_pr_auc = average_precision_score(y_test, y_test_probs)
        train_roc_auc = roc_auc_score(y_train, y_train_probs)
        test_roc_auc = roc_auc_score(y_test, y_test_probs)
        
        
    # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "train_pr_auc": train_pr_auc,
                "test_pr_auc": test_pr_auc,
                "train_roc_auc": train_roc_auc,
                "test_roc_auc": test_roc_auc,
            }
        )

        logger.info("=" * 60)
        logger.info(f"Train ROC-AUC: {train_roc_auc:.4f} | Train PR-AUC: {train_pr_auc:.4f}")
        logger.info(f"Test  ROC-AUC: {test_roc_auc:.4f} | Test  PR-AUC: {test_pr_auc:.4f}")
        logger.info("=" * 60)

        # 5. Persist Model Artifact locally and via MLflow
        clf.save_model(model_path)
        mlflow.xgboost.log_model(xgb_model=clf, name="model") 
        logger.info(f"[+] Model artifact persisted to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Stage")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to params.yaml",
    )
    args = parser.parse_args()
    train(args.config)