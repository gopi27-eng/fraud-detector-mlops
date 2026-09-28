# Real-Time Financial Fraud Detection: Production MLOps Engine

A production-grade, closed-loop MLOps pipeline designed to detect fraudulent transactions using an optimized XGBoost classifier. This system transitions model development into an automated Continuous Training (CT), Continuous Integration (CI), and Continuous Deployment (CD) lifecycle on AWS using DVC, Amazon S3, Amazon ECR, and Amazon EKS.

---

## Architecture Overview

```
       +------------------------------------------------------------+
       |                     GitHub Repository                      |
       |  (Config: config/params.yaml | Pipeline: GitHub Actions)   |
       +------------------------------------------------------------+
                                     |
               +---------------------+---------------------+
               |                     |                     |
               v                     v                     v
      +-----------------+   +-----------------+   +-----------------+
      | Continuous      |   | Continuous      |   | Continuous      |
      | Training (CT)   |   | Integration(CI) |   | Deployment (CD) |
      +-----------------+   +-----------------+   +-----------------+
               |                     |                     |
      [1] DVC Data Sync (S3) [5] Pytest Suite     [7] Build & Tag OCI
      [2] Ingest & Split         - /health Probe  [8] Push to Amazon ECR
      [3] Feature Engineering    - /predict Valid [9] Zero-Downtime
      [4] XGBoost & MLflow       - Lifespan Fixt.     Rollout to EKS
               |                                           |
               v                                           v
      +-----------------+                         +-----------------+
      | Trained Models  |                         | Amazon EKS Node |
      |  - model.json   |                         |  - maxSurge: 0  |
      |  - preprocessor |                         |  - FastAPI Pods |
      +-----------------+                         +-----------------+

```

---

## Key Features

* **Data Version Control (DVC)**: Raw financial transaction datasets tracked with `.dvc` metadata pointers and versioned remotely on Amazon S3.
* **Reproducible Pipeline Stages**: Modular retraining scripts (`01_ingest_and_split.py`, `02_preprocess.py`, `03_train.py`, `04_evaluate.py`) with centralized parameters managed in `config/params.yaml`.
* **Experiment Tracking**: Integrated with MLflow using an embedded SQLite tracking store (`sqlite:///mlflow.db`) to record metrics, hyperparameters, and model artifacts without external service dependencies during CI runner runs.
* **Strict Quality Gate**: Model evaluation computes ROC-AUC and PR-AUC with automated failure thresholds to prevent sub-par models from reaching downstream stages.
* **Robust API Testing**: Comprehensive `pytest` test suite utilizing FastAPI's `TestClient` with lifespan context fixtures to validate memory loading of preprocessors and inference endpoints.
* **Containerized Deployment**: Automated Docker builds pushed to Amazon Elastic Container Registry (ECR).
* **Zero-Downtime Kubernetes Updates**: Production deployment manifests configured with `maxSurge: 0` and `maxUnavailable: 1` on Amazon Elastic Kubernetes Service (EKS) to ensure resource stability on single-node clusters.

---

## Tech Stack

* **Machine Learning**: XGBoost, Scikit-learn, Pandas, PyArrow, NumPy
* **API & Serving**: FastAPI, Uvicorn, Pydantic, HTTPX
* **Tracking & Versioning**: MLflow, DVC, AWS S3
* **Containerization & Orchestration**: Docker, Kubernetes, Amazon EKS, Amazon ECR
* **CI/CD Orchestration**: GitHub Actions

---

## Repository Structure

```text
fraud-detector-mlops/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yaml      # Unified CT / CI / CD orchestration workflow
├── artifacts/
│   ├── metrics.json                 # Evaluation metrics output
│   ├── model.json                   # Serialized XGBoost model artifact
│   └── preprocessor.joblib          # Scikit-learn feature pipeline
├── config/
│   └── params.yaml                  # Central configuration and hyperparameters
├── data/
│   ├── raw/
│   │   └── raw_with_target.csv.dvc  # DVC tracking pointer
│   └── processed/                   # Split train/test Parquet datasets
├── k8s/
│   ├── deployment.yaml              # Kubernetes Deployment with rolling update strategy
│   └── service.yaml                 # LoadBalancer service manifest
├── src/
│   ├── api/
│   │   └── app.py                   # FastAPI service with lifespan event hooks
│   ├── core/
│   │   ├── logger.py                # Logging configuration
│   │   └── model_loader.py          # Dynamic artifact loader with path resolution
│   └── stages/
│       ├── 01_ingest_and_split.py   # Temporal train/test splitting
│       ├── 02_preprocess.py         # Financial feature engineering & encoding
│       ├── 03_train.py              # XGBoost training & MLflow logging
│       └── 04_evaluate.py           # Evaluation quality gate and metric calculation
├── tests/
│   ├── test_api.py                  # Endpoint assertions and lifecycle tests
│   └── test_schemas.py              # Request payload validation tests
├── Dockerfile                       # Multi-stage container definition
├── requirements-serve.txt           # Serving & inference dependencies
└── README.md

```

---

## Local Setup & Pipeline Execution

### 1. Prerequisites

* Python 3.11+
* AWS CLI configured with proper IAM permissions
* Docker & `kubectl`

### 2. Environment Setup

```bash
git clone https://github.com/gopi27-eng/fraud-detector-mlops.git
cd fraud-detector-mlops

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-serve.txt "dvc[s3]" pyarrow pytest httpx

```

### 3. Pull Data via DVC

```bash
dvc pull data/raw/raw_with_target.csv.dvc

```

### 4. Execute Retraining Stages Locally

```bash
export PYTHONPATH=.
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_ALLOW_FILE_STORE=true

python src/stages/01_ingest_and_split.py --config config/params.yaml
python src/stages/02_preprocess.py --config config/params.yaml
python src/stages/03_train.py --config config/params.yaml
python src/stages/04_evaluate.py --config config/params.yaml

```

### 5. Run Test Suite

```bash
pytest tests/ -v -s

```

### 6. Local Server Execution

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

```

---

## API Reference

### Health Check

* **Endpoint**: `GET /health`
* **Response**:

```json
{
  "status": "healthy",
  "model_loaded": true
}

```

### Predict Fraud

* **Endpoint**: `POST /predict`
* **Sample Payload**:

```json
{
  "step": 100,
  "type": "PAYMENT",
  "amount": 250.75,
  "oldbalanceOrg": 5000.00,
  "newbalanceOrig": 4749.25,
  "oldbalanceDest": 1000.00,
  "newbalanceDest": 1250.75
}

```

* **Sample Response**:

```json
{
  "is_fraud": 0,
  "fraud_probability": 0.0142,
  "threshold_applied": 0.5
}

```

---

## CI/CD/CT Pipeline Details

The GitHub Actions workflow (`.github/workflows/mlops-pipeline.yaml`) executes sequentially on each push to `main`:

1. **Continuous Training (CT)**: Authenticates to AWS, pulls raw data with DVC, runs data preparation, fits feature transformers, trains XGBoost, logs to SQLite MLflow, and saves validated artifacts (`artifacts/model.json` and `artifacts/preprocessor.joblib`).
2. **Continuous Integration (CI)**: Downloads artifacts from the CT step, verifies directory layout, and executes `pytest` using client lifespan contexts to test API schema handling and model scoring.
3. **Continuous Deployment (CD)**: Builds the container image, tags it with `$GITHUB_SHA` and `latest`, pushes it to Amazon ECR, and executes a zero-downtime rolling update across the Amazon EKS cluster (`maxSurge: 0`, `maxUnavailable: 1`).