from contextlib import asynccontextmanager
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, status
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.api.model_loader import artifact_loader
from src.schemas.transaction import TransactionPayload

ml_artifacts: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load heavy model binaries once into process memory
    preprocessor, model = artifact_loader.load()
    ml_artifacts["preprocessor"] = preprocessor
    ml_artifacts["model"] = model
    yield
    # Teardown
    ml_artifacts.clear()


app = FastAPI(
    title="High-Risk Financial Transaction Anomaly Predictor",
    description="Production REST API serving fraud probabilities using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="Decision flag based on threshold")
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score"
    )
    model_version: str = Field(default="1.0.0")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if "model" not in ml_artifacts or "preprocessor" not in ml_artifacts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifacts not loaded",
        )
    return {"status": "healthy", "model_version": "1.0.0"}


@app.post(
    "/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK
)
async def predict(payload: TransactionPayload):
    try:
        # 1. Convert payload to DataFrame
        raw_dict = payload.model_dump()
        df = pd.DataFrame([raw_dict])

        # 2. Replicate feature engineering identically
        df["errorBalanceOrig"] = (
            df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
        )
        df["errorBalanceDest"] = (
            df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
        )

        # 3. Transform via preprocessor
        preprocessor = ml_artifacts["preprocessor"]
        model = ml_artifacts["model"]

        transformed_features = preprocessor.transform(df)

        # 4. Predict probability for positive class (fraud)
        probs = model.predict_proba(transformed_features)
        fraud_prob = float(probs[0, 1])

        # 5. Apply operational decision threshold
        is_fraud = fraud_prob >= 0.50

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_prob, 4),
            model_version="1.0.0",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(e)}",
        )