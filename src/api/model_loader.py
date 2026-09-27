from pathlib import Path
from src.core.logger import logger
from typing import Tuple
import joblib
import xgboost as xgb


class ArtifactLoader:
    def __init__(
        self,
        preprocessor_path: str = "artifacts/preprocessor.joblib",
        model_path: str = "artifacts/model.json",
    ):
        self.preprocessor_path = Path(preprocessor_path)
        self.model_path = Path(model_path)
        self.preprocesser = None
        self.model= None
        
    def load(self) -> Tuple[object, xgb.XGBClassifier]:
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor missing at: {self.preprocessor_path}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact missing at: {self.model_path}")

        logger.info(f"[+] Loading preprocessor from {self.preprocessor_path}")
        self.preprocessor = joblib.load(self.preprocessor_path)

        logger.info(f"[+] Loading XGBoost model from {self.model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))

        return self.preprocessor, self.model


# Global singleton instance
artifact_loader = ArtifactLoader()
    