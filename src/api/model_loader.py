import os
from pathlib import Path
from typing import Tuple, Union
import joblib
import xgboost as xgb
from src.core.logger import logger

# BASE_DIR points to repository root (up 2 levels from src/core)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))

DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "model.json"
DEFAULT_PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"


class ArtifactLoader:
    def __init__(
        self,
        preprocessor_path: Union[str, Path] = DEFAULT_PREPROCESSOR_PATH,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    ):
        self.preprocessor_path = Path(preprocessor_path)
        self.model_path = Path(model_path)
        self.preprocessor = None
        self.model = None

    def load(self) -> Tuple[object, xgb.XGBClassifier]:
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor missing at: {self.preprocessor_path}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact missing at: {self.model_path}"
            )

        logger.info(f"[+] Loading preprocessor from {self.preprocessor_path}")
        self.preprocessor = joblib.load(self.preprocessor_path)

        logger.info(f"[+] Loading XGBoost model from {self.model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))

        return self.preprocessor, self.model


# Global singleton instance using dynamically resolved default paths
artifact_loader = ArtifactLoader()