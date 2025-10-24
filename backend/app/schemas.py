from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime

class TrainRequest(BaseModel):
    task_type: str  # "classification" 或 "regression"
    dataset_name: str
    model_name: str
    split_ratio: float = 0.7
    use_pca: bool = False
    pca_params: Dict[str, Any] = {}
    model_params: Dict[str, Any] = {}

class TrainResult(BaseModel):
    task_type: str
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    test_predictions: List[Any]
    test_targets: List[Any]
    train_loss_history: Optional[List[float]] = None
    timestamp: str

class ConfigResponse(BaseModel):
    TASK_MODELS: Dict[str, List[str]]
    TASK_DATASETS: Dict[str, List[str]]
    CLASSIFICATION_METRICS: List[str]
    REGRESSION_METRICS: List[str]
    PCA_PARAMS: Dict[str, Any]
    BASELINE_TIMES: Dict[str, float]
    MODEL_PARAMS: Dict[str, Dict[str, Any]]