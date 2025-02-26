# schemas.py
from pydantic import BaseModel
from typing import List, Optional

class DataPoint(BaseModel):
    """Represents a single training example."""
    features: List[float]
    label: float

class TrainRequest(BaseModel):
    """Request body for /train endpoint."""
    data: List[DataPoint]    # A list of data points to train on

class TrainResponse(BaseModel):
    """Response from /train, containing status and model info."""
    message: str
    model_path: str

class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    features: List[List[float]]  # List of feature vectors for prediction

class PredictResponse(BaseModel):
    """Response from /predict, containing the predictions."""
    predictions: List[float]
