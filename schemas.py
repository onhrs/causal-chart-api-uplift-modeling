# schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class DataPoint(BaseModel):
    """Represents a single training example."""
    features: List[float]
    label: float

class TrainRequest(BaseModel):
    """Request body for /train endpoint."""
    data: List[Dict[str, Any]]  # データポイントのリスト
    features: List[str]  # 使用する特徴量のリスト
    treatment_col: str  # 処置フラグを含む列名
    outcome_col: str  # 結果変数を含む列名
    model_type: str  # モデルタイプ ('s_learner', 't_learner', 'x_learner', 'r_learner', 'causal_tree', 'uplift_rf')
    model_params: Optional[Dict[str, Any]] = None  # モデルのパラメータ

class ModelInfo(BaseModel):
    """モデル情報を表すスキーマ"""
    model_type: str
    features: List[str]
    metrics: Dict[str, float]
    timestamp: str

class TrainResponse(BaseModel):
    """Response from /train, containing status and model info."""
    message: str
    model_path: str
    model_info: ModelInfo

class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    features: List[Dict[str, Any]]  # 予測に使用する特徴量データ
    model_path: str  # 使用するモデルのパス

class PredictResponse(BaseModel):
    """Response from /predict, containing the predictions."""
    predictions: List[float]
    model_type: str

class ModelInfoResponse(BaseModel):
    """モデル情報のレスポンス"""
    model_path: str
    model_type: str
    features: List[str]
    timestamp: str
    metrics: Dict[str, float]

class AvailableModelsResponse(BaseModel):
    """利用可能なモデルのリスト"""
    models: List[ModelInfoResponse]
