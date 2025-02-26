# app.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from schemas import (
    TrainRequest, TrainResponse, 
    PredictRequest, PredictResponse,
    ModelInfoResponse, AvailableModelsResponse
)
from model import train_uplift_model, predict_uplift

app = FastAPI(title="Uplift Modeling API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデル保存ディレクトリの確認
os.makedirs("models", exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to Uplift Modeling API"}

@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    アップリフトモデルをトレーニングするエンドポイント
    """
    try:
        # リクエストからデータフレームを作成
        df = pd.DataFrame(request.data)
        
        # モデルタイプの取得
        model_type = request.model_type
        
        # モデルのトレーニング
        model_info = train_uplift_model(
            df=df,
            features=request.features,
            treatment_col=request.treatment_col,
            outcome_col=request.outcome_col,
            model_type=model_type,
            model_params=request.model_params
        )
        
        # モデルの保存パス
        model_path = f"models/{model_type}_{model_info['timestamp']}.pkl"
        
        # モデル情報の保存
        with open(model_path, "wb") as f:
            pickle.dump(model_info, f)
        
        return {
            "message": f"{model_type} model trained successfully",
            "model_path": model_path,
            "model_info": {
                "model_type": model_type,
                "features": request.features,
                "metrics": model_info.get("metrics", {}),
                "timestamp": model_info["timestamp"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    アップリフトモデルを使用して予測を行うエンドポイント
    """
    try:
        # モデルの読み込み
        model_path = request.model_path
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")
        
        with open(model_path, "rb") as f:
            model_info = pickle.load(f)
        
        # 特徴量データをDataFrameに変換
        features_df = pd.DataFrame(request.features, columns=model_info["features"])
        
        # 予測の実行
        predictions = predict_uplift(model_info, features_df)
        
        return {
            "predictions": predictions.tolist(),
            "model_type": model_info["model_type"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models", response_model=AvailableModelsResponse)
async def list_models():
    """
    利用可能なモデルの一覧を取得するエンドポイント
    """
    try:
        models = []
        for filename in os.listdir("models"):
            if filename.endswith(".pkl"):
                model_path = os.path.join("models", filename)
                with open(model_path, "rb") as f:
                    model_info = pickle.load(f)
                
                models.append({
                    "model_path": model_path,
                    "model_type": model_info["model_type"],
                    "features": model_info["features"],
                    "timestamp": model_info.get("timestamp", "unknown"),
                    "metrics": model_info.get("metrics", {})
                })
        
        return {"models": models}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/model/{model_path:path}", response_model=ModelInfoResponse)
async def get_model_info(model_path: str):
    """
    特定のモデルの詳細情報を取得するエンドポイント
    """
    try:
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")
        
        with open(model_path, "rb") as f:
            model_info = pickle.load(f)
        
        return {
            "model_path": model_path,
            "model_type": model_info["model_type"],
            "features": model_info["features"],
            "timestamp": model_info.get("timestamp", "unknown"),
            "metrics": model_info.get("metrics", {})
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
