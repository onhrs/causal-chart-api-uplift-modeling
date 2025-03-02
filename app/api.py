from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import pickle
import os
import sys
from typing import List, Dict, Any
import logging
import traceback
from pathlib import Path

from app.schemas import (
    TrainRequest, TrainResponse, 
    PredictRequest, PredictResponse,
    ModelInfoResponse, AvailableModelsResponse
)
from models import train_uplift_model, predict_uplift
from utils.config import MODEL_DIR, API_TITLE, API_DESCRIPTION, API_VERSION

# FastAPIアプリケーションの作成
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデル保存ディレクトリの確認
os.makedirs(MODEL_DIR, exist_ok=True)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("uplift-api")

# グローバル例外ハンドラーを追加
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    すべての未処理例外をハンドリングして適切なJSONレスポンスを返す
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

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
        model_filename = f"{model_type}_{model_info['timestamp']}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)
        
        # モデル情報の保存
        with open(model_path, "wb") as f:
            pickle.dump(model_info, f)
        
        return {
            "message": f"{model_type} model trained successfully",
            "model_path": model_filename,  # ファイル名のみを返す
            "model_info": {
                "model_type": model_type,
                "features": request.features,
                "metrics": model_info.get("metrics", {}),
                "timestamp": model_info["timestamp"]
            }
        }
    
    except Exception as e:
        logging.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    アップリフトモデルを使用して予測を行うエンドポイント
    """
    try:
        logger.info(f"Prediction request received with model_path: {request.model_path}")
        
        # モデルの読み込み
        model_path = request.model_path
        
        # ファイル名だけが提供された場合、MODEL_DIRと結合
        if not os.path.isabs(model_path):
            model_path = os.path.join(MODEL_DIR, model_path)
            logger.info(f"Resolved model path to: {model_path}")
        
        # モデルファイルの存在確認
        if not os.path.exists(model_path):
            error_msg = f"Model not found at {model_path}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        # モデルの読み込み
        try:
            with open(model_path, "rb") as f:
                model_info = pickle.load(f)
            logger.info(f"Model loaded successfully: {model_info['model_type']}")
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
        # 特徴量データをDataFrameに変換
        features_df = pd.DataFrame(request.features)
        logger.info(f"Received features: {features_df.columns.tolist()}")
        
        # 特徴量の確認
        missing_features = [f for f in model_info["features"] if f not in features_df.columns]
        if missing_features:
            error_msg = f"Missing features in input data: {missing_features}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 予測に必要な特徴量のみを選択
        features_df = features_df[model_info["features"]]
        
        # 予測の実行
        try:
            predictions = predict_uplift(model_info, features_df)
            logger.info(f"Predictions generated successfully: {len(predictions)} predictions")
            logger.debug(f"Raw predictions: {predictions}")
            
            # 予測値が入れ子になっている場合はフラット化する
            if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
                predictions = [item for sublist in predictions for item in sublist]
                logger.info("Flattened nested predictions")
            
            # numpy配列をPythonのネイティブリストに変換
            if hasattr(predictions, 'tolist'):
                predictions = predictions.tolist()
            
            # 予測値が数値のリストであることを確認
            for i, pred in enumerate(predictions):
                if not isinstance(pred, (int, float)):
                    logger.warning(f"Non-numeric prediction found at index {i}: {pred}, converting to float")
                    predictions[i] = float(pred)
            
            # レスポンスデータの準備
            response_data = {
                "predictions": predictions,  # これでフラットなリストになっているはず
                "model_type": model_info["model_type"]
            }
            logger.info(f"Returning predictions with model_type: {model_info['model_type']}")
            return response_data
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
    except HTTPException:
        # HTTPExceptionはそのまま再発行
        raise
    except Exception as e:
        error_msg = f"Unexpected error during prediction: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/models", response_model=AvailableModelsResponse)
async def list_models():
    """
    利用可能なモデルの一覧を取得するエンドポイント
    """
    try:
        models = []
        if os.path.exists(MODEL_DIR):
            for filename in os.listdir(MODEL_DIR):
                if filename.endswith(".pkl"):
                    model_full_path = os.path.join(MODEL_DIR, filename)
                    with open(model_full_path, "rb") as f:
                        model_info = pickle.load(f)
                    
                    models.append({
                        "model_path": filename,  # ファイル名のみを返す
                        "model_type": model_info["model_type"],
                        "features": model_info["features"],
                        "timestamp": model_info.get("timestamp", "unknown"),
                        "metrics": model_info.get("metrics", {})
                    })
        
        return {"models": models}
    
    except Exception as e:
        logging.error(f"List models error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/model/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str):
    """
    特定のモデルの詳細情報を取得するエンドポイント
    """
    try:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        
        with open(model_path, "rb") as f:
            model_info = pickle.load(f)
        
        return {
            "model_path": model_name,
            "model_type": model_info["model_type"],
            "features": model_info["features"],
            "timestamp": model_info.get("timestamp", "unknown"),
            "metrics": model_info.get("metrics", {})
        }
    
    except HTTPException as e:
        raise
    except Exception as e:
        logging.error(f"Get model info error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
