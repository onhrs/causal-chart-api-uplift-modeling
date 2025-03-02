import sys
import os
import json
import requests

# クライアントユーティリティを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from client_utils.debug_helpers import safe_request

def test_api():
    # APIエンドポイントの設定
    API_URL = "http://localhost:8000"
    
    # 1. APIが動作しているか確認
    try:
        root_response = safe_request("GET", f"{API_URL}/")
        print("API is running!")
    except Exception as e:
        print(f"Error: API is not running or not accessible: {str(e)}")
        return
    
    # 2. モデルをトレーニング
    train_payload = {
        "data": [
            {"Recency": 10, "History": 100, "Mens": 1, "Womens": 0, "Newbie": 0, "treatment": 1, "Conversion": 1},
            {"Recency": 5, "History": 200, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 0},
            {"Recency": 7, "History": 150, "Mens": 1, "Womens": 1, "Newbie": 1, "treatment": 1, "Conversion": 1},
            {"Recency": 3, "History": 300, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 0},
            {"Recency": 12, "History": 120, "Mens": 1, "Womens": 0, "Newbie": 1, "treatment": 1, "Conversion": 1},
            {"Recency": 4, "History": 80, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 1}
        ],
        "features": ["Recency", "History", "Mens", "Womens", "Newbie"],
        "treatment_col": "treatment",
        "outcome_col": "Conversion",
        "model_type": "s_learner",
        "model_params": {
            "n_estimators": 50,
            "max_depth": 3
        }
    }
    
    try:
        train_result = safe_request("POST", f"{API_URL}/train", train_payload)
        print("Train Result:", json.dumps(train_result, indent=2, ensure_ascii=False))
        
        # モデルパスの取得
        model_path = train_result.get("model_path")
        if not model_path:
            print("Error: No model_path in train response")
            return
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return
    
    # 3. 利用可能なモデル一覧を確認
    try:
        models_response = safe_request("GET", f"{API_URL}/models")
        print("Available models:", json.dumps(models_response, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Failed to get models: {str(e)}")
    
    # 4. 予測を実行
    predict_payload = {
        "features": [
            {"Recency": 8, "History": 150, "Mens": 1, "Womens": 0, "Newbie": 1},
            {"Recency": 3, "History": 250, "Mens": 0, "Womens": 1, "Newbie": 0}
        ],
        "model_path": model_path
    }
    
    try:
        # 直接requestsを使用して詳細情報の確認
        print("\nSending prediction request with raw requests for debugging:")
        response = requests.post(f"{API_URL}/predict", json=predict_payload)
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response content: {response.text}")
        
        # safe_requestを使用した予測
        print("\nSending prediction request with safe_request:")
        predict_result = safe_request("POST", f"{API_URL}/predict", predict_payload)
        print("Predict Result:", json.dumps(predict_result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    test_api()
