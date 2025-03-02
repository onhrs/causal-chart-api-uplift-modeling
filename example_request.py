import requests
import json
import pandas as pd
API_URL = "http://localhost:8000"

# 拡張したトレーニングデータ
train_payload = {
    "data": [
        {"Recency": 10, "History": 100, "Mens": 1, "Womens": 0, "Newbie": 0, "treatment": 1, "Conversion": 1},
        {"Recency": 12, "History": 120, "Mens": 1, "Womens": 0, "Newbie": 0, "treatment": 1, "Conversion": 0},
        {"Recency": 8, "History": 90, "Mens": 1, "Womens": 0, "Newbie": 0, "treatment": 0, "Conversion": 0},
        {"Recency": 15, "History": 80, "Mens": 1, "Womens": 0, "Newbie": 0, "treatment": 0, "Conversion": 1},
        {"Recency": 5, "History": 200, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 0},
        {"Recency": 7, "History": 180, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 1},
        {"Recency": 3, "History": 210, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 1, "Conversion": 0},
        {"Recency": 4, "History": 220, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 1, "Conversion": 1},
        {"Recency": 20, "History": 50, "Mens": 0, "Womens": 0, "Newbie": 1, "treatment": 0, "Conversion": 0},
        {"Recency": 25, "History": 30, "Mens": 0, "Womens": 0, "Newbie": 1, "treatment": 1, "Conversion": 1}
    ],
    "features": ["Recency", "History", "Mens", "Womens", "Newbie"],
    "treatment_col": "treatment",
    "outcome_col": "Conversion",
    "model_type": "s_learner",
    "model_params": {
        "n_estimators": 100,
        "max_depth": 5
    }
}

# 1. モデルのトレーニング
response = requests.post(f"{API_URL}/train", json=train_payload)
train_result = response.json()
print("Train Result:", json.dumps(train_result, indent=2, ensure_ascii=False))

# トレーニング済みモデルのパスを取得
model_path = train_result["model_path"]

# 2. 予測の実行
predict_payload = {
    "features": [
        {"Recency": 8, "History": 150, "Mens": 1, "Womens": 0, "Newbie": 1},
        {"Recency": 3, "History": 250, "Mens": 0, "Womens": 1, "Newbie": 0}
    ],
    "model_path": model_path
}

response = requests.post(f"{API_URL}/predict", json=predict_payload)
predict_result = response.json()
print("Predict Result:", json.dumps(predict_result, indent=2, ensure_ascii=False))
