import requests
import json
import sys
from typing import Dict, Any, Optional

def safe_request(method: str, url: str, json_data: Optional[Dict[str, Any]] = None, debug: bool = True) -> Dict[str, Any]:
    """
    リクエストを送信し、適切なエラーハンドリングとデバッグ情報を提供する関数
    
    Args:
        method: HTTPメソッド ('GET', 'POST', 'PUT', 'DELETE')
        url: リクエスト先のURL
        json_data: リクエストボディに含めるJSONデータ
        debug: デバッグ情報を出力するかどうか
        
    Returns:
        処理されたレスポンスのJSON
    """
    try:
        if debug:
            print(f"Sending {method} request to {url}")
            if json_data:
                print(f"Request data: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
        
        # リクエスト送信
        if method.upper() == 'GET':
            response = requests.get(url)
        elif method.upper() == 'POST':
            response = requests.post(url, json=json_data)
        elif method.upper() == 'PUT':
            response = requests.put(url, json=json_data)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, json=json_data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        # レスポンスのステータスコードを確認
        if debug:
            print(f"Response status code: {response.status_code}")
        
        # レスポンスが成功かどうかを確認
        response.raise_for_status()
        
        # レスポンスのテキスト内容を確認（デバッグ用）
        if debug:
            print(f"Response content type: {response.headers.get('Content-Type', 'unknown')}")
            print(f"Response content: {response.text[:1000]}..." if len(response.text) > 1000 else f"Response content: {response.text}")
        
        # JSONレスポンスの解析を試みる
        try:
            json_response = response.json()
            return json_response
        except ValueError as e:
            if debug:
                print(f"Failed to parse JSON response: {str(e)}")
                print(f"Raw response: {response.text}")
            raise ValueError(f"Invalid JSON response from server: {str(e)}")
        
    except requests.exceptions.RequestException as e:
        if debug:
            print(f"Request error: {str(e)}", file=sys.stderr)
            
            # エラーレスポンスの内容を表示
            if hasattr(e, 'response') and e.response is not None:
                print(f"Error status code: {e.response.status_code}", file=sys.stderr)
                print(f"Error response: {e.response.text}", file=sys.stderr)
        
        raise e

def example_usage():
    """使用例"""
    API_URL = "http://localhost:8000"
    
    # 学習リクエスト例
    train_payload = {
        "data": [
            {"Recency": 10, "History": 100, "Mens": 1, "Womens": 0, "Newbie": 0, "treatment": 1, "Conversion": 1},
            {"Recency": 5, "History": 200, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 0},
            {"Recency": 7, "History": 150, "Mens": 1, "Womens": 1, "Newbie": 1, "treatment": 1, "Conversion": 1},
            {"Recency": 3, "History": 300, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 1}
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
    
    # トレーニングリクエスト送信
    train_result = safe_request("POST", f"{API_URL}/train", train_payload)
    print("Train Result:", json.dumps(train_result, indent=2, ensure_ascii=False))
    
    # モデルパスの取得
    model_path = train_result.get("model_path")
    if not model_path:
        print("Error: No model_path in train response")
        return
        
    # 予測リクエスト例
    predict_payload = {
        "features": [
            {"Recency": 8, "History": 150, "Mens": 1, "Womens": 0, "Newbie": 1},
            {"Recency": 3, "History": 250, "Mens": 0, "Womens": 1, "Newbie": 0}
        ],
        "model_path": model_path
    }
    
    # 予測リクエスト送信
    predict_result = safe_request("POST", f"{API_URL}/predict", predict_payload)
    print("Predict Result:", json.dumps(predict_result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    example_usage()
