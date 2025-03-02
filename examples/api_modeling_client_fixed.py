import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from causalml.metrics import qini_score, plot_qini
from sklearn.model_selection import train_test_split

# APIエンドポイント
API_URL = "http://localhost:8000"

def main():
    # 1. データ読み込み
    try:
        df = pd.read_csv("hillstrom.csv")
    except FileNotFoundError:
        print("hillstrom.csvが見つかりません。カレントディレクトリにファイルが存在するか確認してください。")
        return
    
    print(f"データ件数: {len(df)}")
    
    # 2. 前処理
    df.columns = [col.capitalize() for col in df.columns]
    df['treatment'] = (df['Segment'] != 'No E-Mail').astype(int)
    
    # 分析で使う特徴量
    features = ['Recency', 'History', 'Mens', 'Womens', 'Newbie']
    
    # データ分割 (各モデル間で同じテストデータを使う)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    print(f"訓練データ: {len(train_df)}件, テストデータ: {len(test_df)}件")
    
    # 3. トレーニング用データの準備とAPIリクエスト
    train_data = train_df.to_dict('records')
    
    # 各モデルタイプをトレーニング
    model_types = ["s_learner", "t_learner", "x_learner", "r_learner", "causal_tree", "uplift_rf"]
    model_paths = {}
    
    for model_type in model_types:
        print(f"\nトレーニング: {model_type}")
        train_payload = {
            "data": train_data,
            "features": features,
            "treatment_col": "treatment",
            "outcome_col": "Conversion",
            "model_type": model_type,
            "model_params": {
                "n_estimators": 100,
                "max_depth": 5
            }
        }
        
        response = requests.post(f"{API_URL}/train", json=train_payload)
        if response.status_code == 200:
            result = response.json()
            model_path = result["model_path"]
            model_paths[model_type] = model_path
            metrics = result["model_info"]["metrics"]
            print(f"  成功: {model_type}")
            print(f"  実測ATE: {metrics.get('actual_ate'):.5f}")
            print(f"  推定ATE: {metrics.get('estimated_ate'):.5f}")
        else:
            print(f"  失敗: {response.text}")
    
    # 4. テスト用データの準備と予測
    test_features = test_df[features].to_dict('records')
    
    # 5. 各モデルで予測を実行
    predictions = {}
    
    for model_type, model_path in model_paths.items():
        print(f"\n予測: {model_type}")
        predict_payload = {
            "features": test_features,
            "model_path": model_path
        }
        
        response = requests.post(f"{API_URL}/predict", json=predict_payload)
        if response.status_code == 200:
            result = response.json()
            preds = result["predictions"]
            print(f"  成功: {len(preds)}件の予測")
            
            # 予測結果のサイズ修正
            if len(preds) != len(test_df):
                print(f"  警告: 予測サイズ({len(preds)})がテストデータサイズ({len(test_df)})と異なります")
                if len(preds) == 2 * len(test_df):
                    # causal_treeは1件の予測に対して2つの値を返す場合がある
                    # 1つおきに値を取る
                    preds = preds[::2]
                    print(f"  2倍のサイズのため1つおきに選択: {len(preds)}件")
                elif len(preds) > len(test_df):
                    # サイズが大きい場合は切り詰め
                    preds = preds[:len(test_df)]
                    print(f"  サイズが大きいため切り詰め: {len(preds)}件")
                else:
                    # サイズが小さい場合は平均値で埋める
                    mean_val = np.mean(preds)
                    preds = preds + [mean_val] * (len(test_df) - len(preds))
                    print(f"  サイズが小さいため平均値で埋め: {len(preds)}件")
            
            predictions[model_type] = preds
        else:
            print(f"  失敗: {response.text}")
    
    # 6. Qiniスコアの計算
    y_test = test_df['Conversion'].values
    w_test = test_df['treatment'].values
    
    print("\nQiniスコア:")
    all_scores = {}
    
    for model_type, preds in predictions.items():
        try:
            # データフレーム作成
            df_eval = pd.DataFrame({
                'y': y_test,
                'w': w_test,
                'tau': preds
            })
            
            # データの値チェック
            if np.isnan(df_eval['tau']).any():
                print(f"  {model_type}: NaN値があります。平均値で置換します。")
                mean_val = np.nanmean(df_eval['tau'])
                df_eval['tau'] = df_eval['tau'].fillna(mean_val)
            
            # スコア計算
            score = qini_score(df_eval, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True)
            all_scores[model_type] = score
            print(f"  {model_type}: {score:.4f}")
        except Exception as e:
            print(f"  {model_type}のQini計算エラー: {e}")
            
    # 7. 最良モデルのQiniカーブ可視化
    if all_scores:
        best_model = max(all_scores, key=all_scores.get)
        print(f"\n最良モデル: {best_model} (スコア: {all_scores[best_model]:.4f})")
        
        try:
            # Qiniカーブのプロット
            df_qini = pd.DataFrame({
                'y': y_test,
                'w': w_test,
                'pred': predictions[best_model]
            })
            
            plt.figure(figsize=(10, 6))
            plot_qini(df_qini, outcome_col='y', treatment_col='w', prediction_col='pred')
            plt.title(f"Qini Curve - {best_model}")
            plt.savefig("qini_curve_api.png")
            print("Qiniカーブを 'qini_curve_api.png' として保存しました")
        except Exception as e:
            print(f"Qiniカーブ作成エラー: {e}")

if __name__ == "__main__":
    main()
