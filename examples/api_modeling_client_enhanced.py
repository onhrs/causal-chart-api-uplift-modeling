import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import calculate_uplift_metrics, plot_uplift_curve

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
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['treatment'])
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
            
            # R-learnerがときに極端な値を出すので、異常値を修正
            if model_type == 'r_learner':
                outliers = np.abs(preds) > 1
                if np.any(outliers):
                    outlier_count = np.sum(outliers)
                    print(f"  警告: {outlier_count}件の異常値を検出")
                    # 中央値で置き換え
                    preds = np.array(preds)
                    median = np.median(preds[~outliers]) if np.sum(~outliers) > 0 else 0
                    preds[outliers] = median
                    print(f"  異常値を中央値({median:.5f})で置換")
                    preds = preds.tolist()
            
            predictions[model_type] = preds
        else:
            print(f"  失敗: {response.text}")
    
    # 6. 独自実装のアップリフト評価メトリクスで評価
    y_test = test_df['Conversion'].values
    w_test = test_df['treatment'].values
    
    print("\n独自実装アップリフトメトリクス:")
    all_scores = {}
    
    for model_type, preds in predictions.items():
        try:
            # NaN値のチェックと置換
            preds_array = np.array(preds)
            if np.isnan(preds_array).any():
                print(f"  {model_type}: NaN値があります。平均値で置換します。")
                mean_val = np.nanmean(preds_array)
                preds_array = np.nan_to_num(preds_array, nan=mean_val)
            
            # メトリクスを計算
            metrics = calculate_uplift_metrics(y_test, w_test, preds_array)
            all_scores[model_type] = metrics['qini']
            
            print(f"  {model_type}:")
            print(f"    Qini係数: {metrics['qini']:.4f}")
            print(f"    AUUC: {metrics['auuc']:.4f}")
            print(f"    上位20%アップリフト: {metrics['uplift_at_top_k']:.4f}")
            print(f"    予測-実績相関: {metrics['corr']:.4f}")
            
            # アップリフトカーブもプロット
            fig = plot_uplift_curve(y_test, w_test, preds_array, 
                                   title=f"Uplift Curve - {model_type}",
                                   save_path=f"uplift_curve_{model_type}.png")
            print(f"  アップリフトカーブを 'uplift_curve_{model_type}.png' として保存しました")
            plt.close(fig)
            
        except Exception as e:
            print(f"  {model_type}の評価エラー: {e}")
    
    # 7. 全モデルの最終比較
    if all_scores:
        print("\n=== モデル性能比較 ===")
        sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(sorted_models):
            print(f"{i+1}位: {model} (Qini係数: {score:.4f})")
        
        best_model = sorted_models[0][0]
        print(f"\n最良モデル: {best_model} (Qini係数: {all_scores[best_model]:.4f})")
        
        # 全モデルの比較プロット
        plt.figure(figsize=(12, 8))
        
        for model_type, preds in predictions.items():
            # データ準備
            df = pd.DataFrame({
                'y': y_test,
                'w': w_test,
                'uplift': np.array(preds)
            })
            
            # スコアで降順にソート
            df = df.sort_values('uplift', ascending=False)
            
            # 人口割合ごとの累積アップリフト計算
            population_fractions = np.linspace(0, 1, 21)[1:]  # 5%ごと
            incremental_uplifts = []
            
            for fraction in population_fractions:
                subset_size = int(len(df) * fraction)
                subset = df.iloc[:subset_size]
                
                treat_rate = subset[subset['w'] == 1]['y'].mean() if subset[subset['w'] == 1].shape[0] > 0 else 0
                control_rate = subset[subset['w'] == 0]['y'].mean() if subset[subset['w'] == 0].shape[0] > 0 else 0
                
                incremental_uplifts.append(treat_rate - control_rate)
            
            # プロット
            plt.plot(population_fractions, incremental_uplifts, marker='o', label=f"{model_type} (Qini={all_scores[model_type]:.4f})")
        
        # ランダムターゲティング線
        total_treat = df[df['w'] == 1]['y'].sum() / df[df['w'] == 1].shape[0]
        total_control = df[df['w'] == 0]['y'].sum() / df[df['w'] == 0].shape[0]
        total_uplift = total_treat - total_control
        plt.plot(population_fractions, [total_uplift * f for f in population_fractions], 
                linestyle='--', label='Random Targeting')
        
        plt.axhline(y=total_uplift, color='r', linestyle=':', label=f'Average Uplift ({total_uplift:.5f})')
        
        plt.xlabel('Population Fraction')
        plt.ylabel('Uplift')
        plt.title('Uplift Curves - Model Comparison')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig("uplift_model_comparison.png", dpi=300, bbox_inches='tight')
        print("\nモデル比較チャートを 'uplift_model_comparison.png' として保存しました")

if __name__ == "__main__":
    main()
