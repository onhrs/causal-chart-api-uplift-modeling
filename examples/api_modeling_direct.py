import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# APIエンドポイント
API_URL = "http://localhost:8000"

# 評価関数を直接ファイル内に定義
def calculate_uplift_metrics(y_true, treatment, uplift_scores, n_bins=10):
    """アップリフトモデルの評価メトリクスを計算する関数"""
    # データフレーム作成
    df = pd.DataFrame({
        'y': y_true,
        'w': treatment,
        'uplift': uplift_scores
    })
    
    # アップリフト値に基づいてn_bins個の等分位点で分割
    df['bin'] = pd.qcut(df['uplift'], q=n_bins, labels=False, duplicates='drop')
    
    # ビン別の集計
    bin_stats = []
    total_treat = df[df['w'] == 1].shape[0]
    total_control = df[df['w'] == 0].shape[0]
    total_treat_converted = df[(df['w'] == 1) & (df['y'] == 1)].shape[0]
    total_control_converted = df[(df['w'] == 0) & (df['y'] == 1)].shape[0]
    
    for i in range(n_bins):
        bin_df = df[df['bin'] == i]
        if len(bin_df) == 0:
            continue
            
        # 各ビンでの処置群と対照群のサイズとコンバージョン数
        treat_size = bin_df[bin_df['w'] == 1].shape[0]
        control_size = bin_df[bin_df['w'] == 0].shape[0]
        
        if treat_size == 0 or control_size == 0:
            continue
            
        treat_conv = bin_df[(bin_df['w'] == 1) & (bin_df['y'] == 1)].shape[0]
        control_conv = bin_df[(bin_df['w'] == 0) & (bin_df['y'] == 1)].shape[0]
            
        # コンバージョン率とアップリフト
        treat_conv_rate = treat_conv / treat_size
        control_conv_rate = control_conv / control_size
        bin_uplift = treat_conv_rate - control_conv_rate
        
        bin_stats.append({
            'bin': i,
            'size': bin_df.shape[0],
            'treat_size': treat_size,
            'control_size': control_size,
            'treat_conv': treat_conv,
            'control_conv': control_conv,
            'treat_conv_rate': treat_conv_rate,
            'control_conv_rate': control_conv_rate,
            'uplift': bin_uplift,
            'mean_pred_uplift': bin_df['uplift'].mean()
        })
    
    # 結果がない場合はデフォルト値を返す
    if not bin_stats:
        return {
            'auuc': 0.0,
            'qini': 0.0,
            'uplift_at_top_k': 0.0,
            'corr': 0.0
        }
    
    bin_df = pd.DataFrame(bin_stats)
    bin_df = bin_df.sort_values('mean_pred_uplift', ascending=False)
    
    # 各種メトリクスを計算
    # AUUCとQini係数
    total_pop = total_treat + total_control
    cum_treat = 0
    cum_control = 0
    cum_treat_conv = 0
    cum_control_conv = 0
    
    auuc_values = []
    random_auuc_values = []
    
    for _, row in bin_df.iterrows():
        cum_treat += row['treat_size']
        cum_control += row['control_size']
        cum_treat_conv += row['treat_conv']
        cum_control_conv += row['control_conv']
        
        # 累積アップリフト
        cum_treat_rate = cum_treat_conv / cum_treat if cum_treat > 0 else 0
        cum_control_rate = cum_control_conv / cum_control if cum_control > 0 else 0
        
        # 全体のアップリフト
        overall_treat_rate = total_treat_converted / total_treat if total_treat > 0 else 0
        overall_control_rate = total_control_converted / total_control if total_control > 0 else 0
        
        auuc_values.append((cum_treat_rate - cum_control_rate) * (cum_treat + cum_control) / total_pop)
        random_auuc_values.append((overall_treat_rate - overall_control_rate) * (cum_treat + cum_control) / total_pop)
    
    # AUUC
    auuc = np.trapz(auuc_values) if auuc_values else 0
    random_auuc = np.trapz(random_auuc_values) if random_auuc_values else 0
    
    # Qini
    qini = (auuc - random_auuc) / (abs(random_auuc) + 1e-10) if random_auuc != 0 else 0
    
    # 上位20%でのアップリフト
    top_20_pct = max(1, int(len(bin_df) * 0.2))
    top_k_uplift = bin_df.iloc[:top_20_pct]['uplift'].mean() if len(bin_df) > 0 else 0
    
    # 相関
    corr = bin_df['mean_pred_uplift'].corr(bin_df['uplift']) if len(bin_df) > 1 else 0
    
    return {
        'auuc': float(auuc),
        'qini': float(qini),
        'uplift_at_top_k': float(top_k_uplift),
        'corr': float(corr)
    }

def plot_uplift_curve(y_true, treatment, uplift_scores, title="Uplift Curve", save_path=None):
    """アップリフトカーブをプロットする関数"""
    # データフレーム作成
    df = pd.DataFrame({
        'y': y_true,
        'w': treatment,
        'uplift': uplift_scores
    })
    
    # スコアで降順にソート
    df = df.sort_values('uplift', ascending=False)
    
    # 人口割合ごとの累積アップリフト計算
    n_bins = 10
    population_fractions = np.linspace(0, 1, n_bins+1)[1:]
    
    incremental_uplifts = []
    random_uplifts = []
    
    # 全体のアップリフト平均
    total_treat_rate = df[df['w'] == 1]['y'].mean()
    total_control_rate = df[df['w'] == 0]['y'].mean()
    total_uplift = total_treat_rate - total_control_rate
    
    for fraction in population_fractions:
        subset_size = int(len(df) * fraction)
        subset = df.iloc[:subset_size]
        
        # 処置群と対照群のコンバージョン率
        treat_rate = subset[subset['w'] == 1]['y'].mean() if subset[subset['w'] == 1].shape[0] > 0 else 0
        control_rate = subset[subset['w'] == 0]['y'].mean() if subset[subset['w'] == 0].shape[0] > 0 else 0
        
        incremental_uplifts.append(treat_rate - control_rate)
        random_uplifts.append(total_uplift * fraction)
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(population_fractions, incremental_uplifts, marker='o', label='Model Uplift')
    ax.plot(population_fractions, random_uplifts, linestyle='--', label='Random Targeting')
    ax.axhline(y=total_uplift, color='r', linestyle=':', label=f'Average Uplift ({total_uplift:.5f})')
    
    ax.set_xlabel('Population Fraction')
    ax.set_ylabel('Uplift')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # メトリクスをテキストとして追加
    metrics = calculate_uplift_metrics(y_true, treatment, uplift_scores)
    metrics_text = f"AUUC: {metrics['auuc']:.5f}\nQini: {metrics['qini']:.5f}\nTop 20% Uplift: {metrics['uplift_at_top_k']:.5f}"
    ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

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
        
        try:
            response = requests.post(f"{API_URL}/predict", json=predict_payload)
            response.raise_for_status()  # ステータスコードが4xx/5xxの場合は例外を発生
            result = response.json()
            preds = result["predictions"]
            print(f"  成功: {len(preds)}件の予測")
            
            # 予測結果のサイズ修正
            if len(preds) != len(test_df):
                print(f"  警告: 予測サイズ({len(preds)})がテストデータサイズ({len(test_df)})と異なります")
                if len(preds) == 2 * len(test_df):
                    # causal_treeは1件の予測に対して2つの値を返す場合がある
                    preds = preds[::2]
                    print(f"  2倍のサイズのため1つおきに選択: {len(preds)}件")
                elif len(preds) > len(test_df):
                    preds = preds[:len(test_df)]
                    print(f"  サイズが大きいため切り詰め: {len(preds)}件")
                else:
                    mean_val = np.mean(preds)
                    preds = preds + [mean_val] * (len(test_df) - len(preds))
                    print(f"  サイズが小さいため平均値で埋め: {len(preds)}件")
            
            # R-learnerの異常値修正
            if model_type == 'r_learner':
                preds_array = np.array(preds, dtype=float)
                outliers = np.abs(preds_array) > 1
                if np.any(outliers):
                    outlier_count = np.sum(outliers)
                    print(f"  警告: {outlier_count}件の異常値を検出")
                    median = np.median(preds_array[~outliers]) if np.sum(~outliers) > 0 else 0
                    preds_array[outliers] = median
                    print(f"  異常値を中央値({median:.5f})で置換")
                    preds = preds_array.tolist()
            
            predictions[model_type] = preds
        except Exception as e:
            print(f"  失敗: {str(e)}")
    
    # 6. アップリフト評価
    y_test = test_df['Conversion'].values
    w_test = test_df['treatment'].values
    
    print("\nアップリフト評価メトリクス:")
    all_scores = {}
    
    for model_type, preds in predictions.items():
        try:
            # NaN値のチェックと置換
            preds_array = np.array(preds, dtype=float)
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
            
            # アップリフトカーブをプロット
            save_path = f"uplift_curve_{model_type}.png"
            fig = plot_uplift_curve(
                y_test, w_test, preds_array,
                title=f"Uplift Curve - {model_type}",
                save_path=save_path
            )
            print(f"  アップリフトカーブを '{save_path}' として保存しました")
            plt.close(fig)
            
        except Exception as e:
            print(f"  {model_type}の評価エラー: {str(e)}")
    
    # 7. モデル比較
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
                'uplift': np.array(preds, dtype=float)
            })
            
            # スコアで降順にソート
            df = df.sort_values('uplift', ascending=False)
            
            # 人口割合ごとの累積アップリフト計算
            population_fractions = np.linspace(0, 1, 21)[1:]  # 5%ごと
            incremental_uplifts = []
            
            for fraction in population_fractions:
                subset_size = int(len(df) * fraction)
                subset = df.iloc[:subset_size]
                
                treat_size = subset[subset['w'] == 1].shape[0]
                control_size = subset[subset['w'] == 0].shape[0]
                
                if treat_size > 0 and control_size > 0:
                    treat_rate = subset[subset['w'] == 1]['y'].mean()
                    control_rate = subset[subset['w'] == 0]['y'].mean()
                    incremental_uplifts.append(treat_rate - control_rate)
                else:
                    # 処置群または対照群がない場合は直前の値を使用するか0
                    incremental_uplifts.append(incremental_uplifts[-1] if incremental_uplifts else 0)
            
            # プロット
            plt.plot(population_fractions, incremental_uplifts, marker='o', 
                     label=f"{model_type} (Qini={all_scores[model_type]:.4f})")
        
        # ランダムターゲティング線
        total_treat_rate = df[df['w'] == 1]['y'].mean()
        total_control_rate = df[df['w'] == 0]['y'].mean()
        total_uplift = total_treat_rate - total_control_rate
        
        plt.plot(population_fractions, [total_uplift * f for f in population_fractions], 
                 linestyle='--', label='Random Targeting')
        plt.axhline(y=total_uplift, color='r', linestyle=':', 
                   label=f'Average Uplift ({total_uplift:.5f})')
        
        plt.xlabel('Population Fraction')
        plt.ylabel('Uplift')
        plt.title('Uplift Curves - Model Comparison')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        model_comparison_path = "uplift_model_comparison.png"
        plt.savefig(model_comparison_path, dpi=300, bbox_inches='tight')
        print(f"\nモデル比較チャートを '{model_comparison_path}' として保存しました")
        plt.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"エラーが発生しました: {str(e)}")
        print(traceback.format_exc())
