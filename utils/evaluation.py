import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

def calculate_uplift_metrics(y_true: np.ndarray, treatment: np.ndarray, uplift_scores: np.ndarray, 
                            n_bins: int = 10) -> Dict[str, float]:
    """
    アップリフトモデルの評価メトリクスを計算する関数
    
    Parameters
    ----------
    y_true : array-like
        実際の結果変数（コンバージョン等）
    treatment : array-like
        処置フラグ（1=処置群, 0=対照群）
    uplift_scores : array-like
        予測されたアップリフト値
    n_bins : int, default=10
        アップリフトスコアをビニングする区間数
        
    Returns
    -------
    Dict[str, float]
        計算されたメトリクス値の辞書
    """
    # データフレーム作成
    df = pd.DataFrame({
        'y': y_true,
        'w': treatment,
        'uplift': uplift_scores
    })
    
    # アップリフト値に基づいてn_bins個の等分位点で分割
    df['bin'] = pd.qcut(df['uplift'], q=n_bins, labels=False)
    
    # ビン別の集計
    bin_stats = []
    total_treat = df[df['w'] == 1].shape[0]
    total_control = df[df['w'] == 0].shape[0]
    total_treat_converted = df[(df['w'] == 1) & (df['y'] == 1)].shape[0]
    total_control_converted = df[(df['w'] == 0) & (df['y'] == 1)].shape[0]
    
    for i in range(n_bins):
        bin_df = df[df['bin'] == i]
        
        # 各ビンでの処置群と対照群のサイズとコンバージョン数
        treat_size = bin_df[bin_df['w'] == 1].shape[0]
        control_size = bin_df[bin_df['w'] == 0].shape[0]
        
        treat_conv = bin_df[(bin_df['w'] == 1) & (bin_df['y'] == 1)].shape[0]
        control_conv = bin_df[(bin_df['w'] == 0) & (bin_df['y'] == 1)].shape[0]
        
        # 対照群または処置群が0の場合はスキップ
        if treat_size == 0 or control_size == 0:
            continue
            
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
    
    # 結果がない場合（全てのビンで処置群または対照群が0）はデフォルト値を返す
    if not bin_stats:
        return {
            'auuc': 0.0,
            'qini': 0.0,
            'uplift_at_top_k': 0.0,
            'corr': 0.0
        }
    
    # メトリクスの計算
    bin_df = pd.DataFrame(bin_stats)
    
    # 予測アップリフトの降順にソート（高いアップリフト→低いアップリフトの順）
    bin_df = bin_df.sort_values('mean_pred_uplift', ascending=False)
    
    # 簡易的なAUUC（Area Under the Uplift Curve）の計算
    # ランダムターゲティングとの比較で正規化
    cum_treat = 0
    cum_control = 0
    cum_treat_conv = 0
    cum_control_conv = 0
    
    auuc_values = []
    random_auuc_values = []
    
    for _, row in bin_df.iterrows():
        # 累積値を計算
        cum_treat += row['treat_size']
        cum_control += row['control_size']
        cum_treat_conv += row['treat_conv']
        cum_control_conv += row['control_conv']
        
        # 現在の累積アップリフトを計算
        cum_treat_rate = cum_treat_conv / cum_treat if cum_treat > 0 else 0
        cum_control_rate = cum_control_conv / cum_control if cum_control > 0 else 0
        
        # 全体のアップリフト
        overall_treat_rate = total_treat_converted / total_treat if total_treat > 0 else 0
        overall_control_rate = total_control_converted / total_control if total_control > 0 else 0
        
        # 実際の累積アップリフト
        auuc_values.append((cum_treat_rate - cum_control_rate) * (cum_treat + cum_control))
        
        # ランダムターゲティングでの累積アップリフト
        random_frac = (cum_treat + cum_control) / (total_treat + total_control)
        random_auuc_values.append((overall_treat_rate - overall_control_rate) * random_frac * (total_treat + total_control))
    
    # AUUC（アップリフトカーブ下面積）
    auuc = np.trapz(auuc_values) / len(auuc_values)
    
    # ランダムターゲティングのAUUC
    random_auuc = np.trapz(random_auuc_values) / len(random_auuc_values)
    
    # Qini係数 (正規化AUUC) - ランダムと比較した相対的効果
    qini = (auuc - random_auuc) / random_auuc if random_auuc != 0 else 0
    
    # 上位20%でのアップリフト
    top_20_pct = int(n_bins * 0.2)
    if top_20_pct > 0 and top_20_pct <= len(bin_df):
        top_k_uplift = bin_df.iloc[:top_20_pct]['uplift'].mean()
    else:
        top_k_uplift = bin_df['uplift'].iloc[0] if len(bin_df) > 0 else 0
    
    # 予測アップリフトと実際のアップリフトの相関
    corr = bin_df['mean_pred_uplift'].corr(bin_df['uplift'])
    
    return {
        'auuc': float(auuc),
        'qini': float(qini),
        'uplift_at_top_k': float(top_k_uplift),
        'corr': float(corr)
    }

def plot_uplift_curve(y_true: np.ndarray, treatment: np.ndarray, uplift_scores: np.ndarray, 
                      n_bins: int = 10, title: str = "Uplift Curve", 
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    アップリフトカーブをプロットする関数
    
    Parameters
    ----------
    y_true : array-like
        実際の結果変数（コンバージョン等）
    treatment : array-like
        処置フラグ（1=処置群, 0=対照群）
    uplift_scores : array-like
        予測されたアップリフト値
    n_bins : int, default=10
        アップリフトスコアをビニングする区間数
    title : str, default="Uplift Curve"
        グラフのタイトル
    save_path : str, optional
        保存するパス
        
    Returns
    -------
    plt.Figure
        プロットした図
    """
    # データフレーム作成
    df = pd.DataFrame({
        'y': y_true,
        'w': treatment,
        'uplift': uplift_scores
    })
    
    # スコアで降順にソート
    df = df.sort_values('uplift', ascending=False)
    
    # 人口割合ごとの累積アップリフト計算
    population_fractions = np.linspace(0, 1, n_bins+1)[1:]
    
    incremental_uplifts = []
    random_uplifts = []
    
    total_samples = len(df)
    total_treat = df[df['w'] == 1]['y'].sum() / df[df['w'] == 1].shape[0]
    total_control = df[df['w'] == 0]['y'].sum() / df[df['w'] == 0].shape[0]
    total_uplift = total_treat - total_control
    
    for fraction in population_fractions:
        subset_size = int(total_samples * fraction)
        subset = df.iloc[:subset_size]
        
        # 処置群と対照群のコンバージョン率
        treat_rate = subset[subset['w'] == 1]['y'].mean() if subset[subset['w'] == 1].shape[0] > 0 else 0
        control_rate = subset[subset['w'] == 0]['y'].mean() if subset[subset['w'] == 0].shape[0] > 0 else 0
        
        # 増分アップリフト
        incremental_uplift = treat_rate - control_rate
        incremental_uplifts.append(incremental_uplift)
        
        # ランダムターゲティングでのアップリフト
        random_uplifts.append(total_uplift * fraction)
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(population_fractions, incremental_uplifts, marker='o', label='Model Uplift')
    ax.plot(population_fractions, random_uplifts, linestyle='--', label='Random Targeting')
    
    # アップリフトの実測値を水平線としてプロット
    ax.axhline(y=total_uplift, color='r', linestyle=':', label=f'Average Uplift ({total_uplift:.5f})')
    
    ax.set_xlabel('Population Fraction')
    ax.set_ylabel('Uplift')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # メトリクスをテキストとして追加
    metrics = calculate_uplift_metrics(y_true, treatment, uplift_scores, n_bins)
    metrics_text = f"AUUC: {metrics['auuc']:.5f}\nQini: {metrics['qini']:.5f}\nTop 20% Uplift: {metrics['uplift_at_top_k']:.5f}"
    ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
