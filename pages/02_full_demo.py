import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from client_utils.debug_helpers import safe_request
import time
from io import StringIO
import altair as alt
from sklearn.model_selection import train_test_split
from utils.font_settings import set_matplotlib_japanize

# 日本語フォント設定
set_matplotlib_japanize()

# APIエンドポイントの設定
API_URL = os.environ.get("API_URL", "https://causal-chart-api-620283975862.asia-northeast1.run.app")

# ページ設定
st.set_page_config(
    page_title="Full Demo - Uplift Modeling",
    page_icon="📊",
    layout="wide"
)

# ヘルパー関数
def load_hillstrom_dataset():
    """Hillstromデータセットをロードまたはダウンロード"""
    DATA_PATH = "assets/hillstrom.csv"
    
    # ファイルが存在するか確認
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    
    # ファイルが存在しない場合、ダウンロード
    try:
        st.info("Hillstromデータセットをダウンロードしています...")
        
        # URLからデータを取得
        url = "https://raw.githubusercontent.com/Trinhnguyen1704/Causal-inference/master/data/kevin_hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
        data = pd.read_csv(url)
        
        # assetsディレクトリを作成
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        # ファイルを保存
        data.to_csv(DATA_PATH, index=False)
        st.success("データセットがダウンロードされました！")
        
        return data
    
    except Exception as e:
        st.error(f"データセットのダウンロードに失敗しました: {str(e)}")
        
        # サンプルデータを作成して返す
        st.warning("サンプルデータを使用します")
        return create_sample_data()

def create_sample_data():
    """サンプルデータの作成"""
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'recency': np.random.randint(1, 60, n),
        'history': np.random.randint(1, 500, n),
        'mens': np.random.randint(0, 2, n),
        'womens': np.random.randint(0, 2, n),
        'newbie': np.random.randint(0, 2, n),
        'channel': np.random.choice(['Phone', 'Web', 'Multichannel'], n),
        'segment': np.random.choice(['Womens E-Mail', 'Mens E-Mail', 'No E-Mail'], n, p=[0.4, 0.4, 0.2])
    })
    
    # 処置変数の作成
    data['treatment'] = data['segment'].apply(lambda x: 0 if x == 'No E-Mail' else 1)
    
    # 結果変数の生成
    base_conv = 0.05
    treatment_effect = 0.03
    
    # 特徴量による異質性
    recency_effect = -0.0005  # 最近購入ほど効果が高い
    history_effect = 0.0001   # 購入履歴が多いほど効果が高い
    
    # ノイズ付きの効果を生成
    data['individual_effect'] = (
        treatment_effect +
        recency_effect * data['recency'] +
        history_effect * data['history'] +
        np.random.normal(0, 0.01, n)
    )
    
    # 処置に応じた結果を生成
    prob = base_conv + data['treatment'] * data['individual_effect']
    prob = np.clip(prob, 0.01, 0.99)  # 確率を0.01〜0.99に制限
    
    data['visit'] = np.random.binomial(1, prob, n)
    data['conversion'] = np.random.binomial(1, prob * 0.5, n)  # 訪問のうち半分がコンバージョン
    data['spend'] = np.where(data['conversion'] == 1, np.random.normal(50, 10, n), 0)
    
    return data

def plot_qini_curve(df, predicted_uplift, treatment_col='treatment', outcome_col='conversion'):
    # データの準備
    df_plot = df.copy()
    df_plot['predicted_uplift'] = predicted_uplift
    
    # 予測アップリフトでソート
    df_plot = df_plot.sort_values('predicted_uplift', ascending=False)
    
    # グループ作成（10分位）
    n_groups = 10
    df_plot['percentile'] = pd.qcut(range(len(df_plot)), n_groups, labels=False)
    
    # 各グループでの実際のアップリフトを計算
    uplift_by_group = []
    for i in range(n_groups):
        group_df = df_plot[df_plot['percentile'] == i]
        
        # 処置群と対照群の分離
        treatment_outcome = group_df[group_df[treatment_col] == 1][outcome_col].mean()
        control_outcome = group_df[group_df[treatment_col] == 0][outcome_col].mean()
        
        # アップリフト計算
        uplift = treatment_outcome - control_outcome
        pop_size = len(group_df) / len(df_plot)
        
        uplift_by_group.append({
            'group': i,
            'percentile': (i + 1) / n_groups,
            'population_percentage': pop_size,
            'cumulative_population': (i + 1) * pop_size,
            'treatment_outcome': treatment_outcome,
            'control_outcome': control_outcome,
            'uplift': uplift,
            'cumulative_uplift': 0  # 後で計算
        })
    
    # データフレーム変換と累積アップリフトの計算
    result_df = pd.DataFrame(uplift_by_group)
    result_df['cumulative_uplift'] = result_df['uplift'].cumsum()
    
    # ランダムターゲティングのラインの計算（理論値）
    theoretical_random = []
    for p in result_df['cumulative_population']:
        theoretical_random.append(p * (df[df[treatment_col] == 1][outcome_col].mean() - df[df[treatment_col] == 0][outcome_col].mean()))
    
    result_df['random_targeting'] = theoretical_random
    
    return result_df

def main():
    st.title("本格デモ: Hillstromデータセットによるアップリフト分析")
    
    st.markdown("""
    このページでは、Kevin Hillstromの有名なeコマースデータセットを使用して本格的なアップリフトモデリング分析を行います。
    このデータセットには、メールキャンペーンの効果測定のための顧客データと結果が含まれています。
    
    **データセットの概要:**
    - 約64,000件の顧客レコード
    - 「女性向けメール」「男性向けメール」「メールなし（対照群）」の3セグメント
    - 結果変数: 訪問回数、コンバージョン、支出額
    """)
    
    # データの読み込み
    with st.spinner('データを読み込んでいます...'):
        df = load_hillstrom_dataset()
    
    if df is not None:
        st.success('データの読み込みが完了しました！')
        
        # データ前処理と探索のタブ
        tab1, tab2, tab3, tab4 = st.tabs(["データ探索", "モデル比較", "アップリフト分析", "ROI計算"])
        
        with tab1:
            st.header("データ探索と前処理")
            
            # データの概要を表示
            st.subheader("データの概要")
            st.write(f"サンプル数: {len(df)}")
            
            # データのプレビュー
            st.dataframe(df.head())
            
            # データ前処理
            st.subheader("データ前処理")
            
            # 処置変数の作成（'segment'列を使用）
            df['treatment'] = df['segment'].apply(lambda x: 0 if x == 'No E-Mail' else 1)
            
            # 結果変数の選択
            outcome_var = st.selectbox(
                "分析に使用する結果変数を選択:",
                ["visit", "conversion", "spend"],
                help="Hillstromデータセットには複数の結果指標が含まれています"
            )
            
            # 使用する特徴量の選択
            available_features = [col for col in df.columns 
                                 if col not in ['segment', 'treatment', 'visit', 'conversion', 'spend']]
            
            selected_features = st.multiselect(
                "使用する特徴量を選択:",
                available_features,
                default=available_features[:5],
                help="モデルのトレーニングに使用する特徴量を選択します"
            )
            
            # 処理済みデータを保存
            if 'processed_data' not in st.session_state:
                st.session_state.processed_data = {
                    'df': df,
                    'outcome_var': outcome_var,
                    'features': selected_features
                }
            
            # データの基本統計を表示
            if selected_features:
                st.subheader("特徴量の基本統計")
                st.write(df[selected_features].describe())
                
                # 相関行列の表示
                st.subheader("相関行列")
                corr = df[selected_features].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                st.pyplot(fig)
                
                # 処置効果の初期評価
                st.subheader("処置効果の初期評価")
                
                # 処置群と対照群でのターゲット変数の平均値
                treatment_outcome = df[df['treatment'] == 1][outcome_var].mean()
                control_outcome = df[df['treatment'] == 0][outcome_var].mean()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="処置群の結果", value=f"{treatment_outcome:.4f}")
                
                with col2:
                    st.metric(label="対照群の結果", value=f"{control_outcome:.4f}")
                
                with col3:
                    uplift = treatment_outcome - control_outcome
                    st.metric(
                        label="平均処置効果 (ATE)", 
                        value=f"{uplift:.4f}", 
                        delta=f"{100 * uplift / control_outcome:.2f}%" if control_outcome > 0 else "N/A"
                    )
                
                # セグメント別の処置効果
                st.subheader("セグメント別の処置効果")
                
                # セグメント変数の選択
                segment_var = st.selectbox(
                    "セグメント変数を選択:",
                    available_features,
                    help="顧客をセグメント化する変数を選択します"
                )
                
                if segment_var in df.columns:
                    # セグメント値が多すぎる場合は分位数に変換
                    if df[segment_var].nunique() > 5:
                        n_segments = st.slider("セグメント数", min_value=2, max_value=10, value=5)
                        df['segment_group'] = pd.qcut(df[segment_var], n_segments, labels=[f'Q{i+1}' for i in range(n_segments)])
                        segment_column = 'segment_group'
                    else:
                        segment_column = segment_var
                    
                    # セグメント別の処置効果を計算
                    segment_effects = []
                    for segment in df[segment_column].unique():
                        segment_df = df[df[segment_column] == segment]
                        seg_treatment = segment_df[segment_df['treatment'] == 1][outcome_var].mean()
                        seg_control = segment_df[segment_df['treatment'] == 0][outcome_var].mean()
                        seg_effect = seg_treatment - seg_control
                        segment_effects.append({
                            'segment': segment,
                            'treatment_outcome': seg_treatment,
                            'control_outcome': seg_control,
                            'uplift': seg_effect,
                            'perc_change': 100 * seg_effect / seg_control if seg_control > 0 else float('nan')
                        })
                    
                    segment_df = pd.DataFrame(segment_effects)
                    
                    # セグメント効果の表示
                    st.write(segment_df)
                    
                    # セグメント効果のグラフ表示
                    fig, ax = plt.subplots(figsize=(10, 6))
                    barwidth = 0.3
                    x = np.arange(len(segment_df))
                    
                    ax.bar(x - barwidth/2, segment_df['control_outcome'], width=barwidth, label='対照群', color='#1f77b4')
                    ax.bar(x + barwidth/2, segment_df['treatment_outcome'], width=barwidth, label='処置群', color='#ff7f0e')
                    
                    for i, row in enumerate(segment_effects):
                        ax.annotate(
                            f"{row['uplift']:.3f}\n({row['perc_change']:.1f}%)" if not pd.isna(row['perc_change']) else f"{row['uplift']:.3f}",
                            xy=(i, max(row['treatment_outcome'], row['control_outcome']) + 0.02),
                            ha='center'
                        )
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(segment_df['segment'])
                    ax.set_ylabel(outcome_var)
                    ax.set_title(f'セグメント別の{outcome_var}')
                    ax.legend()
                    
                    st.pyplot(fig)
        
        with tab2:
            st.header("アップリフトモデルの比較")
            
            # モデルタイプの選択
            model_types = st.multiselect(
                "比較するモデルタイプを選択:",
                ["s_learner", "t_learner", "x_learner", "r_learner", "causal_tree", "uplift_rf"],
                default=["s_learner", "t_learner", "x_learner"],
                help="比較したいアップリフトモデリング手法を選択します"
            )
            
            # モデルの共通パラメータ
            st.subheader("モデルパラメータ")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.slider(
                    "n_estimators (ツリーの数)", 
                    min_value=10, 
                    max_value=200, 
                    value=50
                )
            
            with col2:
                max_depth = st.slider(
                    "max_depth (最大深さ)", 
                    min_value=1, 
                    max_value=10, 
                    value=3
                )
            
            # データの分割設定
            test_size = st.slider("テストデータの割合", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
            
            # トレーニングの実行
            train_button = st.button("選択したモデルをトレーニング", key="train_multiple_models")
            
            if train_button:
                if not model_types:
                    st.warning("モデルタイプを少なくとも1つ選択してください。")
                elif not selected_features:
                    st.warning("特徴量を少なくとも1つ選択してください。")
                else:
                    # セッションに結果を格納するための準備
                    if 'model_results' not in st.session_state:
                        st.session_state.model_results = {}
                    
                    # 各モデルをトレーニング
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, model_type in enumerate(model_types):
                        try:
                            status_text.text(f"{model_type} モデルをトレーニング中...")
                            
                            # トレーニングリクエストの準備
                            train_data = df.copy()
                            
                            # トレーニングリクエストの準備
                            train_payload = {
                                "data": train_data.to_dict('records'),
                                "features": selected_features,
                                "treatment_col": "treatment",
                                "outcome_col": outcome_var,
                                "model_type": model_type,
                                "model_params": {
                                    "n_estimators": n_estimators,
                                    "max_depth": max_depth
                                }
                            }
                            
                            # APIリクエスト送信
                            train_response = safe_request("POST", f"{API_URL}/train", train_payload, debug=False)
                            
                            if (train_response and "model_path" in train_response):
                                # 結果を保存
                                st.session_state.model_results[model_type] = {
                                    'model_path': train_response["model_path"],
                                    'model_info': train_response["model_info"]
                                }
                                
                                # 進捗を更新
                                progress_bar.progress((idx + 1) / len(model_types))
                                status_text.text(f"{model_type} モデルのトレーニングが完了しました")
                            
                        except Exception as e:
                            st.error(f"{model_type} モデルのトレーニングに失敗しました: {str(e)}")
                    
                    status_text.text("すべてのモデルのトレーニングが完了しました！")
                    progress_bar.progress(1.0)
                    
                    # トレーニング結果の表示
                    if st.session_state.model_results:
                        st.subheader("モデル比較結果")
                        
                        results = []
                        for model_type, result in st.session_state.model_results.items():
                            metrics = result['model_info']['metrics']
                            results.append({
                                'モデル': model_type,
                                '実際のATE': metrics.get('actual_ate', 0),
                                '推定ATE': metrics.get('estimated_ate', 0),
                                '対照群結果': metrics.get('control_outcome', 0),
                                '処置群結果': metrics.get('treatment_outcome', 0),
                                'ATE予測誤差': abs(metrics.get('actual_ate', 0) - metrics.get('estimated_ate', 0))
                            })
                        
                        # 結果のDataFrame
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        
                        # ATE比較グラフ
                        fig, ax = plt.subplots(figsize=(10, 6))
                        x = range(len(results))
                        width = 0.35
                        
                        # 実際のATEと推定ATEの棒グラフ
                        ax.bar([i - width/2 for i in x], results_df['実際のATE'], width, label='実際のATE')
                        ax.bar([i + width/2 for i in x], results_df['推定ATE'], width, label='推定ATE')
                        
                        # ラベルと凡例の設定
                        ax.set_xlabel('モデル')
                        ax.set_ylabel('平均処置効果 (ATE)')
                        ax.set_title('モデル別の実際のATEと推定ATEの比較')
                        ax.set_xticks(x)
                        ax.set_xticklabels(results_df['モデル'])
                        ax.legend()
                        
                        # 値のラベル付け
                        for i, v in enumerate(results_df['実際のATE']):
                            ax.text(i - width/2, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
                        
                        for i, v in enumerate(results_df['推定ATE']):
                            ax.text(i + width/2, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
                        
                        st.pyplot(fig)
                        
                        # 予測誤差のグラフ
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(results_df['モデル'], results_df['ATE予測誤差'], color='#d62728')
                        ax.set_xlabel('モデル')
                        ax.set_ylabel('ATE予測誤差 (絶対値)')
                        ax.set_title('モデル別のATE予測誤差')
                        
                        # 値のラベル付け
                        for i, v in enumerate(results_df['ATE予測誤差']):
                            ax.text(i, v + 0.0005, f'{v:.4f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
            
            # モデルが存在する場合は予測実行
            if 'model_results' in st.session_state and st.session_state.model_results:
                st.subheader("テストデータでの予測")
                
                # 予測に使用するモデルを選択
                predict_model = st.selectbox(
                    "予測に使用するモデルを選択:",
                    list(st.session_state.model_results.keys())
                )
                
                if st.button("予測を実行", key="run_prediction_full"):
                    if predict_model in st.session_state.model_results:
                        model_path = st.session_state.model_results[predict_model]['model_path']
                        
                        # テストデータの準備
                        X_test = df[selected_features]
                        
                        # 予測リクエストの準備
                        predict_payload = {
                            "features": X_test.to_dict('records'),
                            "model_path": model_path
                        }
                        
                        try:
                            # 予測の実行
                            with st.spinner('予測中...'):
                                predict_response = safe_request("POST", f"{API_URL}/predict", predict_payload, debug=False)
                            
                            if predict_response and "predictions" in predict_response:
                                # 予測の保存
                                predicted_uplift = predict_response["predictions"]
                                
                                # データセットにアップリフト予測を追加
                                predict_df = df.copy()
                                predict_df['predicted_uplift'] = predicted_uplift
                                
                                # 予測アップリフトでソート
                                predict_df_sorted = predict_df.sort_values('predicted_uplift', ascending=False)
                                
                                # 上位10件の表示
                                st.subheader("最も高いアップリフトが予測された顧客")
                                cols_to_show = selected_features + ['treatment', outcome_var, 'predicted_uplift']
                                st.dataframe(predict_df_sorted[cols_to_show].head(10))
                                
                                # 下位10件の表示
                                st.subheader("最も低い（または負の）アップリフトが予測された顧客")
                                st.dataframe(predict_df_sorted[cols_to_show].tail(10))
                                
                                # 予測結果のヒストグラム
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.hist(predicted_uplift, bins=30, alpha=0.7)
                                ax.set_xlabel('予測アップリフト')
                                ax.set_ylabel('頻度')
                                ax.set_title('予測アップリフトの分布')
                                ax.axvline(x=0, color='r', linestyle='--')
                                st.pyplot(fig)
                                
                                # セッションに予測結果を保存
                                st.session_state.prediction_results = {
                                    'model': predict_model,
                                    'predictions': predicted_uplift,
                                    'predict_df': predict_df
                                }
                        
                        except Exception as e:
                            st.error(f"予測エラー: {str(e)}")
        
        with tab3:
            st.header("アップリフト分析")
            
            if 'prediction_results' not in st.session_state:
                st.warning("先にモデルをトレーニングして予測を実行してください。")
            else:
                # 予測結果を取得
                predict_df = st.session_state.prediction_results['predict_df']
                predicted_uplift = st.session_state.prediction_results['predictions']
                
                # Qiniカーブの計算
                qini_df = plot_qini_curve(predict_df, predicted_uplift, 'treatment', outcome_var)
                
                # Qiniカーブの表示
                st.subheader("Qiniカーブ")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(qini_df['cumulative_population'], qini_df['cumulative_uplift'], 'b-', marker='o', label='モデルによるターゲティング')
                ax.plot(qini_df['cumulative_population'], qini_df['random_targeting'], 'r--', label='ランダムターゲティング')
                ax.set_xlabel('ターゲット母集団の割合')
                ax.set_ylabel('累積アップリフト')
                ax.set_title('Qiniカーブ')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # バケットごとのアップリフトの表示
                st.subheader("予測アップリフト分位別の実際のアップリフト")
                
                # データの表示
                st.dataframe(qini_df)
                
                # 分位別アップリフトのグラフ
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # 棒グラフの設定
                x = range(len(qini_df))
                bar_width = 0.35
                
                # 処置群と対照群の結果
                ax.bar([i - bar_width/2 for i in x], qini_df['control_outcome'], bar_width, label='対照群', color='#1f77b4')
                ax.bar([i + bar_width/2 for i in x], qini_df['treatment_outcome'], bar_width, label='処置群', color='#ff7f0e')
                
                # アップリフトの線グラフ
                ax2 = ax.twinx()
                ax2.plot(x, qini_df['uplift'], 'g-', marker='o', label='アップリフト')
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                
                # ラベル設定
                ax.set_xlabel('予測アップリフトの分位 (高 → 低)')
                ax.set_ylabel('平均結果')
                ax2.set_ylabel('実際のアップリフト')
                ax.set_title('分位別の処置効果')
                ax.set_xticks(x)
                ax.set_xticklabels([f'{i+1}' for i in range(len(qini_df))])
                
                # 凡例の設定
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                st.pyplot(fig)
                
                # 特徴量重要度
                st.subheader("セグメント分析")
                
                # 高アップリフト顧客と低アップリフト顧客の特徴を比較
                high_uplift = predict_df[predict_df['predicted_uplift'] > 0]
                low_uplift = predict_df[predict_df['predicted_uplift'] <= 0]
                
                # 総計
                total_high = len(high_uplift)
                total_low = len(low_uplift)
                
                # 特徴量ごとの分布比較
                st.write(f"正のアップリフト予測顧客: {total_high}件")
                st.write(f"負または0のアップリフト予測顧客: {total_low}件")
                
                # 各特徴量の平均値比較
                feature_comparison = []
                
                for feature in selected_features:
                    high_mean = high_uplift[feature].mean()
                    low_mean = low_uplift[feature].mean()
                    diff_perc = 100 * (high_mean - low_mean) / (low_mean if low_mean != 0 else 1)
                    
                    feature_comparison.append({
                        '特徴量': feature,
                        '正アップリフト顧客の平均': high_mean,
                        '負アップリフト顧客の平均': low_mean,
                        '差異 (%)': diff_perc
                    })
                
                # 比較データフレーム
                comparison_df = pd.DataFrame(feature_comparison)
                comparison_df_sorted = comparison_df.sort_values(by='差異 (%)', ascending=False)
                
                st.dataframe(comparison_df_sorted)
                
                # 主要特徴量の比較グラフ
                st.subheader("主要特徴量の分布比較")
                
                # 最も差異の大きい特徴量のヒストグラム
                top_features = comparison_df_sorted.head(3)['特徴量'].tolist()
                
                for feature in top_features:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(high_uplift[feature], bins=30, alpha=0.7, label='正アップリフト顧客', color='#ff7f0e')
                    ax.hist(low_uplift[feature], bins=30, alpha=0.7, label='負アップリフト顧客', color='#1f77b4')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('頻度')
                    ax.set_title(f'{feature} の分布比較')
                    ax.legend()
                    st.pyplot(fig)
        
        with tab4:
            st.header("ROI計算と最適ターゲティング")
            
            if 'prediction_results' not in st.session_state:
                st.warning("先にモデルをトレーニングして予測を実行してください。")
            else:
                # 予測結果を取得
                predict_df = st.session_state.prediction_results['predict_df']
                predicted_uplift = st.session_state.prediction_results['predictions']
                
                # ROI計算のためのパラメータ入力
                st.subheader("キャンペーンパラメータ")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    campaign_cost_per_customer = st.number_input(
                        "顧客あたりのキャンペーンコスト",
                        min_value=0.1,
                        max_value=100.0,
                        value=5.0,
                        step=0.1,
                        help="キャンペーン施策にかかる顧客1人あたりの費用"
                    )
                    
                with col2:
                    avg_conv_value = st.number_input(
                        "コンバージョンあたりの平均価値",
                        min_value=1.0,
                        max_value=1000.0,
                        value=50.0,
                        step=1.0,
                        help="1件のコンバージョン（購入など）による平均的な収益"
                    )
                
                # コンバージョンが金額データでない場合の対応
                if outcome_var != 'spend':
                    st.info(f"選択した結果変数は '{outcome_var}' です。金額ベースのROI計算のため、コンバージョンあたりの平均価値を使用します。")
                
                # 予測アップリフトの閾値スライダー
                uplift_threshold = st.slider(
                    "アップリフト閾値（これ以上の顧客のみに施策を実施）",
                    min_value=float(min(predicted_uplift)),
                    max_value=float(max(predicted_uplift)),
                    value=0.0,
                    step=0.001,
                    format="%.3f"
                )
                
                # ROIの計算
                target_customers = predict_df[predict_df['predicted_uplift'] >= uplift_threshold]
                non_target_customers = predict_df[predict_df['predicted_uplift'] < uplift_threshold]
                
                # ターゲット顧客数
                n_targeted = len(target_customers)
                total_customers = len(predict_df)
                targeting_percentage = 100 * n_targeted / total_customers if total_customers > 0 else 0
                
                # 予測される増分コンバージョン
                if outcome_var == 'spend':
                    # spendの場合は金額そのものを使用
                    incremental_value = target_customers['predicted_uplift'].sum()
                else:
                    # コンバージョン数にコンバージョンあたりの価値を掛ける
                    incremental_value = target_customers['predicted_uplift'].sum() * avg_conv_value
                
                # キャンペーン総コスト
                total_cost = n_targeted * campaign_cost_per_customer
                
                # ROI計算
                if total_cost > 0:
                    roi = (incremental_value - total_cost) / total_cost * 100
                else:
                    roi = 0
                
                # 結果の表示
                st.subheader("ROI分析")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="ターゲット顧客数", 
                        value=f"{n_targeted:,}",
                        delta=f"{targeting_percentage:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        label="予測される増分価値", 
                        value=f"¥{incremental_value:,.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="予測ROI", 
                        value=f"{roi:.1f}%",
                        delta="良好" if roi > 0 else "不十分"
                    )
                
                # 詳細分析の表示
                st.subheader("ROI詳細")
                
                roi_details = {
                    "項目": ["ターゲット顧客数", "総顧客数", "ターゲティング率", 
                           "キャンペーン単価", "総キャンペーンコスト", "予測される増分価値", "ROI"],
                    "値": [f"{n_targeted:,}", f"{total_customers:,}", f"{targeting_percentage:.1f}%",
                         f"¥{campaign_cost_per_customer:.2f}", f"¥{total_cost:,.2f}", 
                         f"¥{incremental_value:,.2f}", f"{roi:.1f}%"]
                }
                
                st.table(pd.DataFrame(roi_details))
                
                # ROIカーブの計算
                st.subheader("異なる閾値でのROIシミュレーション")
                
                # アップリフトの分位数を計算
                percentiles = np.linspace(0, 100, 21)  # 0%から100%まで5%刻み
                thresholds = np.percentile(predicted_uplift, percentiles)
                
                roi_results = []
                for threshold in thresholds:
                    # 各閾値でのターゲット顧客
                    target = predict_df[predict_df['predicted_uplift'] >= threshold]
                    n_target = len(target)
                    
                    # 増分価値
                    if outcome_var == 'spend':
                        incr_value = target['predicted_uplift'].sum()
                    else:
                        incr_value = target['predicted_uplift'].sum() * avg_conv_value
                    
                    # コストとROI
                    campaign_cost = n_target * campaign_cost_per_customer
                    
                    if campaign_cost > 0:
                        threshold_roi = (incr_value - campaign_cost) / campaign_cost * 100
                    else:
                        threshold_roi = 0
                    
                    # 結果を保存
                    roi_results.append({
                        '閾値': threshold,
                        'ターゲット顧客数': n_target,
                        'ターゲット率': 100 * n_target / total_customers if total_customers > 0 else 0,
                        '増分価値': incr_value,
                        'キャンペーンコスト': campaign_cost,
                        'ROI': threshold_roi,
                        '純利益': incr_value - campaign_cost
                    })
                
                # 結果をデータフレームに変換
                roi_df = pd.DataFrame(roi_results)
                
                # 最適閾値（純利益最大）を見つける
                if len(roi_df) > 0:
                    optimal_idx = roi_df['純利益'].idxmax()
                    optimal_threshold = roi_df.iloc[optimal_idx]['閾値']
                    optimal_target_rate = roi_df.iloc[optimal_idx]['ターゲット率']
                    optimal_roi = roi_df.iloc[optimal_idx]['ROI']
                    optimal_profit = roi_df.iloc[optimal_idx]['純利益']
                    
                    st.success(f"最適なアップリフト閾値: {optimal_threshold:.4f}")
                    st.info(f"この閾値では全体の {optimal_target_rate:.1f}% の顧客にキャンペーンを実施し、ROIは {optimal_roi:.1f}%、純利益は ¥{optimal_profit:,.2f} と予測されます")
                
                # ROIカーブのグラフ
                fig, ax1 = plt.subplots(figsize=(12, 7))
                
                # ROIの折れ線グラフ
                color = 'tab:blue'
                ax1.set_xlabel('ターゲット顧客の割合 (%)')
                ax1.set_ylabel('ROI (%)', color=color)
                ax1.plot(roi_df['ターゲット率'], roi_df['ROI'], 'o-', color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                
                # 純利益の折れ線グラフ（第2軸）
                ax2 = ax1.twinx()
                color = 'tab:green'
                ax2.set_ylabel('純利益', color=color)
                ax2.plot(roi_df['ターゲット率'], roi_df['純利益'], 's-', color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                
                # 最適ポイントのマーキング
                if 'optimal_idx' in locals():
                    opt_x = roi_df.iloc[optimal_idx]['ターゲット率']
                    opt_y1 = roi_df.iloc[optimal_idx]['ROI']
                    opt_y2 = roi_df.iloc[optimal_idx]['純利益']
                    
                    ax1.plot(opt_x, opt_y1, 'o', color='red', markersize=8)
                    ax2.plot(opt_x, opt_y2, 's', color='red', markersize=8)
                    ax1.annotate('最適ポイント', 
                                 xy=(opt_x, opt_y1),
                                 xytext=(opt_x+5, opt_y1+20),
                                 arrowprops=dict(arrowstyle='->'))
                
                plt.title('ターゲット率に対するROIと純利益')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # 予算ベースの最適化
                st.subheader("予算ベースの最適化")
                
                # 予算入力
                total_budget = st.number_input(
                    "キャンペーンの総予算",
                    min_value=100,
                    max_value=1000000,
                    value=10000,
                    step=100,
                    help="キャンペーン全体の予算"
                )
                
                # 予算制約下での最適化
                # 顧客単価が分かっているので、予算内で何人にアプローチできるかを計算
                max_customers = int(total_budget / campaign_cost_per_customer)
                
                # 予算内に収まる顧客数を表示
                st.write(f"予算 ¥{total_budget:,} で、最大 {max_customers:,} 人の顧客にアプローチできます")
                
                # 予測アップリフトでソートして上位を選択
                sorted_df = predict_df.sort_values('predicted_uplift', ascending=False)
                budget_targets = sorted_df.head(max_customers)
                
                # 予算内ターゲット顧客の分析
                avg_uplift_targeted = budget_targets['predicted_uplift'].mean()
                
                # 増分価値
                if outcome_var == 'spend':
                    budget_incremental_value = budget_targets['predicted_uplift'].sum()
                else:
                    budget_incremental_value = budget_targets['predicted_uplift'].sum() * avg_conv_value
                
                # ROI計算
                budget_roi = (budget_incremental_value - total_budget) / total_budget * 100 if total_budget > 0 else 0
                
                # 結果表示
                st.subheader("予算ベースの結果")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="ターゲット顧客数", 
                        value=f"{len(budget_targets):,}",
                        delta=f"{100 * len(budget_targets) / len(predict_df):.1f}%"
                    )
                
                with col2:
                    st.metric(
                        label="平均アップリフト", 
                        value=f"{avg_uplift_targeted:.4f}",
                    )
                
                with col3:
                    st.metric(
                        label="予測ROI", 
                        value=f"{budget_roi:.1f}%"
                    )
                
                # 顧客分布との比較
                st.subheader("アップリフト分布と選択された顧客")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 全体分布のヒストグラム
                ax.hist(predicted_uplift, bins=30, alpha=0.5, label='全顧客')
                
                # 予算内でターゲットとなる顧客のヒストグラム
                ax.hist(budget_targets['predicted_uplift'], bins=30, alpha=0.7, color='orange', label='ターゲット顧客')
                
                # 凡例と軸ラベル
                ax.legend()
                ax.set_xlabel('予測アップリフト')
                ax.set_ylabel('顧客数')
                ax.set_title('アップリフト分布とターゲット顧客')
                
                # ターゲット可能な最小アップリフト値
                min_targeted_uplift = budget_targets['predicted_uplift'].min()
                ax.axvline(x=min_targeted_uplift, color='r', linestyle='--', alpha=0.7)
                ax.annotate(f'最小アップリフト閾値: {min_targeted_uplift:.4f}', 
                           xy=(min_targeted_uplift, ax.get_ylim()[1]*0.9),
                           xytext=(min_targeted_uplift+0.01, ax.get_ylim()[1]*0.9),
                           arrowprops=dict(arrowstyle='->'))
                
                st.pyplot(fig)
                
                # ターゲット顧客リストのダウンロード
                st.subheader("ターゲット顧客リスト")
                
                # ダウンロード用に選択する列
                download_cols = selected_features + ['predicted_uplift']
                download_df = budget_targets[download_cols].reset_index(drop=True)
                
                # データフレームのプレビュー
                st.write(download_df.head())
                
                # CSVダウンロード
                csv = download_df.to_csv(index=False)
                st.download_button(
                    label="ターゲット顧客リストをCSVでダウンロード",
                    data=csv,
                    file_name=f"uplift_target_customers_{len(budget_targets)}.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()