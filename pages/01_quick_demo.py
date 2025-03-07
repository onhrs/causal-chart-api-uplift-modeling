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
from utils.font_settings import set_matplotlib_japanize

# 日本語フォント設定
set_matplotlib_japanize()

# APIエンドポイントの設定
API_URL = os.environ.get("API_URL", "https://causal-chart-api-620283975862.asia-northeast1.run.app")

# サンプルデータ
default_data = [
    {"Recency": 10, "History": 100, "Mens": 1, "Womens": 0, "Newbie": 0, "treatment": 1, "Conversion": 1},
    {"Recency": 5, "History": 200, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 0},
    {"Recency": 7, "History": 150, "Mens": 1, "Womens": 1, "Newbie": 1, "treatment": 1, "Conversion": 1},
    {"Recency": 3, "History": 300, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 1},
    {"Recency": 8, "History": 220, "Mens": 1, "Womens": 0, "Newbie": 1, "treatment": 1, "Conversion": 0},
    {"Recency": 2, "History": 180, "Mens": 0, "Womens": 0, "Newbie": 1, "treatment": 0, "Conversion": 0}
]

# ページ設定
st.set_page_config(
    page_title="Quick Demo - Uplift Modeling",
    page_icon="🚀",
    layout="wide"
)

def plot_conversion_rates(df):
    # 処置群と対照群のコンバージョン率を計算
    treatment_conv = df[df['treatment'] == 1]['Conversion'].mean()
    control_conv = df[df['treatment'] == 0]['Conversion'].mean()
    uplift = treatment_conv - control_conv
    
    # グラフの描画
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['対照群', '処置群'], [control_conv, treatment_conv], color=['#1f77b4', '#ff7f0e'])
    
    # アップリフトを表示
    ax.plot([0, 1], [control_conv, treatment_conv], 'k--', alpha=0.7)
    ax.annotate(f'アップリフト: {uplift:.3f}', 
                xy=(0.5, (control_conv + treatment_conv) / 2),
                xytext=(0.5, (control_conv + treatment_conv) / 2 + 0.1),
                ha='center', va='bottom', fontsize=12,
                arrowprops=dict(arrowstyle='->'))
    
    # グラフの設定
    ax.set_ylim(0, 1)
    ax.set_ylabel('コンバージョン率')
    ax.set_title('処置群と対照群のコンバージョン率比較')
    
    # 値のラベル付け
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    return fig

def predict_with_model(model_path, features_data):
    # モデルを使って予測
    try:
        predict_payload = {
            "features": features_data,
            "model_path": model_path
        }
        
        with st.spinner('予測中...'):
            predict_response = safe_request("POST", f"{API_URL}/predict", predict_payload, debug=False)
        
        return predict_response
    
    except Exception as e:
        st.error(f"予測エラー: {str(e)}")
        return None

def main():
    st.title("クイックデモ: アップリフトモデリング")
    
    st.markdown("""
    このデモでは、少数のサンプルデータを使用してアップリフトモデリングの基本的な機能を試すことができます。
    データを編集し、モデルをトレーニングして予測結果を確認してみましょう。
    """)
    
    # セッション状態の初期化
    if 'demo_data' not in st.session_state:
        st.session_state.demo_data = default_data.copy()
    
    if 'trained_model_path' not in st.session_state:
        st.session_state.trained_model_path = None
        
    if 'model_info' not in st.session_state:
        st.session_state.model_info = None
    
    # タブの作成
    tab1, tab2, tab3 = st.tabs(["データの準備", "モデルトレーニング", "予測と評価"])
    
    with tab1:
        st.header("トレーニングデータ")
        
        # データフレーム表示と編集
        st.markdown("以下のデータを編集してトレーニングに使用します。")
        
        # データフレーム編集用のデータコピー
        edited_df = st.data_editor(
            pd.DataFrame(st.session_state.demo_data),
            use_container_width=True,
            num_rows="dynamic"
        )
        
        # 編集されたデータを保存
        if st.button("データを保存", key="save_data"):
            st.session_state.demo_data = edited_df.to_dict('records')
            st.success("データが保存されました！")
            
        # 処置群と対照群のコンバージョン率比較のグラフ表示
        if len(edited_df) > 0:
            st.subheader("データの基本統計")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 処置群と対照群のサンプル数
                treatment_count = len(edited_df[edited_df['treatment'] == 1])
                control_count = len(edited_df[edited_df['treatment'] == 0])
                
                st.metric(label="処置群サンプル数", value=treatment_count)
                st.metric(label="対照群サンプル数", value=control_count)
            
            with col2:
                # コンバージョン率の集計
                treatment_conv = edited_df[edited_df['treatment'] == 1]['Conversion'].mean()
                control_conv = edited_df[edited_df['treatment'] == 0]['Conversion'].mean()
                
                st.metric(label="処置群コンバージョン率", value=f"{treatment_conv:.2%}")
                st.metric(label="対照群コンバージョン率", value=f"{control_conv:.2%}")
                st.metric(label="単純アップリフト", value=f"{treatment_conv - control_conv:.2%}", 
                          delta=f"{treatment_conv - control_conv:.3f}")
            
            # グラフの表示
            st.pyplot(plot_conversion_rates(edited_df))
    
    with tab2:
        st.header("モデルトレーニング")
        
        # モデルタイプの選択
        model_type = st.selectbox(
            "モデルタイプ",
            ["s_learner", "t_learner", "x_learner", "r_learner", "causal_tree", "uplift_rf"],
            help="使用するアップリフトモデリングの手法を選択します。"
        )
        
        # モデルパラメータの設定
        st.subheader("モデルパラメータ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("n_estimators (ツリーの数)", min_value=10, max_value=200, value=50)
        
        with col2:
            max_depth = st.slider("max_depth (最大深さ)", min_value=1, max_value=10, value=3)
        
        # トレーニングボタン
        if st.button("モデルをトレーニング", key="train_model"):
            if len(edited_df) < 4:
                st.error("トレーニングには最低4件のデータが必要です。")
            else:
                try:
                    # トレーニングリクエストの準備
                    train_payload = {
                        "data": st.session_state.demo_data,
                        "features": ["Recency", "History", "Mens", "Womens", "Newbie"],
                        "treatment_col": "treatment",
                        "outcome_col": "Conversion",
                        "model_type": model_type,
                        "model_params": {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth
                        }
                    }
                    
                    # APIリクエスト送信
                    with st.spinner('モデルトレーニング中...'):
                        train_response = safe_request("POST", f"{API_URL}/train", train_payload, debug=False)
                    
                    if train_response and "model_path" in train_response:
                        st.session_state.trained_model_path = train_response["model_path"]
                        st.session_state.model_info = train_response["model_info"]
                        
                        st.success(f"モデルが正常にトレーニングされました！ モデルID: {train_response['model_path']}")
                        
                        # モデル情報の表示
                        st.subheader("モデル情報")
                        st.json(train_response["model_info"])
                        
                        # メトリクスの表示
                        metrics = train_response["model_info"].get("metrics", {})
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("実際のATE", f"{metrics.get('actual_ate', 0): .4f}")
                        
                        with col2:
                            st.metric("推定ATE", f"{metrics.get('estimated_ate', 0): .4f}")
                        
                        with col3:
                            st.metric("対照群結果", f"{metrics.get('control_outcome', 0): .4f}")
                        
                        with col4:
                            st.metric("処置群結果", f"{metrics.get('treatment_outcome', 0): .4f}")
                    else:
                        st.error("モデルトレーニングに失敗しました。")
                
                except Exception as e:
                    st.error(f"トレーニングエラー: {str(e)}")
        
        # トレーニング済みモデルの表示
        if st.session_state.trained_model_path:
            st.info(f"最後にトレーニングしたモデル: {st.session_state.trained_model_path} ({st.session_state.model_info['model_type']})")
    
    with tab3:
        st.header("予測と評価")
        
        if not st.session_state.trained_model_path:
            st.warning("先にモデルをトレーニングしてください。")
        else:
            st.subheader("新しいデータでの予測")
            
            # サンプルデータの作成
            new_samples = [
                {"Recency": 8, "History": 150, "Mens": 1, "Womens": 0, "Newbie": 1},
                {"Recency": 3, "History": 250, "Mens": 0, "Womens": 1, "Newbie": 0},
                {"Recency": 6, "History": 200, "Mens": 1, "Womens": 1, "Newbie": 0},
                {"Recency": 4, "History": 120, "Mens": 0, "Womens": 0, "Newbie": 1}
            ]
            
            # 新しいデータを編集可能に表示
            edited_new_data = st.data_editor(
                pd.DataFrame(new_samples),
                use_container_width=True,
                num_rows="dynamic"
            )
            
            # 予測実行
            if st.button("予測を実行", key="run_prediction"):
                prediction_result = predict_with_model(
                    st.session_state.trained_model_path,
                    edited_new_data.to_dict('records')
                )
                
                if prediction_result and "predictions" in prediction_result:
                    # 予測結果の表示
                    predictions = prediction_result["predictions"]
                    result_df = edited_new_data.copy()
                    result_df["予測アップリフト"] = predictions
                    
                    st.subheader("予測結果")
                    st.dataframe(result_df)
                    
                    # アップリフト予測値のソート
                    result_df_sorted = result_df.sort_values(by="予測アップリフト", ascending=False)
                    
                    # グラフ表示
                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = ['#ff9999' if x > 0 else '#99ccff' for x in predictions]
                    bars = ax.bar(range(len(predictions)), predictions, color=colors)
                    
                    # ゼロラインの追加
                    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    
                    # ラベル設定
                    ax.set_xlabel('サンプルID')
                    ax.set_ylabel('予測アップリフト')
                    ax.set_title('各顧客の予測アップリフト')
                    
                    # 値のラベル付け
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., 
                                height + 0.001 if height > 0 else height - 0.003,
                                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top')
                    
                    st.pyplot(fig)
                    
                    # ターゲティング推奨
                    st.subheader("ターゲティング推奨")
                    
                    positive_uplift = result_df[result_df["予測アップリフト"] > 0]
                    if len(positive_uplift) > 0:
                        st.success(f"以下の {len(positive_uplift)} 件の顧客にキャンペーンを実施することで、最適な結果が期待できます:")
                        st.dataframe(positive_uplift)
                    else:
                        st.warning("アップリフトが正の顧客は見つかりませんでした。キャンペーンの効果が期待できません。")
                    
                    negative_uplift = result_df[result_df["予測アップリフト"] < 0]
                    if len(negative_uplift) > 0:
                        st.error(f"以下の {len(negative_uplift)} 件の顧客にキャンペーンを実施すると、逆効果になる可能性があります:")
                        st.dataframe(negative_uplift)

if __name__ == "__main__":
    main()
