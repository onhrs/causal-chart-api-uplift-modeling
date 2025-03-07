import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from client_utils.debug_helpers import safe_request
from utils.font_settings import set_matplotlib_japanize
import time

# 日本語フォント設定
set_matplotlib_japanize()

# ページ設定
st.set_page_config(
    page_title="Uplift Modeling Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# APIエンドポイントの設定
API_URL = os.environ.get("API_URL", "https://backend-api-620283975862.asia-northeast1.run.app")

# 利用可能なモデルタイプとその説明
MODEL_TYPES = {
    "s_learner": "S-Learner: 単一のモデルで結果を予測し、処置を特徴量として使用する簡潔な手法",
    "t_learner": "T-Learner: 処置群と対照群に別々のモデルを適用し、予測の差分を計算する直感的な手法",
    "x_learner": "X-Learner: T-Learnerを拡張し、処置効果を予測するための2段階推定を実施する手法",
    "r_learner": "R-Learner: 処置と結果の関係を別々に推定し、2つの予測を組み合わせる手法",
    "causal_tree": "Causal Tree: 決定木を使用して処置効果を直接推定する解釈性の高い手法",
    "uplift_rf": "Uplift Random Forest: 複数のツリーを組み合わせてアップリフトを予測する高精度な手法"
}

def check_api_connection():
    """APIの接続を確認し、接続状態とともにメッセージを表示"""
    with st.spinner("APIサーバーへの接続を確認中..."):
        try:
            # まずはCloud Runの起動を考慮して少し待機
            time.sleep(1)
            
            # /docsエンドポイントにアクセス
            response = requests.get(f"{API_URL}/docs", timeout=10)
            if response.status_code == 200:
                st.success(f"API接続成功！エンドポイント: {API_URL}")
                return True
            else:
                st.error(f"API接続失敗: ステータスコード {response.status_code}")
                st.info("Cloud Runサービスが起動中かもしれません。数秒後に再試行してください。")
                return False
        except requests.exceptions.ConnectionError:
            st.error(f"APIサーバーに接続できません: {API_URL}")
            st.info("バックエンドサービスが起動中かもしれません。しばらく待ってからページを更新してください。")
            return False
        except requests.exceptions.Timeout:
            st.warning(f"API接続がタイムアウトしました: {API_URL}")
            st.info("バックエンドが起動中の可能性があります。数秒後に再試行してください。")
            return False
        except Exception as e:
            st.error(f"API接続エラー: {str(e)}")
            return False

def main():
    # サイドバーの設定
    st.sidebar.title("Uplift Modeling Demo")
    st.sidebar.info("""
    このアプリケーションはアップリフトモデリングのデモです。
    ナビゲーションメニューから各デモページに移動できます。
    """)
    
    # メインコンテンツ
    st.title("アップリフトモデリング デモアプリケーション")
    
    st.markdown("""
    ## アップリフトモデリングとは？
    
    アップリフトモデリングとは、マーケティングや介入施策の**増分効果**を予測するための手法です。
    単なる結果予測ではなく、「介入を行った場合と行わなかった場合の差分」を推定します。
    
    ### 主な特徴
    
    - 処置の効果がある顧客セグメントの特定
    - マーケティング予算の効率的な配分
    - 介入効果の個別予測
    
    ### アプリケーションの使い方
    
    左のサイドバーから以下のデモページにアクセスできます：
    
    1. **クイックデモ**: 少数のサンプルデータを使った簡易デモ
    2. **本格デモ**: Hillstromデータセットを使った詳細な分析
    
    ### API接続状況
    """)
    
    # APIの接続確認
    api_connected = check_api_connection()
    
    # アップリフトモデリングの図解表示
    st.markdown("## アップリフトモデリングの概念図")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 従来の予測モデル
        予測の出力は「コンバージョンするかどうか」
        """)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title("従来の予測モデル")
        ax.text(5, 5, "顧客属性 → コンバージョン確率", ha='center', va='center', fontsize=12)
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        st.markdown("""
        ### アップリフトモデル
        予測の出力は「介入による効果の増分」
        """)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title("アップリフトモデル")
        ax.text(5, 7, "介入なしの場合のコンバージョン確率", ha='center', va='center', fontsize=10)
        ax.text(5, 5, "介入ありの場合のコンバージョン確率", ha='center', va='center', fontsize=10)
        ax.text(5, 3, "↓", ha='center', va='center', fontsize=14)
        ax.text(5, 1, "アップリフト = 効果の増分", ha='center', va='center', fontsize=12, weight='bold')
        ax.axis('off')
        st.pyplot(fig)
    
    # 利用可能なモデルタイプの解説
    st.markdown("## アップリフトモデリングの手法")
    st.write("本デモでは、以下のアップリフトモデリング手法を使用できます：")
    
    for model_type, description in MODEL_TYPES.items():
        st.markdown(f"**{model_type}**: {description}")

    
if __name__ == "__main__":
    main()
