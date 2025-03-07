# アップリフトモデリング デモアプリケーション

このリポジトリには、アップリフトモデリングのデモアプリケーションが含まれています。アップリフトモデリングとは、マーケティングキャンペーンなどの介入効果を個人レベルで予測するための機械学習手法です。

## 機能

- 複数のアップリフトモデリング手法（S-learner, T-learner, X-learner, R-learner, Causal Trees, Uplift Random Forest）の比較
- クイックデモ：少数のサンプルデータを使用した簡易デモンストレーション
- 本格デモ：Hillstromデータセットを使用した詳細分析
- ROI計算と最適ターゲティングの提案
- アップリフト予測結果のビジュアライゼーション

## セットアップ

### 前提条件

- Python 3.7以上
- pip または conda

### インストール

1. リポジトリをクローン：

```bash
git clone https://github.com/yourusername/uplift-modeling-demo.git
cd uplift-modeling-demo
```

2. 依存関係をインストール：

```bash
pip install -r requirements.txt
```

## 使い方

### APIサーバーの起動

アップリフトモデリングのバックエンドAPIを起動：

```bash
uvicorn app.api:app --reload
```

デフォルトでは、APIは http://localhost:8000 で実行されます。

### Streamlitアプリケーションの起動

フロントエンドのStreamlitアプリケーションを起動：

```bash
streamlit run app.py
```

デフォルトでは、アプリケーションは http://localhost:8501 で実行されます。

## アプリケーション構成

- `app.py`: メインのStreamlitアプリケーション
- `pages/`: Streamlitのマルチページ機能用のディレクトリ
  - `01_quick_demo.py`: 少数のサンプルデータを使用した簡易デモ
  - `02_full_demo.py`: Hillstromデータセットを使用した詳細分析
- `app/`: FastAPI用のバックエンドコード
  - `api.py`: APIエンドポイント定義
  - `schemas.py`: Pydanticモデル（リクエストとレスポンスのバリデーション）
- `models/`: アップリフトモデリング関連のコード
- `utils/`: ユーティリティ関数とヘルパークラス
- `client_utils/`: APIクライアント用のユーティリティ
- `assets/`: データセットなどの静的ファイル

## 依存ライブラリ

- streamlit: インタラクティブなWebアプリケーション作成用
- fastapi: 高性能なAPIフレームワーク
- uvicorn: ASGIサーバー
- pandas, numpy: データ処理用
- scikit-learn: 機械学習用
- causalml: アップリフトモデリング実装用
- matplotlib, seaborn, altair: データ可視化用

## ライセンス

MITライセンス
