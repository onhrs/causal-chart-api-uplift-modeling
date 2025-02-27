# Uplift Modeling API

このAPIは、アップリフトモデリングを実行するためのRESTful APIです。マーケティングキャンペーンなどの介入効果を個人レベルで予測するために使用できます。

## 概要

アップリフトモデリングは、処置（介入）が個々のユーザーに与える因果効果を推定する手法です。このAPIを使用すると、様々なアップリフトモデリング手法を用いて、どのユーザーに対してどの程度の効果が期待できるかを予測できます。

## 機能

- 様々なアップリフトモデリング手法のトレーニング
  - S-learner: 処置変数を特徴量に加えた単一モデルを構築
  - T-learner: 処置群と対照群に対して別々のモデルを構築
  - X-learner: T-learnerを拡張し、ヘテロジニアス効果を推定
  - R-learner: ダブルロバスト推定に基づく手法
  - Causal Tree: 因果効果に基づき木を構築
  - Uplift Random Forest: 複数のCausal Treeによるアンサンブル
- モデルを使用した予測
- モデル情報の取得
- 利用可能なモデルの一覧表示

## インストール方法

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/causal-chart-api-uplift-modeling.git
cd causal-chart-api-uplift-modeling

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### APIの起動

```bash
uvicorn app:app --reload
```

これにより、APIが http://localhost:8000 で起動します。

### APIドキュメンテーション

APIドキュメントには以下のURLでアクセスできます：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## APIエンドポイント

### トレーニング
`POST /train`

アップリフトモデルをトレーニングします。

**リクエスト例:**
```json
{
  "data": [
    {"Recency": 10, "History": 100, "Mens": 1, "Womens": 0, "Newbie": 0, "treatment": 1, "Conversion": 1},
    {"Recency": 5, "History": 200, "Mens": 0, "Womens": 1, "Newbie": 0, "treatment": 0, "Conversion": 0},
    ...
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
```

**レスポンス例:**
```json
{
  "message": "s_learner model trained successfully",
  "model_path": "models/s_learner_20230525_123456.pkl",
  "model_info": {
    "model_type": "s_learner",
    "features": ["Recency", "History", "Mens", "Womens", "Newbie"],
    "metrics": {
      "actual_ate": 0.05,
      "estimated_ate": 0.048,
      "control_outcome": 0.12,
      "treatment_outcome": 0.17
    },
    "timestamp": "20230525_123456"
  }
}
```

### 予測
`POST /predict`

トレーニング済みのアップリフトモデルを使用して予測を行います。

**リクエスト例:**
```json
{
  "features": [
    {"Recency": 10, "History": 100, "Mens": 1, "Womens": 0, "Newbie": 0},
    {"Recency": 5, "History": 200, "Mens": 0, "Womens": 1, "Newbie": 0},
    ...
  ],
  "model_path": "models/s_learner_20230525_123456.pkl"
}
```

**レスポンス例:**
```json
{
  "predictions": [0.12, 0.05, ...],
  "model_type": "s_learner"
}
```

### モデル一覧
`GET /models`

利用可能なモデルの一覧を取得します。

**レスポンス例:**
```json
{
  "models": [
    {
      "model_path": "models/s_learner_20230525_123456.pkl",
      "model_type": "s_learner",
      "features": ["Recency", "History", "Mens", "Womens", "Newbie"],
      "timestamp": "20230525_123456",
      "metrics": {
        "actual_ate": 0.05,
        "estimated_ate": 0.048,
        "control_outcome": 0.12,
        "treatment_outcome": 0.17
      }
    },
    ...
  ]
}
```

### モデル詳細
`GET /model/{model_path}`

特定のモデルの詳細情報を取得します。

**レスポンス例:**
```json
{
  "model_path": "models/s_learner_20230525_123456.pkl",
  "model_type": "s_learner",
  "features": ["Recency", "History", "Mens", "Womens", "Newbie"],
  "timestamp": "20230525_123456",
  "metrics": {
    "actual_ate": 0.05,
    "estimated_ate": 0.048,
    "control_outcome": 0.12,
    "treatment_outcome": 0.17
  }
}
```

## 各アップリフトモデルの特徴

### S-learner (Single-Learner)
処置変数を他の特徴量と共に説明変数として使用し、単一のモデルを構築します。実装が単純ですが、処置効果の推定精度が他の手法に比べて低い場合があります。

### T-learner (Two-Learner)
処置群と対照群に対して別々のモデルを構築します。個別の予測値の差分により処置効果を推定します。群ごとのサンプルサイズが小さい場合には注意が必要です。

### X-learner (Cross-Learner)
処置効果の推定を交差させることで、T-learnerを改良した手法です。特に処置群と対照群のサイズが偏っている場合に有効です。

### R-learner (Robinson-Learner)
ダブルロバスト推定に基づき、処置効果の推定精度を向上させます。複雑なモデルですが、適切に実装すれば高い精度が期待できます。

### Causal Tree
処置効果に基づいて木を構築するアルゴリズムです。解釈性が高く、ユーザーセグメントごとの処置効果の違いを確認しやすい特徴があります。

### Uplift Random Forest
複数のCausal Treeによるアンサンブル手法です。単一の木より安定した予測が可能ですが、解釈性は低下します。

## データフォーマット

トレーニングデータは以下の列を含むことを推奨します：

- 特徴量: ユーザーの属性や行動データ
- 処置フラグ: 1/0または'treatment'/'control'などの処置の有無を表す変数
- 結果変数: コンバージョンや購入額などの目標変数

## 参考文献

- Gutierrez, P., & Gérardy, J. Y. (2017). Causal Inference and Uplift Modeling: A review of the literature.
- Kunzel, S. R., et al. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning.
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects.

## ライセンス

MIT License
