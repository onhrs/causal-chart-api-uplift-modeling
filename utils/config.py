import os
from pathlib import Path

# プロジェクトのベースディレクトリ
BASE_DIR = Path(__file__).resolve().parent.parent

# モデルを保存するディレクトリ
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# データ関連の設定
DEFAULT_FEATURES = ["Recency", "History", "Mens", "Womens", "Newbie"]
DEFAULT_TREATMENT_COL = "treatment"
DEFAULT_OUTCOME_COL = "Conversion"

# モデル関連の設定
DEFAULT_MODEL_TYPE = "s_learner"
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5
}

# APIの詳細
API_TITLE = "Uplift Modeling API"
API_DESCRIPTION = "アップリフトモデリングのためのAPI"
API_VERSION = "0.1.0"

# ロギングの設定
LOG_LEVEL = "INFO"
