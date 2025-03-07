import os
from pathlib import Path

# アプリケーションのベースディレクトリ
BASE_DIR = Path(__file__).parent.parent.absolute()

# モデル保存用ディレクトリ
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")

# データセット保存用ディレクトリ
DATA_DIR = os.path.join(BASE_DIR, "assets")

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

# API設定
API_TITLE = "Uplift Modeling API"
API_DESCRIPTION = "API for training and using uplift models"
API_VERSION = "0.1.0"
API_PORT = 8000

# ロギングの設定
LOG_LEVEL = "INFO"

# 起動時にディレクトリを作成
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
