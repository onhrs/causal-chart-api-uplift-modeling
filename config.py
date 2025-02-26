# config.py
import os

# モデル保存パス
MODEL_DIR = "models"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "latest_model.pkl")

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

# APIの設定
API_TITLE = "Uplift Modeling API"
API_DESCRIPTION = "An API for training and using uplift models"
API_VERSION = "1.0.0"
