# model.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Union

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def train_uplift_model(
    df: pd.DataFrame,
    features: List[str],
    treatment_col: str,
    outcome_col: str,
    model_type: str = "s_learner",
    model_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    アップリフトモデルをトレーニングする関数
    
    Args:
        df: トレーニングデータを含むDataFrame
        features: 使用する特徴量のリスト
        treatment_col: 処置フラグを含む列名
        outcome_col: 結果変数を含む列名
        model_type: モデルタイプ ('s_learner', 't_learner', 'x_learner', 'r_learner', 'causal_tree', 'uplift_rf')
        model_params: モデルのパラメータ
        
    Returns:
        モデル情報を含む辞書
    """
    # デフォルトパラメータ
    if model_params is None:
        model_params = {}
    
    # 特徴量と目的変数の抽出
    X = df[features]
    y = df[outcome_col]
    
    # 処置変数の処理（文字列の場合は'control'と'treatment'に変換）
    if df[treatment_col].dtype == 'object' or df[treatment_col].dtype == 'bool':
        treatment = df[treatment_col].apply(lambda x: 'control' if x in [0, False, 'control', 'Control'] else 'treatment')
    else:
        treatment = df[treatment_col].apply(lambda x: 'control' if x == 0 else 'treatment')
    
    # データ分割（学習:テスト = 7:3）
    X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
        X, y, treatment, test_size=0.3, random_state=42
    )
    
    # モデルの選択とトレーニング
    model = None
    if model_type == "s_learner":
        base_learner = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 5),
            random_state=0
        )
        model = BaseSRegressor(learner=base_learner, control_name='control')
        model.fit(X_train, treatment=treat_train, y=y_train)
        
    elif model_type == "t_learner":
        base_learner = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 5),
            random_state=0
        )
        model = BaseTRegressor(learner=base_learner, control_name='control')
        model.fit(X_train, treatment=treat_train, y=y_train)
        
    elif model_type == "x_learner":
        base_learner = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 5),
            random_state=0
        )
        model = BaseXRegressor(learner=base_learner, control_name='control')
        model.fit(X_train, treatment=treat_train, y=y_train)
        
    elif model_type == "r_learner":
        outcome_learner = RandomForestRegressor(
            max_depth=model_params.get('max_depth', 5),
            random_state=0
        )
        effect_learner = RandomForestRegressor(
            max_depth=model_params.get('max_depth', 5),
            random_state=0
        )
        model = BaseRRegressor(
            outcome_learner=outcome_learner,
            effect_learner=effect_learner,
            control_name='control'
        )
        model.fit(X_train, treatment=treat_train, y=y_train)
        
    elif model_type == "causal_tree":
        model = UpliftTreeClassifier(
            control_name='control',
            max_depth=model_params.get('max_depth', 5),
            min_samples_leaf=model_params.get('min_samples_leaf', 100)
        )
        model.fit(X_train.values, treatment=treat_train.values, y=y_train.values)
        
    elif model_type == "uplift_rf":
        model = UpliftRandomForestClassifier(
            control_name='control',
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 5),
            random_state=0
        )
        model.fit(X_train.values, treatment=treat_train.values, y=y_train.values)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # テストデータでの評価
    if model_type in ["causal_tree", "uplift_rf"]:
        cate_pred = model.predict(X_test.values).flatten()
    else:
        cate_pred = model.predict(X_test)
    
    # 実測ATEの計算
    mean_conv_control = y_test[treat_test == 'control'].mean()
    mean_conv_treat = y_test[treat_test == 'treatment'].mean()
    actual_ate = mean_conv_treat - mean_conv_control
    
    # 推定ATEの計算
    estimated_ate = cate_pred.mean()
    
    # Qiniスコアの計算
    w_test = (treat_test == 'treatment').astype(int)
    
    # モデル情報の保存
    model_info = {
        "model": model,
        "model_type": model_type,
        "features": features,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics": {
            "actual_ate": float(actual_ate),
            "estimated_ate": float(estimated_ate),
            "control_outcome": float(mean_conv_control),
            "treatment_outcome": float(mean_conv_treat)
        }
    }
    
    return model_info

def predict_uplift(model_info: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """
    アップリフトモデルを使用して予測を行う関数
    
    Args:
        model_info: モデル情報を含む辞書
        X: 予測に使用する特徴量データ
        
    Returns:
        アップリフト予測値の配列
    """
    model = model_info["model"]
    model_type = model_info["model_type"]
    
    # モデルタイプに応じた予測
    if model_type in ["causal_tree", "uplift_rf"]:
        predictions = model.predict(X.values).flatten()
    else:
        predictions = model.predict(X)
    
    return predictions
