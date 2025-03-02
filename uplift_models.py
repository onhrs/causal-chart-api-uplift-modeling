import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pickle
import os
import datetime
from abc import ABC, abstractmethod

class UpliftModelBase(ABC):
    """アップリフトモデルの基本クラス"""
    
    def __init__(self, model_type="base"):
        self.model_type = model_type
        self.features = None
        self.treatment_col = None
        self.outcome_col = None
        self.model = None
        self.model_params = {}
        self.metrics = {}
    
    @abstractmethod
    def fit(self, data, features, treatment_col, outcome_col, model_params=None):
        """モデルの学習"""
        pass
    
    @abstractmethod
    def predict(self, features):
        """アップリフト（処置効果）の予測"""
        pass
    
    def save_model(self, model_dir="models"):
        """モデルの保存"""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_type}_{timestamp}.pkl"
        model_path = os.path.join(model_dir, filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        
        return model_path
    
    @staticmethod
    def load_model(model_path):
        """モデルの読み込み"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def _calculate_metrics(self, data, treatment_col, outcome_col):
        """モデル評価指標の計算"""
        treatment = data[treatment_col].values
        outcome = data[outcome_col].values
        
        # 実際のATE（平均処置効果）を計算
        treatment_outcome = outcome[treatment == 1].mean()
        control_outcome = outcome[treatment == 0].mean()
        actual_ate = treatment_outcome - control_outcome
        
        # 推定したATEを計算
        X = data[self.features]
        estimated_uplift = self.predict(X)
        estimated_ate = estimated_uplift.mean()
        
        self.metrics = {
            "actual_ate": float(actual_ate),
            "estimated_ate": float(estimated_ate),
            "control_outcome": float(control_outcome),
            "treatment_outcome": float(treatment_outcome)
        }
        
        return self.metrics
    
    def get_model_info(self):
        """モデル情報の取得"""
        return {
            "model_type": self.model_type,
            "features": self.features,
            "metrics": self.metrics,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }


class SLearner(UpliftModelBase):
    """S-Learner（シングルモデル）アプローチ"""
    
    def __init__(self):
        super().__init__(model_type="s_learner")
    
    def fit(self, data, features, treatment_col, outcome_col, model_params=None):
        """S-Learnerモデルの学習
        処置変数を特徴量に加えた単一モデルを構築
        """
        self.features = features
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        if model_params is None:
            model_params = {"n_estimators": 100, "max_depth": 5}
        self.model_params = model_params
        
        # 特徴量と処置変数を結合
        X = data[features + [treatment_col]]
        y = data[outcome_col]
        
        # モデル構築
        if y.nunique() <= 2:  # 2値分類問題
            self.model = RandomForestClassifier(**model_params)
        else:  # 回帰問題
            self.model = RandomForestRegressor(**model_params)
        
        self.model.fit(X, y)
        
        # 評価指標の計算
        self._calculate_metrics(data, treatment_col, outcome_col)
        
        return self
    
    def predict(self, features):
        """アップリフト予測"""
        if self.model is None:
            raise ValueError("モデルがまだ学習されていません。")
        
        # 処置なしの予測
        X_control = features.copy()
        X_control[self.treatment_col] = 0
        y_pred_control = self.model.predict_proba(X_control)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X_control)
        
        # 処置ありの予測
        X_treatment = features.copy()
        X_treatment[self.treatment_col] = 1
        y_pred_treatment = self.model.predict_proba(X_treatment)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X_treatment)
        
        # アップリフトを計算
        uplift = y_pred_treatment - y_pred_control
        
        return uplift


class TLearner(UpliftModelBase):
    """T-Learner（2モデル）アプローチ"""
    
    def __init__(self):
        super().__init__(model_type="t_learner")
        self.model_treatment = None
        self.model_control = None
    
    def fit(self, data, features, treatment_col, outcome_col, model_params=None):
        """T-Learnerモデルの学習
        処置群と対照群に対して別々のモデルを構築
        """
        self.features = features
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        if model_params is None:
            model_params = {"n_estimators": 100, "max_depth": 5}
        self.model_params = model_params
        
        # データを処置群と対照群に分割
        treated = data[data[treatment_col] == 1]
        control = data[data[treatment_col] == 0]
        
        X_treated = treated[features]
        y_treated = treated[outcome_col]
        
        X_control = control[features]
        y_control = control[outcome_col]
        
        # 各群のモデル構築
        if y_treated.nunique() <= 2:  # 2値分類問題
            self.model_treatment = RandomForestClassifier(**model_params)
            self.model_control = RandomForestClassifier(**model_params)
        else:  # 回帰問題
            self.model_treatment = RandomForestRegressor(**model_params)
            self.model_control = RandomForestRegressor(**model_params)
        
        self.model_treatment.fit(X_treated, y_treated)
        self.model_control.fit(X_control, y_control)
        
        # 評価指標の計算
        self._calculate_metrics(data, treatment_col, outcome_col)
        
        return self
    
    def predict(self, features):
        """アップリフト予測"""
        if self.model_treatment is None or self.model_control is None:
            raise ValueError("モデルがまだ学習されていません。")
        
        # 処置群のモデルによる予測
        y_pred_treatment = self.model_treatment.predict_proba(features)[:, 1] if hasattr(self.model_treatment, 'predict_proba') else self.model_treatment.predict(features)
        
        # 対照群のモデルによる予測
        y_pred_control = self.model_control.predict_proba(features)[:, 1] if hasattr(self.model_control, 'predict_proba') else self.model_control.predict(features)
        
        # アップリフトを計算
        uplift = y_pred_treatment - y_pred_control
        
        return uplift


class XLearner(UpliftModelBase):
    """X-Learner（クロスモデル）アプローチ"""
    
    def __init__(self):
        super().__init__(model_type="x_learner")
        self.model_treatment = None
        self.model_control = None
        self.model_effect_treated = None
        self.model_effect_control = None
        self.propensity_model = None
    
    def fit(self, data, features, treatment_col, outcome_col, model_params=None):
        """X-Learnerモデルの学習
        T-learnerを拡張し、ヘテロジニアス効果を推定
        """
        self.features = features
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        if model_params is None:
            model_params = {"n_estimators": 100, "max_depth": 5}
        self.model_params = model_params
        
        # データを処置群と対照群に分割
        treated = data[data[treatment_col] == 1]
        control = data[data[treatment_col] == 0]
        
        X_treated = treated[features]
        y_treated = treated[outcome_col]
        
        X_control = control[features]
        y_control = control[outcome_col]
        
        # Step 1: T-Learnerと同様に基本モデルを構築
        if y_treated.nunique() <= 2:  # 2値分類問題
            self.model_treatment = RandomForestClassifier(**model_params)
            self.model_control = RandomForestClassifier(**model_params)
        else:  # 回帰問題
            self.model_treatment = RandomForestRegressor(**model_params)
            self.model_control = RandomForestRegressor(**model_params)
        
        self.model_treatment.fit(X_treated, y_treated)
        self.model_control.fit(X_control, y_control)
        
        # Step 2: 処置効果の予測
        # 処置群に対する対照群モデルの予測
        y_control_pred_treated = self.model_control.predict_proba(X_treated)[:, 1] if hasattr(self.model_control, 'predict_proba') else self.model_control.predict(X_treated)
        # 処置群の実際の結果と対照群モデルの予測の差
        d_treated = y_treated.values - y_control_pred_treated
        
        # 対照群に対する処置群モデルの予測
        y_treated_pred_control = self.model_treatment.predict_proba(X_control)[:, 1] if hasattr(self.model_treatment, 'predict_proba') else self.model_treatment.predict(X_control)
        # 処置群モデルの予測と対照群の実際の結果の差
        d_control = y_treated_pred_control - y_control.values
        
        # Step 3: 処置効果モデルの構築
        self.model_effect_treated = RandomForestRegressor(**model_params)
        self.model_effect_control = RandomForestRegressor(**model_params)
        
        self.model_effect_treated.fit(X_treated, d_treated)
        self.model_effect_control.fit(X_control, d_control)
        
        # Step 4: 傾向スコアモデルの構築（処置を受ける確率）
        X_all = data[features]
        y_treatment = data[treatment_col]
        self.propensity_model = RandomForestClassifier(**model_params)
        self.propensity_model.fit(X_all, y_treatment)
        
        # 評価指標の計算
        self._calculate_metrics(data, treatment_col, outcome_col)
        
        return self
    
    def predict(self, features):
        """アップリフト予測"""
        if self.model_effect_treated is None or self.model_effect_control is None:
            raise ValueError("モデルがまだ学習されていません。")
        
        # 処置効果の予測
        tau_treated = self.model_effect_treated.predict(features)
        tau_control = self.model_effect_control.predict(features)
        
        # 傾向スコアの予測
        propensity_scores = self.propensity_model.predict_proba(features)[:, 1]
        
        # 傾向スコアで重み付けされた処置効果の計算
        uplift = propensity_scores * tau_control + (1 - propensity_scores) * tau_treated
        
        return uplift


class RLearner(UpliftModelBase):
    """R-Learner（Robinson）アプローチ"""
    
    def __init__(self):
        super().__init__(model_type="r_learner")
        self.outcome_model = None
        self.propensity_model = None
        self.effect_model = None
    
    def fit(self, data, features, treatment_col, outcome_col, model_params=None):
        """R-Learnerモデルの学習
        ダブルロバスト推定に基づく手法
        """
        self.features = features
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        if model_params is None:
            model_params = {"n_estimators": 100, "max_depth": 5}
        self.model_params = model_params
        
        X = data[features]
        T = data[treatment_col].values
        Y = data[outcome_col].values
        
        # Step 1: 結果変数のモデル構築（m(x) = E[Y|X=x]）
        self.outcome_model = RandomForestRegressor(**model_params)
        self.outcome_model.fit(X, Y)
        m_x = self.outcome_model.predict(X)
        
        # Step 2: 傾向スコアモデルの構築（e(x) = P[T=1|X=x]）
        self.propensity_model = RandomForestClassifier(**model_params)
        self.propensity_model.fit(X, T)
        e_x = self.propensity_model.predict_proba(X)[:, 1]
        
        # Step 3: 残差の計算
        residual_y = Y - m_x
        residual_t = T - e_x
        
        # Step 4: 処置効果モデルの構築
        self.effect_model = RandomForestRegressor(**model_params)
        
        # 重み付き最小二乗法のための重み計算
        weights = residual_t ** 2
        
        # 重み付きサンプルでモデル学習
        self.effect_model.fit(X, residual_y / residual_t, sample_weight=weights)
        
        # 評価指標の計算
        self._calculate_metrics(data, treatment_col, outcome_col)
        
        return self
    
    def predict(self, features):
        """アップリフト予測"""
        if self.effect_model is None:
            raise ValueError("モデルがまだ学習されていません。")
        
        # 処置効果の予測
        tau_x = self.effect_model.predict(features)
        
        return tau_x


class CausalTree(UpliftModelBase):
    """Causal Tree（因果木）アプローチ"""
    
    def __init__(self):
        super().__init__(model_type="causal_tree")
        self.tree = None
    
    def fit(self, data, features, treatment_col, outcome_col, model_params=None):
        """Causal Treeモデルの学習
        因果効果に基づき木を構築
        """
        self.features = features
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        if model_params is None:
            model_params = {"max_depth": 5}
        self.model_params = model_params
        
        X = data[features]
        T = data[treatment_col].values
        Y = data[outcome_col].values
        
        # 簡易的なCausal Tree実装：
        # 差分を目標変数として決定木を学習
        
        # 処置群と対照群に分割
        X_treated = X[T == 1]
        Y_treated = Y[T == 1]
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        
        # Difference-in-Differences風の目標変数構築
        # (より洗練された実装では、匹配やプロペンシティスコアを使用します)
        y_diff = np.mean(Y_treated) - np.mean(Y_control)
        
        # 決定木を学習
        self.tree = DecisionTreeRegressor(**model_params)
        self.tree.fit(X, T * (Y - np.mean(Y_control)) + (1 - T) * (np.mean(Y_treated) - Y))
        
        # 評価指標の計算
        self._calculate_metrics(data, treatment_col, outcome_col)
        
        return self
    
    def predict(self, features):
        """アップリフト予測"""
        if self.tree is None:
            raise ValueError("モデルがまだ学習されていません。")
        
        # 処置効果の予測
        tau_x = self.tree.predict(features)
        
        return tau_x


class UpliftRandomForest(UpliftModelBase):
    """Uplift Random Forest（アップリフトランダムフォレスト）アプローチ"""
    
    def __init__(self):
        super().__init__(model_type="uplift_random_forest")
        self.forest = None
        self.n_trees = 10
    
    def fit(self, data, features, treatment_col, outcome_col, model_params=None):
        """Uplift Random Forestモデルの学習
        複数のCausal Treeによるアンサンブル
        """
        self.features = features
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        if model_params is None:
            model_params = {"max_depth": 5, "n_trees": 10}
        self.model_params = model_params
        self.n_trees = model_params.get("n_trees", 10)
        
        # 木のパラメータからn_treesを削除
        tree_params = model_params.copy()
        if "n_trees" in tree_params:
            del tree_params["n_trees"]
        
        X = data[features]
        T = data[treatment_col].values
        Y = data[outcome_col].values
        
        # 複数のCausal Treeを学習
        self.forest = []
        for _ in range(self.n_trees):
            # ブートストラップサンプリング
            idx = np.random.choice(len(X), len(X), replace=True)
            bootstrap_data = data.iloc[idx]
            
            # Causal Treeを構築
            tree = CausalTree()
            tree.fit(bootstrap_data, features, treatment_col, outcome_col, tree_params)
            
            self.forest.append(tree)
        
        # 評価指標の計算
        self._calculate_metrics(data, treatment_col, outcome_col)
        
        return self
    
    def predict(self, features):
        """アップリフト予測"""
        if self.forest is None or len(self.forest) == 0:
            raise ValueError("モデルがまだ学習されていません。")
        
        # 各木の予測を収集
        predictions = np.array([tree.predict(features) for tree in self.forest])
        
        # 平均予測を計算
        uplift = np.mean(predictions, axis=0)
        
        return uplift


def get_model_class(model_type):
    """モデルタイプに応じたクラスを返す"""
    model_map = {
        "s_learner": SLearner,
        "t_learner": TLearner,
        "x_learner": XLearner,
        "r_learner": RLearner,
        "causal_tree": CausalTree,
        "uplift_random_forest": UpliftRandomForest
    }
    
    if model_type not in model_map:
        raise ValueError(f"サポートされていないモデルタイプ: {model_type}")
    
    return model_map[model_type]()
