# 改良版
# https://chatgpt.com/share/67bf207f-3fd4-8000-aa54-e2c57cd13e86

import pandas as pd
import numpy as np
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.metrics import qini_score, plot_qini
import matplotlib.pyplot as plt

# ==========
# 1. データ読み込み
# ==========
df = pd.read_csv("hillstrom.csv")  # CSVパスは適宜修正
print("データ件数:", len(df))
df.head()

# ==========
# 2. 前処理
# ==========
# 大文字始まりにそろえる (例: 'recency' -> 'Recency')
df.columns = [col.capitalize() for col in df.columns]

# 処置フラグ(1/0) とグループラベル(control/treatment)を作成
df['treatment'] = (df['Segment'] != 'No E-Mail').astype(int)
df['group'] = df['Segment'].apply(lambda x: 'control' if x == 'No E-Mail' else 'treatment')

# 分析で使う特徴量例
features = ['Recency', 'History', 'Mens', 'Womens', 'Newbie']
X = df[features]
y = df['Conversion']
treatment = df['group']  # 'control' or 'treatment'

# ==========
# 3. データ分割（学習:テスト = 7:3）
# ==========
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    X, y, treatment, test_size=0.3, stratify=df['treatment'], random_state=42
)
print("訓練データ件数:", len(X_train), "  テストデータ件数:", len(X_test))
print("訓練データ - 処置群割合:", (treat_train == 'treatment').mean().round(3))
print("テストデータ - 処置群割合:", (treat_test == 'treatment').mean().round(3))

# ==========
# 4. メタラーナー(S/T/X/R)による推定
# ==========
# S-learner
s_learner = BaseSRegressor(
    learner=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0),
    control_name='control'
)
s_learner.fit(X_train, treatment=treat_train, y=y_train)
cate_s = s_learner.predict(X_test)

# T-learner
t_learner = BaseTRegressor(
    learner=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0),
    control_name='control'
)
t_learner.fit(X_train, treatment=treat_train, y=y_train)
cate_t = t_learner.predict(X_test)

# X-learner
x_learner = BaseXRegressor(
    learner=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0),
    control_name='control'
)
x_learner.fit(X_train, treatment=treat_train, y=y_train)
cate_x = x_learner.predict(X_test)

# R-learner
r_learner = BaseRRegressor(
    outcome_learner=RandomForestRegressor(max_depth=5, random_state=0),
    effect_learner=RandomForestRegressor(max_depth=5, random_state=0),
    control_name='control'
)
r_learner.fit(X_train, treatment=treat_train, y=y_train)
cate_r = r_learner.predict(X_test)

# ==========
# 5. 因果木(UpliftTree) / RF(UpliftRandomForest)による推定
# ==========
uplift_tree = UpliftTreeClassifier(control_name='control', max_depth=5, min_samples_leaf=100)
uplift_tree.fit(X_train.values, treatment=treat_train.values, y=y_train.values)
uplift_tree_cate = uplift_tree.predict(X_test.values).flatten()

uplift_forest = UpliftRandomForestClassifier(control_name='control', random_state=0)
uplift_forest.fit(X_train.values, treatment=treat_train.values, y=y_train.values)
uplift_forest_cate = uplift_forest.predict(X_test.values).flatten()

# ==========
# 6. 実測ATEとの比較
# ==========
mean_conv_control = y_test[treat_test == 'control'].mean()
mean_conv_treat = y_test[treat_test == 'treatment'].mean()
actual_ate = mean_conv_treat - mean_conv_control
print("テストデータ実測ATE:", actual_ate.round(5),
      f"(対照: {mean_conv_control.round(5)}, 処置: {mean_conv_treat.round(5)})")

print("推定ATE(S-learner):", cate_s.mean().round(5))
print("推定ATE(T-learner):", cate_t.mean().round(5))
print("推定ATE(X-learner):", cate_x.mean().round(5))
print("推定ATE(R-learner):", cate_r.mean().round(5))
print("推定ATE(Causal Tree):", uplift_tree_cate.mean().round(5))
print("推定ATE(Uplift RF):", uplift_forest_cate.mean().round(5))

# ==========
# 7. Qiniスコアの計算
# ==========
# w_test を0/1に変換
w_test = (treat_test == 'treatment').astype(int)

# 予測結果をまとめる辞書
models = {
    'S-learner': cate_s,
    'T-learner': cate_t,
    'X-learner': cate_x,
    'R-learner': cate_r,
    'CausalTree': uplift_tree_cate,
    'UpliftRF': uplift_forest_cate
}

for name, cate_pred in models.items():
    # cate_predが多次元の場合はflattenしておく
    if cate_pred.ndim > 1:
        cate_pred = cate_pred.flatten()
    # 必要ならノイズを微小に加える
    cate_pred = cate_pred + np.random.normal(0, 1e-6, size=len(cate_pred))

    df_eval = pd.DataFrame({
        'y': y_test.values,
        'w': w_test.values,
        'tau': cate_pred
    })
    # Qiniスコア計算
    score = qini_score(df_eval, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True)
    print(f"Qiniスコア({name}): {score:.4f}")

# Qiniカーブのプロット例（1モデルだけやるサンプル）
df_qini = pd.DataFrame({
    'y': y_test.values,
    'w': w_test.values,
    'pred_uplift': cate_s  # ここでは S-learnerの結果を例示
})
score_s = qini_score(df_qini, outcome_col='y', treatment_col='w')
print(f"\n[S-learner] Qiniスコア(非normalize): {score_s:.4f}")

plot_qini(df_qini, outcome_col='y', treatment_col='w')
plt.title("Qini Curve - S-learner")
plt.show()
