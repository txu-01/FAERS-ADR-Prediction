# E:/MED/model_patient_level.py
# 更稳健：自动丢弃“全缺失”的数值/类别特征，避免 OneHotEncoder 收到 0 列

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)

IN_FILE = Path(r"E:/MED/ASCII/cleaned/patient_level_dataset.csv")
OUT_DIR = Path(r"E:/MED/ASCII/cleaned/model_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) 读取数据
df = pd.read_csv(IN_FILE)

target = "y_multiADR"
assert target in df.columns, f"{target} 不在数据里"

# 候选数值特征
num_feats = [
    "n_diab_records", "n_diab_unique", "is_multi_drug",
    "mol_weight_mean","mol_weight_max","mol_weight_sum",
    "logP_mean","logP_max","logP_min",
    "TPSA_mean","TPSA_max","TPSA_sum",
    "HBD_mean","HBD_max","HBD_sum",
    "HBA_mean","HBA_max","HBA_sum",
    "age","weight","height"
]
num_feats = [c for c in num_feats if c in df.columns]

# 候选类别特征
cat_feats = [c for c in ["sex"] if c in df.columns]

# 可能的机制 one-hot 列（0/1 列）
possible_mech = [c for c in df.columns
                 if c not in (["CASEID","ADR_count",target] + num_feats + cat_feats)]
mech_feats = []
for c in possible_mech:
    vals = pd.Series(df[c].dropna().unique())
    if len(vals) > 0:
        try:
            if set(vals.astype(int)) <= {0,1}:
                mech_feats.append(c)
        except Exception:
            pass
num_feats += mech_feats

# —— 关键改动：丢弃“全缺失”的特征 —— #
def drop_all_nan(feats, frame):
    keep = []
    for c in feats:
        if c in frame.columns:
            if frame[c].notna().sum() > 0:  # 至少有一个非缺失
                keep.append(c)
    return keep

num_feats = drop_all_nan(num_feats, df)
cat_feats = drop_all_nan(cat_feats, df)

# 如果类别列仍存在，但“有效类别值”不足（只有一个或全空），也移除
final_cat_feats = []
for c in cat_feats:
    non_null = df[c].dropna()
    if non_null.nunique() >= 2:  # 需要至少两个类别才有意义
        final_cat_feats.append(c)
cat_feats = final_cat_feats

# 准备 X, y
X = df[num_feats + cat_feats].copy()
y = df[target].astype(int)

# 训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 数值预处理
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))
])

# 类别预处理（如果没有类别列，就设为 'drop'）
if len(cat_feats) > 0:
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
else:
    cat_transformer = "drop"

pre = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_feats),
        ("cat", cat_transformer, cat_feats) if len(cat_feats) > 0 else ("cat", "drop", [])
    ],
    remainder="drop"
)

# ---- 模型1：逻辑回归 ----
log_reg = Pipeline(steps=[
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
log_reg.fit(X_train, y_train)
proba_lr = log_reg.predict_proba(X_test)[:,1]
pred_lr = (proba_lr >= 0.5).astype(int)

# ---- 模型2：随机森林 ----
rf = Pipeline(steps=[
    ("pre", pre),
    ("clf", RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        random_state=42, class_weight="balanced_subsample", n_jobs=-1
    ))
])
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:,1]
pred_rf = (proba_rf >= 0.5).astype(int)

def summarize(name, y_true, proba, pred):
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    cm = confusion_matrix(y_true, pred)
    print(f"\n=== {name} ===")
    print(f"AUC: {auc:.3f} | PR-AUC: {ap:.3f}")
    print("Confusion matrix:\n", cm)
    print(classification_report(y_true, pred, digits=3))

summarize("LogisticRegression", y_test, proba_lr, pred_lr)
summarize("RandomForest", y_test, proba_rf, pred_rf)

# ---- 导出特征重要性/系数 ----
# 取预处理后特征名
num_cols = list(num_feats)
cat_cols = []
if len(cat_feats) > 0:
    ohe = log_reg.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    cat_cols = list(ohe.get_feature_names_out(cat_feats))
feature_names = num_cols + cat_cols

# 逻辑回归系数
lr_coef = log_reg.named_steps["clf"].coef_.ravel()
coef_df = pd.DataFrame({"feature": feature_names, "lr_coef": lr_coef}).sort_values("lr_coef", ascending=False)
coef_df.to_csv(OUT_DIR / "feature_coefficients_logreg.csv", index=False)

# 随机森林重要性
# 用 rf 的 feature_importances_（注意和特征名一一对应）
rf_model = rf.named_steps["clf"]
imp = pd.DataFrame({"feature": feature_names, "rf_importance": rf_model.feature_importances_}).sort_values("rf_importance", ascending=False)
imp.to_csv(OUT_DIR / "feature_importance_rf.csv", index=False)

print(f"\n✅ 已保存：\n- {OUT_DIR/'feature_coefficients_logreg.csv'}\n- {OUT_DIR/'feature_importance_rf.csv'}")
print(f"数值特征数: {len(num_feats)} | 类别特征数: {len(cat_feats)}")
