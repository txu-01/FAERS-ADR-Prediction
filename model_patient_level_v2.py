# E:/MED/model_patient_level_v2.py
# 使用 v2 数据，XGBoost 5折CV，按F1选阈值；导出CV指标/特征重要性

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

IN_FILE = Path(r"E:/MED/ASCII/cleaned/patient_level_dataset_v2.csv")
OUT_DIR = Path(r"E:/MED/ASCII/cleaned/model_xgb_v2_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_FILE)
y = df["y_multiADR"].astype(int).values

# —— 数值与类别 —— #
exclude = {"CASEID","ADR_count","y_multiADR"}
num_feats = []
cat_feats = []
for c in df.columns:
    if c in exclude: 
        continue
    # 类别：sex, age_bin 这类
    if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
        if df[c].notna().sum() and df[c].dropna().nunique() >= 2:
            cat_feats.append(c)
    else:
        # 机制计数/0-1、计数型、连续型
        if df[c].notna().sum() > 0:
            num_feats.append(c)

X = df[num_feats + cat_feats].copy()

# 预处理（数值中位数填充；类别众数+one-hot）
pre = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_feats) if cat_feats else ("cat","drop",[])
    ],
    remainder="drop"
)

def get_feature_names(preproc):
    names = list(num_feats)
    if cat_feats:
        ohe = preproc.named_transformers_["cat"].named_steps["ohe"]
        names += list(ohe.get_feature_names_out(cat_feats))
    return names

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X))
oof_pred  = np.zeros(len(X), dtype=int)
metrics = []
fis = []

for fold, (tr,va) in enumerate(skf.split(X,y), start=1):
    print(f"\n=== Fold {fold}/5 ===")
    pre_fit = pre.fit(X.iloc[tr], y[tr])
    Xtr = pre_fit.transform(X.iloc[tr])
    Xva = pre_fit.transform(X.iloc[va])

    pos = np.sum(y[tr]==1); neg = np.sum(y[tr]==0)
    spw = neg / max(pos,1)

    clf = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.045,
        subsample=0.9, colsample_bytree=0.9,
        min_child_weight=2, reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist",
        scale_pos_weight=spw, random_state=42, n_jobs=-1
    )
    clf.fit(Xtr, y[tr])

    proba = clf.predict_proba(Xva)[:,1]
    oof_proba[va] = proba

    # 阈值按F1最优
    pre_arr, rec_arr, thr_arr = precision_recall_curve(y[va], proba)
    f1_arr = 2*pre_arr*rec_arr/(pre_arr+rec_arr+1e-12)
    best_i = np.nanargmax(f1_arr)
    thr = 0.5 if best_i>=len(thr_arr) else thr_arr[best_i]
    pred = (proba>=thr).astype(int)
    oof_pred[va] = pred

    auc = roc_auc_score(y[va], proba)
    ap  = average_precision_score(y[va], proba)
    f1  = f1_score(y[va], pred)
    metrics.append({"fold":fold,"AUC":auc,"PR-AUC":ap,"F1@best":f1,"thr":thr})

    # 特征重要性（gain）
    feat = get_feature_names(pre_fit)
    gain = clf.get_booster().get_score(importance_type="gain")
    fis.append(pd.DataFrame({"feature":[f for i,f in enumerate(feat)], "gain":[gain.get(f"f{i}",0.0) for i in range(len(feat))]}))

# 汇总
cv = pd.DataFrame(metrics)
cv.to_csv(OUT_DIR/"cv_metrics.csv", index=False)

oof_auc = roc_auc_score(y, oof_proba)
oof_ap  = average_precision_score(y, oof_proba)
cm = confusion_matrix(y, oof_pred)
rep = classification_report(y, oof_pred, digits=3)

with open(OUT_DIR/"oof_summary.txt","w",encoding="utf-8") as f:
    f.write(cv.to_string(index=False)+"\n")
    f.write(f"\nOOF AUC: {oof_auc:.4f} | OOF PR-AUC: {oof_ap:.4f}\n")
    f.write("Confusion matrix:\n"+str(cm)+"\n")
    f.write(rep)

fi_mean = pd.concat(fis, ignore_index=True).groupby("feature")["gain"].mean().sort_values(ascending=False).reset_index()
fi_mean.to_csv(OUT_DIR/"feature_importance_xgb_gain.csv", index=False)

print("\n=== CV metrics ===")
print(cv)
print(f"\nOOF AUC: {oof_auc:.3f} | OOF PR-AUC: {oof_ap:.3f}")
print("OOF Confusion matrix:\n", cm)
print(rep)
print("Saved to:", OUT_DIR.resolve())
