# train_baseline_common.py
# C1: Baseline 多标签模型（仅表格特征，含：不平衡处理 + 验证集阈值调优 + micro/macro 指标）
# - 将长表 X 压成“每个 PRIMARYID 一行”，与 Y 对齐
# - 选择疑似主药：ROLE_COD 以 PS > SS > C 优先
# - 选择 Top-50 ADR 作为标签
# - LightGBM OVR 训练，每个标签在验证集上调最优阈值
# - 输出 micro/macro 的 F1 / AUC、预测文件、最佳阈值

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier
import joblib

# ========= 0) 路径配置（按需修改） =========
OUTDIR = Path(r"E:\MED\outputs")
CLEAN_DIR = Path(r"E:\MED\ASCII\cleaned")
EXT_CSV = Path(r"E:\MED\drug_list_ext.csv")

X_PATH = OUTDIR / "X_patient_level.parquet"
Y_PATH = OUTDIR / "Y_multilabel_PT.parquet"
DRUG_PATH = CLEAN_DIR / "drug_clean.parquet"

# ========= 1) 读取数据 =========
X_raw = pd.read_parquet(X_PATH)
Y = pd.read_parquet(Y_PATH)
drug = pd.read_parquet(DRUG_PATH)
ext = pd.read_csv(EXT_CSV)

# ========= 2) 药名映射（为挑主药做准备） =========
def standardize(s):
    if pd.isna(s): return s
    return str(s).strip().upper().replace('-', ' ').replace('_',' ')

keys = {k.upper(): k for k in ext['drug_name'].astype(str)}
syn = {
    'OZEMPIC':'Semaglutide','RYBELSUS':'Semaglutide','WEGOVY':'Semaglutide',
    'TRULICITY':'Dulaglutide','BYDUREON':'Exenatide','BYETTA':'Exenatide',
    'MOUNJARO':'Tirzepatide','JARDIANCE':'Empagliflozin',
    'FORXIGA':'Dapagliflozin','FARXIGA':'Dapagliflozin',
    'INVOKANA':'Canagliflozin','JANUVIA':'Sitagliptin',
    'TRADJENTA':'Linagliptin','ONGLYZA':'Saxagliptin',
    'GLUCOPHAGE':'Metformin','ACTOS':'Pioglitazone',
}
def map_name(x):
    xn = standardize(x)
    if pd.isna(xn) or xn == "": return np.nan
    if xn in keys: return keys[xn]
    tok = xn.split()[0]
    if tok in keys: return keys[tok]
    if tok in syn: return syn[tok]
    for ku, ko in keys.items():
        if ku in xn: return ko
    return np.nan

drug['DRUG_STD'] = drug['DRUGNAME'].apply(standardize)
drug['drug_mapped'] = drug['DRUG_STD'].apply(map_name)

# ========= 3) 为每个 PRIMARYID 选“疑似主药”（PS > SS > C）并与 X 匹配 =========
role_rank = {'PS':0, 'SS':1, 'C':2}
drug['role_rank'] = drug['ROLE_COD'].map(role_rank).fillna(99)

cand = (drug.dropna(subset=['drug_mapped'])
              .sort_values(['PRIMARYID','role_rank'])
              .drop_duplicates(subset=['PRIMARYID'], keep='first')
              [['PRIMARYID','drug_mapped']]
              .rename(columns={'drug_mapped':'picked_drug'}))

Xm = X_raw.merge(cand, on='PRIMARYID', how='inner')
X_view = Xm[Xm['drug_mapped'] == Xm['picked_drug']].copy()
X_view = X_view.sort_values('PRIMARYID').drop_duplicates('PRIMARYID', keep='first')

# ========= 4) 与 Y 对齐（保证行数一致） =========
if Y.index.name != "PRIMARYID":
    if "PRIMARYID" in Y.columns:
        Y = Y.set_index("PRIMARYID")
    else:
        raise ValueError("Y 没有 PRIMARYID 索引/列，请检查 Y_multilabel_PT.parquet")

mask = X_view['PRIMARYID'].astype(str).isin(Y.index.astype(str))
X_view = X_view.loc[mask].copy()
Y_view = Y.loc[X_view['PRIMARYID'].astype(str)].copy()
X_view = X_view.set_index('PRIMARYID')

print("Aligned shapes => X_view:", X_view.shape, " Y_view:", Y_view.shape)

# ========= 5) 选择 Top-50 ADR 标签 =========
label_counts = Y_view.sum().sort_values(ascending=False)
TOPK = 50
top_labels = label_counts.head(TOPK).index
Y_view = Y_view[top_labels]

# ========= 6) 特征工程（仅表格特征） =========
num_cols = ["AGE","mol_weight","logP","TPSA","HBD","HBA"]
cat_cols = ["SEX","YEAR","QTR","drug_mapped","ATC","mechanism_tags"]

# 补缺列（以防某列缺失）
for c in num_cols:
    if c not in X_view.columns: X_view[c] = np.nan
for c in cat_cols:
    if c not in X_view.columns: X_view[c] = pd.Series(pd.NA, index=X_view.index)

# 数值：转数值+缺失填充（中位数）
X_view[num_cols] = X_view[num_cols].apply(pd.to_numeric, errors='coerce')
X_view[num_cols] = X_view[num_cols].fillna(X_view[num_cols].median())

# 类别：转为 category（LightGBM 可以直接吃）
for c in cat_cols:
    X_view[c] = X_view[c].astype('category')

# ========= 7) 划分数据集（外部 Test） =========
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X_view[num_cols+cat_cols], Y_view, test_size=0.2, random_state=42, stratify=Y_view.sum(axis=1)>0
)

# ========= 8) 数值标准化（fit 在外部训练集上，避免泄漏） =========
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_full[num_cols])
X_test_num  = scaler.transform(X_test[num_cols])

X_train_num_df = pd.DataFrame(X_train_num, index=X_train_full.index, columns=num_cols)
X_test_num_df  = pd.DataFrame(X_test_num,  index=X_test.index,  columns=num_cols)

X_train_full_proc = pd.concat([X_train_num_df, X_train_full[cat_cols]], axis=1)
X_test_proc       = pd.concat([X_test_num_df,  X_test[cat_cols]],       axis=1)

# ========= 9) 在训练集内再切分出验证集（用于阈值调优） =========
tr_idx, va_idx = train_test_split(
    X_train_full_proc.index, test_size=0.2, random_state=42
)
X_tr_proc = X_train_full_proc.loc[tr_idx]
X_va_proc = X_train_full_proc.loc[va_idx]

# ========= 10) 训练 OVR + 每个标签在验证集调阈值 =========
models = {}
best_thresholds = {}
proba_test_dict = {}
pred_test_dict  = {}

for label in top_labels:
    print(f"[Train] {label}")
    y_tr = Y_train_full.loc[tr_idx, label].astype(int).values
    y_va = Y_train_full.loc[va_idx, label].astype(int).values
    y_ts = Y_test.loc[X_test_proc.index, label].astype(int).values

    clf = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective='binary',
        metric='auc',
        n_jobs=-1,
        is_unbalance=True,  # 处理类别不平衡
        random_state=42
    )
    clf.fit(X_tr_proc, y_tr)

    # 验证集找最佳 F1 阈值
    p_va = clf.predict_proba(X_va_proc)[:, 1]
    grid = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_va, (p_va >= t).astype(int)) for t in grid]
    best_t = float(grid[int(np.argmax(f1s))])
    best_thresholds[label] = best_t

    # 测试集概率与预测
    p_ts = clf.predict_proba(X_test_proc)[:, 1]
    yhat_ts = (p_ts >= best_t).astype(int)

    proba_test_dict[label] = p_ts
    pred_test_dict[label]  = yhat_ts
    models[label] = clf

# ========= 11) 评估（micro / macro） =========
P_test = np.column_stack([proba_test_dict[l] for l in top_labels])
Yhat_test = np.column_stack([pred_test_dict[l] for l in top_labels])
Y_true = Y_test.loc[X_test_proc.index, top_labels].values

micro_f1  = f1_score(Y_true, Yhat_test, average='micro', zero_division=0)
macro_f1  = f1_score(Y_true, Yhat_test, average='macro', zero_division=0)
micro_auc = roc_auc_score(Y_true, P_test, average='micro')
macro_auc = roc_auc_score(Y_true, P_test, average='macro')

print(f"\n✅ Baseline+Tuned done.")
print(f"micro-F1={micro_f1:.3f}  macro-F1={macro_f1:.3f}  micro-AUC={micro_auc:.3f}  macro-AUC={macro_auc:.3f}")

# ========= 12) 保存结果 =========
OUTDIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(P_test, index=X_test_proc.index, columns=top_labels).to_csv(OUTDIR/"pred_proba_topk.csv")
pd.DataFrame(Yhat_test, index=X_test_proc.index, columns=top_labels, dtype=int).to_csv(OUTDIR/"pred_label_topk.csv")
pd.Series(best_thresholds, name="best_threshold").to_csv(OUTDIR/"best_thresholds.csv")
joblib.dump(models, OUTDIR/"lgbm_ovr_topk.pkl")
joblib.dump(scaler, OUTDIR/"scaler_numeric.pkl")
print("Saved to:", OUTDIR)

