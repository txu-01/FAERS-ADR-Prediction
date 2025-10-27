# -*- coding: utf-8 -*-
"""
Binary ADR task:
  Positive  = 样本在 TOP_K 个 ADR 中出现 ">=2 个" 阳性（multi-ADR）
  Negative  = 样本在 TOP_K 个 ADR 中出现 "恰好 1 个" 阳性（single-ADR）
  丢弃      = 在 TOP_K 中无任何阳性的样本（0 个），以免引入噪声

数据：
  - X: outputs/X_patient_level.parquet  (含理化性质；若含 SMILES 自动计算指纹)
  - Y: outputs/Y_multilabel_PT.parquet  (多标签稀疏0/1矩阵，列为 MedDRA PT)

输出：
  - 指标：ROC-AUC, F1, Precision, Recall, PR-AUC
  - 保存预测/阈值/特征重要性
"""

import os, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========= 参数 =========
DATA_DIR = r"outputs"
X_FILE   = "X_patient_level.parquet"
Y_FILE   = "Y_multilabel_PT.parquet"

TOP_K = 50            # 统计“是否>=2个ADR”时考虑的前K个标签（按出现频次排序）
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = -1

# 指纹开关（如 X 有 SMILES 列则启用）
USE_FP = True
FP_NBITS = 1024
FP_RADIUS = 2
FP_PREFIX = "FP"

# RF 参数（稳妥 & 快速）
RF_KW = dict(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
)

# ========= 工具 =========
def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass
    return df

def has_smiles(X: pd.DataFrame):
    for k in ["SMILES", "smiles", "Smiles"]:
        if k in X.columns:
            return k
    return None

def cache_key(s: pd.Series) -> str:
    h = hashlib.md5()
    h.update(("|".join("" if pd.isna(v) else str(v) for v in s.values)).encode("utf-8"))
    return h.hexdigest()

def compute_morgan(smiles: pd.Series, nBits=1024, radius=2) -> pd.DataFrame:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    arr = np.zeros((len(smiles), nBits), dtype=np.uint8)
    for i, s in enumerate(smiles.fillna("")):
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            arr[i, :] = np.array(gen.GetFingerprintAsNumPy(m), dtype=np.uint8)
    cols = [f"{FP_PREFIX}{i}" for i in range(nBits)]
    return pd.DataFrame(arr, index=smiles.index, columns=cols)

def topk_matrix(Y: pd.DataFrame, k: int) -> pd.DataFrame:
    freq = Y.sum(0).sort_values(ascending=False)
    cols = freq.index[:k]
    return Y[cols]

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """在验证集上扫阈值，最大化F1"""
    ps, rs, ts = precision_recall_curve(y_true, y_prob)
    denom = ps + rs
    f1s = np.where(denom > 0, 2 * ps * rs / denom, 0.0)
    best_idx = int(np.argmax(f1s))
    if best_idx == 0:
        return 0.5
    return float(np.clip(ts[best_idx - 1], 0.3, 0.99))

# ========= 主流程 =========
if __name__ == "__main__":
    x_path = os.path.join(DATA_DIR, X_FILE)
    y_path = os.path.join(DATA_DIR, Y_FILE)

    X = pd.read_parquet(x_path)
    Y = pd.read_parquet(y_path)

    # 用 PRIMARYID 对齐
    if "PRIMARYID" in X.columns: X = X.set_index("PRIMARYID")
    if "PRIMARYID" in Y.columns: Y = Y.set_index("PRIMARYID")
    if not X.index.is_unique: X = X[~X.index.duplicated(keep="first")]
    if not Y.index.is_unique: Y = Y[~Y.index.duplicated(keep="first")]
    inter = X.index.intersection(Y.index)
    X, Y = X.loc[inter].copy(), Y.loc[inter].copy()
    print(f"原始数据对齐后: X={X.shape}, Y={Y.shape}")

    # 取前K标签并构造二分类标签
    Yk = topk_matrix(Y, TOP_K).astype(int)
    row_sum = Yk.sum(axis=1).astype(int)
    # 正例：>=2；负例：==1；丢弃：==0
    mask = row_sum >= 1
    y_bin = (row_sum >= 2).astype(int)[mask]
    X = X.loc[mask]
    print(f"用于二分类的样本: {len(y_bin)} (pos={int(y_bin.sum())}, neg={int((1-y_bin).sum())})")

    # 特征工程：理化 + 可选指纹
    smiles_col = has_smiles(X) if USE_FP else None
    if smiles_col:
        print("检测到SMILES，计算 Morgan 指纹...")
        cache_dir = os.path.join(DATA_DIR, "cache"); os.makedirs(cache_dir, exist_ok=True)
        key = cache_key(X[smiles_col]); cache_file = os.path.join(cache_dir, f"fp_{FP_NBITS}_{FP_RADIUS}_{key}.parquet")
        if os.path.exists(cache_file):
            FP = pd.read_parquet(cache_file).reindex(X.index)
            print(f"指纹已从缓存加载: {FP.shape}")
        else:
            FP = compute_morgan(X[smiles_col], FP_NBITS, FP_RADIUS)
            FP.to_parquet(cache_file)
            print(f"指纹计算完成并缓存: {FP.shape}")
        X_phys = X.drop(columns=[smiles_col], errors="ignore")
    else:
        print("未检测到SMILES，跳过指纹")
        FP = None
        X_phys = X

    X_phys = safe_numeric(X_phys)
    num_cols = X_phys.select_dtypes(include=[np.number]).columns.tolist()
    X_phys = X_phys[num_cols].fillna(0)
    X_all = pd.concat([X_phys, FP], axis=1) if FP is not None else X_phys
    X_all.columns = X_all.columns.astype(str)
    print(f"最终特征: {X_all.shape}（理化{len(num_cols)}{' + 指纹'+str(FP.shape[1]) if FP is not None else ''}）")

    # 划分数据（分层）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_bin, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bin
    )

    # 再切验证集用于阈值寻优
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
    )

    # 训练 RF
    clf = RandomForestClassifier(**RF_KW)
    clf.fit(X_tr_sub, y_tr_sub)

    # 在验证集上找最佳阈值
    if hasattr(clf, "predict_proba"):
        p_val = clf.predict_proba(X_val)[:, 1]
    else:
        p_val = clf.decision_function(X_val)
        p_val = (p_val - p_val.min()) / (p_val.max() - p_val.min() + 1e-9)
    best_th = find_best_threshold(y_val.values, p_val)

    # 用全部训练数据重训
    clf.fit(X_tr, y_tr)

    # 测试评估
    p_te = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_te)
    y_hat = (p_te >= best_th).astype(int)

    auc = roc_auc_score(y_te, p_te)
    pr_auc = average_precision_score(y_te, p_te)
    f1 = f1_score(y_te, y_hat)
    ps, rs, _ = precision_recall_curve(y_te, p_te)
    prec = (y_hat[y_te==1].sum() / max(y_hat.sum(), 1)) if y_hat.sum()>0 else 0.0
    rec = (y_hat & y_te.values).sum() / max(y_te.sum(), 1)

    print("\n✅ Binary task finished (multi-ADR ≥2 vs single-ADR =1)")
    print(f"AUC={auc:.3f} | PR-AUC={pr_auc:.3f} | F1={f1:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")
    print(f"Best threshold on val = {best_th:.3f}")

    # 混淆矩阵与分类报告
    cm = confusion_matrix(y_te, y_hat)
    print("\nConfusion matrix [ [TN FP] [FN TP] ]:\n", cm)
    print("\nClassification report:\n", classification_report(y_te, y_hat, digits=3))

    # 保存产物
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.Series(p_te, index=X_te.index, name="proba").to_csv(os.path.join(DATA_DIR, "binary_proba_test.csv"))
    pd.Series(y_hat, index=X_te.index, name="pred").to_csv(os.path.join(DATA_DIR, "binary_pred_test.csv"))
    pd.Series([best_th], index=["best_threshold"]).to_csv(os.path.join(DATA_DIR, "binary_best_threshold.csv"))

    # 简单的“TOP”特征重要性（理化+前100个指纹）
    importances = pd.Series(clf.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    top_imp = importances.head(30)
    top_imp.to_csv(os.path.join(DATA_DIR, "binary_top_feature_importances.csv"))
    print("\nTop-30 feature importances saved to outputs/binary_top_feature_importances.csv")
