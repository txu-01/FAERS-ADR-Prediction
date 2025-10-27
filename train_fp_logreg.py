# -*- coding: utf-8 -*-
"""
RandomForest + per-label threshold tuning
- 读取: outputs/X_patient_level.parquet, outputs/Y_multilabel_PT.parquet
- 自动对齐 PRIMARYID
- 如有 SMILES 列, 计算 1024-bit Morgan 指纹并拼接
- 选前 TOP_K 个标签训练
- 在训练集内部再切一个验证集, 为每个标签找最佳阈值(最大化F1)
- 最终在测试集上用各自阈值评估
"""

import os, hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# ----------------- 可调参数 -----------------
DATA_DIR = r"outputs"
X_FILE = "X_patient_level.parquet"
Y_FILE = "Y_multilabel_PT.parquet"
TOP_K = 30
TEST_SIZE = 0.2
VAL_SIZE = 0.2               # 从训练集中再切一部分做阈值寻优
RANDOM_STATE = 42
N_JOBS = -1

USE_FP = True
FP_NBITS = 1024
FP_RADIUS = 2
FP_PREFIX = "FP"
# ------------------------------------------------

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

def topk_labels(Y: pd.DataFrame, k: int):
    freq = Y.sum(0).sort_values(ascending=False)
    cols = freq.index[:k]
    return Y[cols], list(cols), list(freq.iloc[:k].astype(int).values)

def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    对每个标签在验证集上寻找使 F1 最大的阈值。
    y_true, y_prob 形状: (n_samples, n_labels)
    返回 shape=(n_labels,) 的阈值数组, 默认回退为 0.5。
    """
    L = y_true.shape[1]
    thres = np.full(L, 0.5, dtype=float)
    for j in range(L):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        # 必须既有正也有负才有意义
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        ps, rs, ts = precision_recall_curve(yt, yp)
        # 避免除零
        denom = ps + rs
        f1s = np.where(denom > 0, 2 * ps * rs / denom, 0.0)
        best_idx = np.argmax(f1s)
        # precision_recall_curve 返回的阈值 ts 比 ps/rs 少1个点
        if best_idx == 0:
            best_t = 0.5
        else:
            best_t = float(ts[best_idx - 1])
        thres[j] = np.clip(best_t, 0.01, 0.99)
    return thres

# ================= 主流程 =================
if __name__ == "__main__":
    x_path = os.path.join(DATA_DIR, X_FILE)
    y_path = os.path.join(DATA_DIR, Y_FILE)

    X = pd.read_parquet(x_path)
    Y = pd.read_parquet(y_path)

    if "PRIMARYID" in X.columns: X = X.set_index("PRIMARYID")
    if "PRIMARYID" in Y.columns: Y = Y.set_index("PRIMARYID")
    if not X.index.is_unique: X = X[~X.index.duplicated(keep="first")]
    if not Y.index.is_unique: Y = Y[~Y.index.duplicated(keep="first")]

    inter = X.index.intersection(Y.index)
    X, Y = X.loc[inter].copy(), Y.loc[inter].copy()
    print(f"原始数据: X={X.shape}, Y={Y.shape}")

    Yk, label_names, freqs = topk_labels(Y, TOP_K)
    print(f"选取前{TOP_K}个ADR标签: {Yk.shape}")
    print("Top labels:", ", ".join([f"{n}(pos={f})" for n, f in zip(label_names, freqs)]))

    smiles_col = has_smiles(X) if USE_FP else None
    if smiles_col:
        print("检测到SMILES列，计算 Morgan 指纹...")
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
        print("未检测到SMILES列，跳过指纹。")
        FP = None
        X_phys = X

    X_phys = safe_numeric(X_phys)
    num_cols = X_phys.select_dtypes(include=[np.number]).columns.tolist()
    X_phys = X_phys[num_cols].fillna(0)

    X_all = pd.concat([X_phys, FP], axis=1) if FP is not None else X_phys
    X_all.columns = X_all.columns.astype(str)
    print(f"最终特征形状: {X_all.shape}（理化{len(num_cols)}{' + 指纹'+str(FP.shape[1]) if FP is not None else ''}）")

    # train/test
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, Yk, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 再切出验证集做阈值寻优
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(X_tr, y_tr, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    base = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    clf = MultiOutputClassifier(base, n_jobs=N_JOBS)
    clf.fit(X_tr_sub, y_tr_sub)

    # 概率用于阈值寻优
    def proba_matrix(model, X):
        L = len(model.estimators_)
        P = np.zeros((X.shape[0], L), dtype=float)
        for j, est in enumerate(model.estimators_):
            if hasattr(est, "predict_proba"):
                prob = est.predict_proba(X)
                if isinstance(prob, list):
                    prob = prob[0]
                P[:, j] = prob[:, 1]
            else:
                P[:, j] = est.predict(X)
        return P

    P_val = proba_matrix(clf, X_val)
    thres = tune_thresholds(y_val.values, P_val)
    # 保存阈值
    th_path = os.path.join(DATA_DIR, "rf_label_thresholds.npy")
    np.save(th_path, thres)

    # 在全部训练集上再拟合一次
    clf.fit(X_tr, y_tr)
    P_te = proba_matrix(clf, X_te)
    y_hat = (P_te >= thres).astype(int)

    micro_f1 = f1_score(y_te.values, y_hat, average="micro", zero_division=0)
    macro_f1 = f1_score(y_te.values, y_hat, average="macro", zero_division=0)

    # AUC 仅对有正负样本的标签计算
    aucs = []
    for j in range(P_te.shape[1]):
        yt = y_te.iloc[:, j].values
        if 0 < yt.sum() < len(yt):
            try:
                aucs.append(roc_auc_score(yt, P_te[:, j]))
            except Exception:
                pass
    micro_auc = float(np.mean(aucs)) if len(aucs) else np.nan

    print("\n✅ RF+阈值调优 完成")
    print(f"micro-F1={micro_f1:.3f}   macro-F1={macro_f1:.3f}   micro-AUC={micro_auc:.3f}")
    print(f"阈值已保存: {th_path}")

    # 保存测试集预测
    out_pred = os.path.join(DATA_DIR, "rf_pred_with_threshold.parquet")
    pd.DataFrame(y_hat, index=y_te.index, columns=y_te.columns).to_parquet(out_pred)
    print(f"Predictions saved to: {out_pred}")
