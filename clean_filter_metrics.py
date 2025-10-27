# E:/MED/clean_filter_metrics.py
import pandas as pd
import numpy as np
from pathlib import Path

METRICS_FILE = Path(r"E:/MED/drug_level_metrics.csv")
LIST_FILE    = Path(r"E:/MED/drug_list_ext.csv")
OUT_DIR      = Path(r"E:/MED/cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLEAN    = OUT_DIR / "drug_level_metrics_clean.csv"
OUT_UNMATCH  = OUT_DIR / "drug_level_unmatched.csv"

def std_name(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r'^"|"$', "", regex=True)
    return s

# 商品名→通用名映射（可按需扩展）
BRAND_TO_GENERIC = {
    "OZEMPIC": "SEMAGLUTIDE",
    "RYBELSUS": "SEMAGLUTIDE",
    "WEGOVY": "SEMAGLUTIDE",
    "TRULICITY": "DULAGLUTIDE",
    "BYETTA": "EXENATIDE",
    "BYDUREON": "EXENATIDE",
    "MOUNJARO": "TIRZEPATIDE",
    "JARDIANCE": "EMPAGLIFLOZIN",
    "FARXIGA": "DAPAGLIFLOZIN",
    "INVOKANA": "CANAGLIFLOZIN",
    "STEGLATRO": "ERTUGLIFLOZIN",  # 若未来补充该药于drug_list_ext
    "JANUVIA": "SITAGLIPTIN",
    "ONGLYZA": "SAXAGLIPTIN",
    "TRADJENTA": "LINAGLIPTIN",
    "NESINA": "ALOGLIPTIN",
    "ACTOS": "PIOGLITAZONE",
    "AVANDIA": "ROSIGLITAZONE",
    "AMARYL": "GLIMEPIRIDE",
    "GLUCOPHAGE": "METFORMIN",
    "GLUCOTROL": "GLIPIZIDE",
    "DIABETA": "GLYBURIDE",
    "MICRONASE": "GLYBURIDE",
    "DAONIL": "GLIBENCLAMIDE",
    "DIAMICRON": "GLICLAZIDE"
}

# 读取
metrics = pd.read_csv(METRICS_FILE)
druglist = pd.read_csv(LIST_FILE)

# 标准化两侧药名
# metrics 侧列名为 DRUGNAME_STD（上游脚本已生成），若没有就尝试其它候选
name_col_metrics = "DRUGNAME_STD" if "DRUGNAME_STD" in metrics.columns else (
    "DRUGNAME" if "DRUGNAME" in metrics.columns else None
)
if name_col_metrics is None:
    raise ValueError("未在 drug_level_metrics.csv 中找到药名列（DRUGNAME_STD / DRUGNAME）")

metrics["NAME_STD"] = std_name(metrics[name_col_metrics])

# druglist 侧使用 drug_name
if "drug_name" not in druglist.columns:
    raise ValueError("drug_list_ext.csv 中未找到列 'drug_name'")
druglist["NAME_STD"] = std_name(druglist["drug_name"])

# 先把 metrics 中的商品名映射为通用名
metrics["NAME_STD_MAP"] = metrics["NAME_STD"].map(lambda x: BRAND_TO_GENERIC.get(x, x))

# 构造允许匹配集合（用通用名的大写）
allowed = set(druglist["NAME_STD"].unique())

# 过滤：仅保留在 druglist 中的药物
mask_keep = metrics["NAME_STD_MAP"].isin(allowed)
clean = metrics.loc[mask_keep].copy()

# 把通用名写回一个清晰列以便后续连接或展示
clean["GENERIC_STD"] = metrics.loc[mask_keep, "NAME_STD_MAP"].values

# 未匹配的输出，方便你检查是否需要扩展映射或补理化表
unmatched = metrics.loc[~mask_keep, [name_col_metrics, "NAME_STD"]].drop_duplicates()

# 保存
clean.to_csv(OUT_CLEAN, index=False)
unmatched.to_csv(OUT_UNMATCH, index=False)

# 简要汇报
print("✅ 清理完成")
print(f"  原始药物条目：{metrics['NAME_STD'].nunique()}")
print(f"  匹配到 drug_list_ext 的药物条目：{clean['GENERIC_STD'].nunique()}")
print(f"  导出：{OUT_CLEAN}")
print(f"  未匹配清单：{OUT_UNMATCH}")
