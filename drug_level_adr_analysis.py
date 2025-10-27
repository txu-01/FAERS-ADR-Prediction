# drug_level_adr_analysis.py
# 功能：基于 FAERS cleaned 数据和理化性质表做药物层 ADR 分析
# 运行方式：python drug_level_adr_analysis.py
# 输入文件：
#   - cleaned/demo_clean.parquet
#   - cleaned/drug_clean.parquet
#   - cleaned/reac_clean.parquet
#   - drug_list_ext.csv
# 输出文件：
#   - drug_level_metrics.csv

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ========== 路径设置 ==========
# parquet 文件在 cleaned 文件夹中
DATA_DIR = Path(r"E:/MED/cleaned")
DRUG_PROP_FILE = Path(r"E:/MED/drug_list_ext.csv")

OUT_CSV = Path(r"E:/MED/drug_level_metrics.csv")

# ========== 工具函数 ==========
def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def standardize_drug_name_series(s):
    s = s.astype(str).str.upper().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r'^"|"$', "", regex=True)
    return s

def add_standard_name(df, candidates, new_col="DRUGNAME_STD"):
    col = first_existing(df, candidates)
    if col is None:
        df[new_col] = np.nan
        used = None
    else:
        df[new_col] = standardize_drug_name_series(df[col])
        used = col
    return df, used

def build_diabetes_regex():
    names = [
        "METFORMIN",
        "INSULIN","GLARGINE","DEGLUDEC","DETEMIR","LISPRO","ASPART","GLULISINE","REGULAR INSULIN","NPH",
        "SEMAGLUTIDE","LIRAGLUTIDE","DULAGLUTIDE","EXENATIDE","ALBIGLUTIDE","TIRZEPATIDE",
        "EMPAGLIFLOZIN","DAPAGLIFLOZIN","CANAGLIFLOZIN","ERTUGLIFLOZIN",
        "SITAGLIPTIN","SAXAGLIPTIN","LINAGLIPTIN","ALOGLIPTIN","VILDAGLIPTIN",
        "PIOGLITAZONE","ROSIGLITAZONE",
        "ACARBOSE","MIGLITOL","VOGLIBOSE",
        "NATEGLINIDE","REPAGLINIDE","MITIGLINIDE",
        "GLIPIZIDE","GLIMEPIRIDE","GLYBURIDE","GLIBENCLAMIDE","GLICLAZIDE",
        "PRAMLINTIDE","COLESEVELAM","BROMOCRIPTINE"
    ]
    brands = [
        "OZEMPIC","RYBELSUS","WEGOVY","TRULICITY","VICTOZA","BYDUREON","BYETTA","MOUNJARO",
        "JARDIANCE","FARXIGA","INVOKANA","STEGLATRO",
        "JANUVIA","ONGLYZA","TRADJENTA","NESINA",
        "ACTOS","AVANDIA",
        "AMARYL","GLUCOPHAGE","GLUCOTROL","DIABETA","MICRONASE","DAONIL","DIAMICRON"
    ]
    pats = names + brands
    return re.compile("|".join([re.escape(x) for x in pats]))

# ========== 读取数据 ==========
drug_prop = pd.read_csv(DRUG_PROP_FILE)
drug = pd.read_parquet(Path(r"E:\MED\ASCII\cleaned\drug_clean.parquet"))
reac = pd.read_parquet(Path(r"E:\MED\ASCII\cleaned\reac_clean.parquet"))

# ========== 标准化药名 ==========
drug, drug_name_col_drug = add_standard_name(
    drug,
    candidates=["DRUGNAME_STD","DRUGNAME","drugname","SUBSTANCENAME","PROD_AI","prod_ai","MEDICINALPRODUCT"]
)
drug_prop, drug_name_col_prop = add_standard_name(
    drug_prop,
    candidates=["drug_name","DRUGNAME_STD","DRUGNAME","name","NAME","PreferredName","preferred_name"]
)

# ========== 过滤糖尿病相关药物 ==========
diab_rx = build_diabetes_regex()
drug["IS_DIAB_DRUG"] = drug["DRUGNAME_STD"].fillna("").str.contains(diab_rx, regex=True)

caseid_drug = first_existing(drug, ["CASEID","primaryid","PRIMARYID","caseid"])
caseid_reac = first_existing(reac, ["CASEID","primaryid","PRIMARYID","caseid"])
reac_term_col = first_existing(reac, ["PT","pt","reactionmeddrapt","REAC_PT"])

dm = drug.loc[drug["IS_DIAB_DRUG"] == True].copy()

# 若存在主嫌疑药，优先用PS
role_col = first_existing(dm, ["ROLE_COD","role_cod","role","ROLE"])
if role_col is not None:
    dm_ps = dm[dm[role_col].astype(str).str.upper().eq("PS")].copy()
    if dm_ps.empty:
        dm_ps = dm.copy()
else:
    dm_ps = dm.copy()

# ========== 计算每个CASE的ADR数量 ==========
reac_counts = (
    reac.groupby(caseid_reac)
        .size()
        .reset_index(name="ADR_count")
)
reac_counts = reac_counts.rename(columns={caseid_reac: "CASEID"})

dm_ps = dm_ps.rename(columns={caseid_drug: "CASEID"})
dm_ps = dm_ps.merge(reac_counts, on="CASEID", how="left")

# ========== 药物层指标 ==========
metrics = (
    dm_ps.groupby("DRUGNAME_STD")
         .agg(
             n_reports=("CASEID", pd.Series.nunique),
             total_adr=("ADR_count", "sum")
         )
         .reset_index()
)
metrics["total_adr"] = metrics["total_adr"].fillna(0).astype(int)
metrics["mean_ADR_per_case"] = np.where(
    metrics["n_reports"] > 0,
    metrics["total_adr"] / metrics["n_reports"],
    np.nan
)

# ADR多样性（不同PT数量）
case_to_pts = (
    reac[[caseid_reac, reac_term_col]]
    .dropna()
    .rename(columns={caseid_reac: "CASEID", reac_term_col: "PT"})
    .drop_duplicates()
)
dm_cases = dm_ps[["DRUGNAME_STD","CASEID"]].dropna().drop_duplicates()
dm_pt = dm_cases.merge(case_to_pts, on="CASEID", how="left")

diversity = (
    dm_pt.groupby("DRUGNAME_STD")["PT"]
         .nunique()
         .reset_index(name="unique_PT_count")
)

metrics = metrics.merge(diversity, on="DRUGNAME_STD", how="left")

# ========== 合并理化性质并输出 ==========
drug_prop_cols_no_dup = [c for c in drug_prop.columns if c != "DRUGNAME_STD"]
merged = metrics.merge(drug_prop, on="DRUGNAME_STD", how="left")

merged = merged.sort_values(["n_reports","total_adr"], ascending=[False, False]).reset_index(drop=True)
merged.to_csv(OUT_CSV, index=False)

# ========== 结果预览 ==========
keep = ["DRUGNAME_STD","n_reports","total_adr","mean_ADR_per_case","unique_PT_count"]
for cand in ["LOGP","LogP","cLogP","MW","MOL_WEIGHT","TPSA","HBA","HBD"]:
    if cand in merged.columns and cand not in keep:
        keep.append(cand)

print("\n=== 结果预览（前20行）===")
print(merged[keep].head(20).to_string(index=False))
print(f"\n✅ 已导出结果文件：{OUT_CSV.resolve()}")
