# E:/MED/build_patient_level_v2.py
# 在 v1 基础上增强特征：加入总用药计数、PS主嫌疑药计数、途径(口服/皮下)特征、机制计数、理化性质std/极值差、比率特征等
# 输出：E:/MED/ASCII/cleaned/patient_level_dataset_v2.csv

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"E:/MED/ASCII/cleaned")
DRUG_FILE = BASE/"drug_clean.parquet"
REAC_FILE = BASE/"reac_clean.parquet"
DEMO_FILE = BASE/"demo_clean.parquet"        # 可选
DRUGLIST_FILE = Path(r"E:/MED/drug_list_ext.csv")
OUT_FILE = BASE/"patient_level_dataset_v2.csv"

# ------------ helpers ------------
def std_str(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r'^"|"$', "", regex=True)
    return s

def first_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

BRAND_TO_GENERIC = {
    "OZEMPIC":"SEMAGLUTIDE","RYBELSUS":"SEMAGLUTIDE","WEGOVY":"SEMAGLUTIDE",
    "TRULICITY":"DULAGLUTIDE","BYETTA":"EXENATIDE","BYDUREON":"EXENATIDE",
    "MOUNJARO":"TIRZEPATIDE",
    "JARDIANCE":"EMPAGLIFLOZIN","FARXIGA":"DAPAGLIFLOZIN","INVOKANA":"CANAGLIFLOZIN","STEGLATRO":"ERTUGLIFLOZIN",
    "JANUVIA":"SITAGLIPTIN","ONGLYZA":"SAXAGLIPTIN","TRADJENTA":"LINAGLIPTIN","NESINA":"ALOGLIPTIN",
    "ACTOS":"PIOGLITAZONE","AVANDIA":"ROSIGLITAZONE",
    "AMARYL":"GLIMEPIRIDE","GLUCOPHAGE":"METFORMIN","GLUCOTROL":"GLIPIZIDE",
    "DIABETA":"GLYBURIDE","MICRONASE":"GLYBURIDE","DAONIL":"GLIBENCLAMIDE","DIAMICRON":"GLICLAZIDE"
}

# ------------ load ------------
drug = pd.read_parquet(DRUG_FILE)
reac = pd.read_parquet(REAC_FILE)
druglist = pd.read_csv(DRUGLIST_FILE)

# key columns
drug_name_col = first_col(drug, ["DRUGNAME_STD","DRUGNAME","drugname","SUBSTANCENAME","PROD_AI","MEDICINALPRODUCT"])
case_drug_col = first_col(drug, ["CASEID","caseid","PRIMARYID","primaryid","PrimaryID"])
case_reac_col = first_col(reac, ["CASEID","caseid","PRIMARYID","primaryid","PrimaryID"])
route_col = first_col(drug, ["ROUTE","route","ROUTE_COD","route_cod"])
role_col  = first_col(drug, ["ROLE_COD","role_cod","ROLE","role"])
assert drug_name_col and case_drug_col and case_reac_col, "必要列缺失（DRUGNAME/CASEID）"

# std names
drug["DRUG_STD"] = std_str(drug[drug_name_col])
drug["GENERIC_STD"] = drug["DRUG_STD"].map(lambda x: BRAND_TO_GENERIC.get(x, x))

druglist["NAME_STD"] = std_str(druglist["drug_name"])
allowed = set(druglist["NAME_STD"].unique())

# label: ADR_count per case
adr = reac.groupby(case_reac_col).size().reset_index(name="ADR_count").rename(columns={case_reac_col:"CASEID"})

# ---------------- 1) 全量用药计数（不只糖尿病） ----------------
drug_all = drug.rename(columns={case_drug_col:"CASEID"}).copy()
# 总记录/总不同药物
per_case_all = drug_all.groupby("CASEID").agg(
    n_total_records=("DRUG_STD","count"),
    n_total_unique=("DRUG_STD", pd.Series.nunique)
)

# 主嫌疑药（PS）计数
if role_col and role_col in drug_all.columns:
    drug_all["IS_PS"] = drug_all[role_col].astype(str).str.upper().eq("PS").astype(int)
    ps_counts = drug_all.groupby("CASEID")["IS_PS"].sum().to_frame("n_ps_records")
else:
    ps_counts = pd.DataFrame(index=per_case_all.index, data={"n_ps_records":0})
per_case_all = per_case_all.join(ps_counts, how="left")

# 途径（粗分：ORAL vs SUBCUTANEOUS vs OTHER）
if route_col and route_col in drug_all.columns:
    r = drug_all[["CASEID", route_col]].copy()
    r[route_col] = r[route_col].astype(str).str.upper()
    r["is_oral"] = r[route_col].str.contains("ORAL|PO|BY MOUTH", regex=True).astype(int)
    r["is_sc"]   = r[route_col].str.contains("SUBCUTANEOUS|SC|SUBCUT", regex=True).astype(int)
    per_route = r.groupby("CASEID")[["is_oral","is_sc"]].mean()  # 比例
else:
    per_route = pd.DataFrame(index=per_case_all.index, data={"is_oral":np.nan,"is_sc":np.nan})
per_case_all = per_case_all.join(per_route, how="left")

# ---------------- 2) 仅糖尿病药物子集 ----------------
drug_dm = drug_all.copy()
drug_dm["IS_DIAB"] = drug_dm["GENERIC_STD"].isin(allowed)
drug_dm = drug_dm[drug_dm["IS_DIAB"]].copy()

# merge properties + mechanism
props = ["mol_weight","logP","TPSA","HBD","HBA","ATC","mechanism_tags"]
dm_props = druglist[["NAME_STD"]+props].rename(columns={"NAME_STD":"GENERIC_STD"})
drug_dm = drug_dm.merge(dm_props, on="GENERIC_STD", how="left")

# 机制计数 per CASE
mech_oh = pd.DataFrame(index=drug_dm["CASEID"].unique())
if "mechanism_tags" in drug_dm.columns:
    m = (drug_dm.assign(mech=drug_dm["mechanism_tags"].fillna(""))
                .assign(mech=lambda d: d["mech"].str.split(r"[;,|]+"))
                .explode("mech"))
    m["mech"] = m["mech"].str.strip()
    m = m.loc[m["mech"].ne(""), ["CASEID","mech"]]
    if not m.empty:
        mech_oh = m.assign(val=1).pivot_table(index="CASEID", columns="mech", values="val", aggfunc="sum", fill_value=0)

# 糖尿病药物计数特征
per_case_dm = (drug_dm.groupby("CASEID")
    .agg(n_diab_records=("GENERIC_STD","count"),
         n_diab_unique=("GENERIC_STD", pd.Series.nunique))
)
per_case_dm["is_multi_diab"] = (per_case_dm["n_diab_unique"]>=2).astype(int)

# 物化性质：均值/最大/和/标准差/极差
agg_funcs = {
    "mol_weight":["mean","max","sum","std"],
    "logP":["mean","max","min","std"],
    "TPSA":["mean","max","sum","std"],
    "HBD":["mean","max","sum","std"],
    "HBA":["mean","max","sum","std"]
}
per_case_props = drug_dm.groupby("CASEID").agg(agg_funcs)
per_case_props.columns = ["_".join(c) for c in per_case_props.columns.to_flat_index()]
# 极差
for base in ["logP","TPSA","HBD","HBA","mol_weight"]:
    if f"{base}_max" in per_case_props.columns and f"{base}_mean" in per_case_props.columns:
        per_case_props[f"{base}_range"] = per_case_props[f"{base}_max"] - per_case_props.get(f"{base}_min", per_case_props[f"{base}_mean"])

# 合并糖尿病子集特征 + 机制
X_dm = per_case_dm.join(per_case_props, how="left").join(mech_oh, how="left")

# ---------------- 3) 汇总报告层 + 标签 ----------------
X = per_case_all.join(X_dm, how="left").reset_index()
X = X.merge(adr, on="CASEID", how="left")
X["ADR_count"] = X["ADR_count"].fillna(0).astype(int)
X["y_multiADR"] = (X["ADR_count"]>=2).astype(int)

# 比率/交互：糖尿病药占比、PS占比、总唯一药物对数
X["ratio_diab_total_unique"] = X["n_diab_unique"] / X["n_total_unique"].replace(0, np.nan)
X["ps_ratio"] = X["n_ps_records"] / X["n_total_records"].replace(0, np.nan)
X["ln_total_unique"] = np.log1p(X["n_total_unique"])
X["ln_diab_unique"]  = np.log1p(X["n_diab_unique"])

# 加入 demo（age/sex，如存在）
try:
    demo = pd.read_parquet(DEMO_FILE)
    case_demo_col = first_col(demo, ["CASEID","caseid","PRIMARYID","primaryid","PrimaryID"])
    if case_demo_col:
        demo = demo.rename(columns={case_demo_col:"CASEID"})
        age_col = first_col(demo, ["AGE","age","AGE_YRS","age_yrs"])
        sex_col = first_col(demo, ["SEX","sex","GENDER","gender"])
        keep = ["CASEID"]
        if age_col: keep.append(age_col)
        if sex_col: keep.append(sex_col)
        dsmall = demo[keep].copy()
        if age_col: dsmall = dsmall.rename(columns={age_col:"age"})
        if sex_col: dsmall = dsmall.rename(columns={sex_col:"sex"})
        X = X.merge(dsmall, on="CASEID", how="left")
        # 年龄分箱
        if "age" in X.columns:
            bins = [-1,18,40,65,120]; labels=["<18","18-40","40-65","65+"]
            X["age_bin"] = pd.cut(X["age"], bins=bins, labels=labels)
except Exception as e:
    print("[WARN] merge demo failed:", e)

# 缺失填充（计数类缺失置0）
for c in ["n_total_records","n_total_unique","n_ps_records","n_diab_records","n_diab_unique","is_multi_diab"]:
    if c in X.columns:
        X[c] = X[c].fillna(0).astype(int)

X.to_csv(OUT_FILE, index=False)
print("✅ Saved:", OUT_FILE)
print("  n CASEID:", X["CASEID"].nunique(), "| pos ratio:", X["y_multiADR"].mean().round(3))
print("  columns:", len(X.columns))
