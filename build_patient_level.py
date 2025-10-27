# E:/MED/build_patient_level.py
# 生成患者/报告层（CASEID）的建模数据：y=1{ADR_count>=2}
# 输入：
#   - E:/MED/cleaned/drug_clean.parquet
#   - E:/MED/cleaned/reac_clean.parquet
#   - E:/MED/cleaned/demo_clean.parquet  (可选，有则并入年龄/性别)
#   - E:/MED/drug_list_ext.csv
# 输出：
#   - E:/MED/cleaned/patient_level_dataset.csv

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- Paths ----------------
DATA_DIR = Path(r"E:/MED/ASCII/cleaned")

DRUG_FILE = DATA_DIR / "drug_clean.parquet"
REAC_FILE = DATA_DIR / "reac_clean.parquet"
DEMO_FILE = DATA_DIR / "demo_clean.parquet"   # optional
DRUGLIST_FILE = Path(r"E:/MED/drug_list_ext.csv")
OUT_FILE = DATA_DIR / "patient_level_dataset.csv"

# ---------------- Helpers ----------------
def std_str(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r'^"|"$', "", regex=True)
    return s

def first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# 常见商品名→通用名映射（可扩展）
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
    "STEGLATRO": "ERTUGLIFLOZIN",
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
    "DIAMICRON": "GLICLAZIDE",
}

# ---------------- Load ----------------
drug = pd.read_parquet(DRUG_FILE)
reac = pd.read_parquet(REAC_FILE)
druglist = pd.read_csv(DRUGLIST_FILE)

# 标准化药名
drug_name_col = first_col(drug, ["DRUGNAME_STD","DRUGNAME","drugname","SUBSTANCENAME","PROD_AI","MEDICINALPRODUCT"])
if drug_name_col is None:
    raise ValueError("在 drug_clean.parquet 中未找到药物名列")

drug["DRUG_STD"] = std_str(drug[drug_name_col])
drug["GENERIC_STD"] = drug["DRUG_STD"].map(lambda x: BRAND_TO_GENERIC.get(x, x))

# CASEID 列
case_col_drug = first_col(drug, ["CASEID","caseid","PRIMARYID","primaryid","PrimaryID"])
if case_col_drug is None:
    raise ValueError("在 drug_clean.parquet 中未找到 CASEID/PRIMARYID 列")

# drug_list：标准化名称
if "drug_name" not in druglist.columns:
    raise ValueError("drug_list_ext.csv 缺少 drug_name 列")
druglist["NAME_STD"] = std_str(druglist["drug_name"])

# 只保留在 drug_list 出现的糖尿病药物
allowed = set(druglist["NAME_STD"].unique())
drug["IS_DIAB"] = drug["GENERIC_STD"].isin(allowed)
drug_dm = drug.loc[drug["IS_DIAB"]].copy()

# ---------------- ADR_count（报告层） ----------------
case_col_reac = first_col(reac, ["CASEID","caseid","PRIMARYID","primaryid","PrimaryID"])
reac_term_col = first_col(reac, ["PT","pt","reactionmeddrapt","REAC_PT"])
if case_col_reac is None:
    raise ValueError("在 reac_clean.parquet 中未找到 CASEID/PRIMARYID 列")

adr_count = reac.groupby(case_col_reac).size().reset_index(name="ADR_count")
adr_count = adr_count.rename(columns={case_col_reac: "CASEID"})

# ---------------- 药物→报告 物化特征聚合 ----------------
# 将 druglist 中的理化性质并到 drug_dm
props = ["mol_weight","logP","TPSA","HBD","HBA","ATC","mechanism_tags"]
drug_props = druglist[["NAME_STD"] + props].copy()
drug_props = drug_props.rename(columns={"NAME_STD":"GENERIC_STD"})

drug_dm = drug_dm.rename(columns={case_col_drug: "CASEID"})
drug_dm = drug_dm.merge(drug_props, on="GENERIC_STD", how="left")

# 机制标签 one-hot（计数），mechanism_tags 可能是形如 "SGLT2_INHIBITION"
if "mechanism_tags" in drug_dm.columns:
    mech = (drug_dm.assign(mech=drug_dm["mechanism_tags"].fillna(""))
                     .assign(mech=lambda d: d["mech"].str.split(r"[;,|]+"))
                     .explode("mech"))
    mech["mech"] = mech["mech"].str.strip()
    mech = mech.loc[mech["mech"].ne("") , ["CASEID","mech"]]
    mech_onehot = (mech.assign(val=1).pivot_table(index="CASEID", columns="mech", values="val", aggfunc="sum", fill_value=0))
else:
    mech_onehot = pd.DataFrame()

# 每个 CASEID 的药物计数特征
per_case_basic = (
    drug_dm.groupby("CASEID")
           .agg(
               n_diab_records = ("GENERIC_STD","count"),          # 记录数（同一药多条也会计数）
               n_diab_unique  = ("GENERIC_STD", pd.Series.nunique) # 不同糖尿病药物数
           )
)

per_case_basic["is_multi_drug"] = (per_case_basic["n_diab_unique"] >= 2).astype(int)

# 物化性质聚合（均值/最大/求和）
agg_funcs = {
    "mol_weight":["mean","max","sum"],
    "logP":["mean","max","min"],
    "TPSA":["mean","max","sum"],
    "HBD":["mean","max","sum"],
    "HBA":["mean","max","sum"],
}
per_case_props = drug_dm.groupby("CASEID").agg(agg_funcs)
# 展平多重索引列名：mol_weight_mean 等
per_case_props.columns = ["_".join(c) for c in per_case_props.columns.to_flat_index()]

# 合并成报告层特征表
X_case = per_case_basic.join(per_case_props, how="left")
if not mech_onehot.empty:
    X_case = X_case.join(mech_onehot, how="left")

# ---------------- 合并 ADR_count 并生成标签 ----------------
X_case = X_case.reset_index()
X_case = X_case.merge(adr_count, on="CASEID", how="left")
X_case["ADR_count"] = X_case["ADR_count"].fillna(0).astype(int)
X_case["y_multiADR"] = (X_case["ADR_count"] >= 2).astype(int)

# ---------------- 合并 demo（可选） ----------------
demo_cols_keep = []
try:
    demo = pd.read_parquet(DEMO_FILE)
    # 识别 demo 中的 CASEID 列
    case_col_demo = first_col(demo, ["CASEID","caseid","PRIMARYID","primaryid","PrimaryID"])
    if case_col_demo is not None:
        demo = demo.rename(columns={case_col_demo: "CASEID"})
        # 自动寻找可能的人口学列
        age_col = first_col(demo, ["AGE","age","AGE_YRS","age_yrs"])
        sex_col = first_col(demo, ["SEX","sex","GENDER","gender"])
        wt_col  = first_col(demo, ["WT","weight","wt","Weight","BODY_WEIGHT"])
        ht_col  = first_col(demo, ["HT","height","ht","Height"])
        keep_map = {}
        if age_col: keep_map[age_col] = "age"
        if sex_col: keep_map[sex_col] = "sex"
        if wt_col:  keep_map[wt_col]  = "weight"
        if ht_col:  keep_map[ht_col]  = "height"
        demo_small = demo[["CASEID"] + list(keep_map.keys())].rename(columns=keep_map)
        X_case = X_case.merge(demo_small, on="CASEID", how="left")
        demo_cols_keep = list(keep_map.values())
except Exception as e:
    print(f"[WARN] 合并 demo 失败（可忽略）：{e}")

# ---------------- 清理并输出 ----------------
# 缺失填充（基础计数类缺失置0）
for c in ["n_diab_records","n_diab_unique","is_multi_drug"]:
    if c in X_case.columns:
        X_case[c] = X_case[c].fillna(0).astype(int)

# 输出
X_case.to_csv(OUT_FILE, index=False)

# 简要汇报
n_cases = X_case["CASEID"].nunique()
pos = int(X_case["y_multiADR"].sum())
neg = int((1 - X_case["y_multiADR"]).sum())
print("✅ 已生成患者层数据：", OUT_FILE)
print(f"   覆盖 CASEID 数：{n_cases}")
print(f"   标签分布：y=1（ADR>=2）：{pos}，y=0：{neg}，阳性比例={pos/(pos+neg):.3f}")
print("   可用人口学列：", demo_cols_keep)
print("   物化聚合示例列：", [c for c in X_case.columns if c.startswith('logP_') or c.startswith('TPSA_')][:8])
