# -*- coding: utf-8 -*-
# stepA_context_features.py
# 从 cleaned/ 的 DEMO / DRUG / REAC 聚合患者/报告层特征
# 输出：outputs/X_context_features.parquet

import os
import pandas as pd
import numpy as np

# ====== 路径设置（按你之前的结构）======
CLEAN_DIR = r"ASCII\cleaned"   # 你保存清洗表的目录
OUT_DIR   = r"outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def _safe_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件：{path}")
    return pd.read_parquet(path)

def main():
    demo_path = os.path.join(CLEAN_DIR, "demo_clean.parquet")
    drug_path = os.path.join(CLEAN_DIR, "drug_clean.parquet")
    reac_path = os.path.join(CLEAN_DIR, "reac_clean.parquet")

    demo = _safe_read(demo_path)
    drug = _safe_read(drug_path)
    reac = _safe_read(reac_path)

    # 索引统一为字符串
    for df in (demo, drug, reac):
        if "PRIMARYID" in df.columns:
            df["PRIMARYID"] = df["PRIMARYID"].astype(str)

    # -------- DEMO 特征 --------
    keep_demo = ["PRIMARYID","AGE_YR","SEX","SERIOUS","I_F_COD","COUNTRY","EVENT_DT","REPT_DT"]
    for c in keep_demo:
        if c not in demo.columns: demo[c] = np.nan
    demo_feat = demo[keep_demo].copy()

    demo_feat["age_bin"] = pd.cut(demo_feat["AGE_YR"], bins=[-1,39,64,200],
                                  labels=["age_<40","age_40_64","age_65p"])
    demo_feat["age_missing"] = demo_feat["AGE_YR"].isna().astype(int)
    demo_feat["sex"] = demo_feat["SEX"].map({"M":"M","F":"F"}).fillna("UNK")
    # 有些清洗表里 SERIOUS 已经是 0/1；若缺失则置 0
    demo_feat["serious"] = demo_feat["SERIOUS"].fillna(0).astype(int)
    demo_feat["reporter"] = demo_feat["I_F_COD"].fillna("UNK").astype(str)
    demo_feat["country"] = demo_feat["COUNTRY"].fillna("UNK").astype(str)

    demo_g = demo_feat.groupby("PRIMARYID").agg({
        "AGE_YR":"mean",
        "age_bin":"first",
        "age_missing":"first",
        "sex":"first",
        "serious":"max",
        "reporter":"first",
        "country":"first",
        "EVENT_DT":"first",
        "REPT_DT":"first"
    }).reset_index()

    # -------- DRUG 特征 --------
    need_drug = ["PRIMARYID","DRUGNAME","ROLE_COD","ROUTE"]
    for c in need_drug:
        if c not in drug.columns: drug[c] = np.nan
    drug_use = drug[need_drug].copy()

    drug_g = drug_use.groupby("PRIMARYID").agg(
        num_drugs_per_report=("DRUGNAME","nunique")
    ).reset_index()
    drug_g["polypharmacy_ge3"] = (drug_g["num_drugs_per_report"] >= 3).astype(int)

    # 主药/并用药计数（如果有 ROLE_COD）
    role_ct = (drug_use.assign(ROLE_COD=drug_use["ROLE_COD"].fillna("UNK").astype(str))
               .pivot_table(index="PRIMARYID", columns="ROLE_COD",
                            values="DRUGNAME", aggfunc="nunique", fill_value=0))
    role_ct.columns = [f"role_{c.lower()}" for c in role_ct.columns]
    drug_g = drug_g.merge(role_ct, left_on="PRIMARYID", right_index=True, how="left")

    # 给药途径多样性
    route_div = (drug_use.assign(ROUTE=drug_use["ROUTE"].fillna("UNK").astype(str))
                 .groupby("PRIMARYID")["ROUTE"].nunique().rename("num_route_types"))
    drug_g = drug_g.merge(route_div, left_on="PRIMARYID", right_index=True, how="left")

    # -------- REAC 特征（仅统计，不作为训练特征以免泄漏）--------
    pt_col = "PT_NAME" if "PT_NAME" in reac.columns else ("PT" if "PT" in reac.columns else None)
    if pt_col is None:
        reac["PT"] = "UNK"
        pt_col = "PT"
    reac_use = reac[["PRIMARYID", pt_col]].rename(columns={pt_col:"PT"}).copy()
    reac_g = (reac_use.assign(PT=reac_use["PT"].astype(str))
              .groupby("PRIMARYID")["PT"].nunique()
              .rename("num_adr_reported").reset_index())

    # -------- 合并 --------
    X_ctx = demo_g.merge(drug_g, on="PRIMARYID", how="outer") \
                  .merge(reac_g, on="PRIMARYID", how="left")

    # 缺失填充
    for c in ["num_drugs_per_report","polypharmacy_ge3","num_route_types"]:
        X_ctx[c] = X_ctx[c].fillna(0).astype(int)
    for c in ["age_bin","sex","reporter","country"]:
        X_ctx[c] = X_ctx[c].fillna("UNK").astype(str)

    out_path = os.path.join(OUT_DIR, "X_context_features.parquet")
    X_ctx.to_parquet(out_path, index=False)
    print("✅ Saved:", out_path, X_ctx.shape)

if __name__ == "__main__":
    main()
