import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------- 药名规范化 ---------------------------

def standardize_drugname(s):
    if pd.isna(s):
        return s
    return (str(s)
            .strip()
            .upper()
            .replace('-', ' ')
            .replace('_', ' '))

def build_synonym_map():
    return {
        'OZEMPIC': 'Semaglutide',
        'RYBELSUS': 'Semaglutide',
        'WEGOVY': 'Semaglutide',
        'TRULICITY': 'Dulaglutide',
        'BYDUREON': 'Exenatide',
        'BYETTA': 'Exenatide',
        'MOUNJARO': 'Tirzepatide',
        'JARDIANCE': 'Empagliflozin',
        'FORXIGA': 'Dapagliflozin',
        'FARXIGA': 'Dapagliflozin',
        'INVOKANA': 'Canagliflozin',
        'JANUVIA': 'Sitagliptin',
        'TRADJENTA': 'Linagliptin',
        'ONGLYZA': 'Saxagliptin',
        'GLUCOPHAGE': 'Metformin',
        'ACTOS': 'Pioglitazone',
    }

def map_to_external(drugname_series, external_df):
    keys = {k.upper(): k for k in external_df['drug_name'].astype(str).tolist()}
    synonyms = build_synonym_map()
    out = []
    for x in drugname_series.astype(str):
        xn = standardize_drugname(x)
        if xn in keys:
            out.append(keys[xn]); continue
        token = xn.split()[0] if xn else ""
        if token in keys:
            out.append(keys[token]); continue
        if token in synonyms:
            out.append(synonyms[token]); continue
        found = None
        for k_up, k_orig in keys.items():
            if k_up in xn:
                found = k_orig; break
        out.append(found)
    return pd.Series(out, index=drugname_series.index)

# --------------------------- PRR / ROR 计算 ---------------------------

def disproportionality_counts(drug_pt_df, drug_col='drug_mapped', pt_col='pt'):
    patients = drug_pt_df['PRIMARYID'].nunique()
    a = (drug_pt_df.dropna(subset=[drug_col, pt_col])
         .groupby([drug_col, pt_col])['PRIMARYID'].nunique().rename('a'))
    n_drug = (drug_pt_df.dropna(subset=[drug_col])
              .groupby(drug_col)['PRIMARYID'].nunique().rename('n_drug'))
    n_pt = (drug_pt_df.dropna(subset=[pt_col])
            .groupby(pt_col)['PRIMARYID'].nunique().rename('n_pt'))
    out = a.to_frame()
    out = out.join(n_drug, on=drug_col)
    out = out.join(n_pt, on=pt_col)
    out['b'] = out['n_drug'] - out['a']
    out['c'] = out['n_pt'] - out['a']
    out['d'] = patients - out[['a','b','c']].sum(axis=1)
    return out[['a','b','c','d']].reset_index()

def prr_ror(df_counts):
    eps = 0.5
    a = df_counts['a']+eps; b = df_counts['b']+eps
    c = df_counts['c']+eps; d = df_counts['d']+eps
    prr = (a/(a+b)) / (c/(c+d))
    ror = (a*d)/(b*c)
    se_log_prr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    se_log_ror = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = 1.96
    out = df_counts.copy()
    out['PRR'] = prr
    out['PRR_L'] = np.exp(np.log(prr)-z*se_log_prr)
    out['PRR_H'] = np.exp(np.log(prr)+z*se_log_prr)
    out['ROR'] = ror
    out['ROR_L'] = np.exp(np.log(ror)-z*se_log_ror)
    out['ROR_H'] = np.exp(np.log(ror)+z*se_log_ror)
    return out

# --------------------------- 合并表格 ---------------------------

def build_model_table(demo, drug, reac, external_df):
    demo['FDA_DT'] = pd.to_datetime(demo['FDA_DT'], errors='coerce')
    demo['YEAR'] = demo['FDA_DT'].dt.year
    demo['QTR'] = demo['FDA_DT'].dt.quarter

    drug['DRUG_STD'] = drug['DRUGNAME'].apply(standardize_drugname)
    drug['drug_mapped'] = map_to_external(drug['DRUG_STD'], external_df)
    drug = drug[drug['ROLE_COD'].isin(['PS','SS','C'])]

    reac = reac.rename(columns={'PT':'pt'})

    merged = (drug[['PRIMARYID','drug_mapped']]
              .merge(reac[['PRIMARYID','pt']], on='PRIMARYID', how='inner')
              .dropna(subset=['drug_mapped','pt']))

    merged = merged.merge(
        demo[['PRIMARYID','AGE','AGE_COD','SEX','YEAR','QTR']],
        on='PRIMARYID', how='left'
    )

    ext = external_df.rename(columns={'drug_name':'drug_mapped'})
    X = merged.merge(ext, on='drug_mapped', how='left')

    Y = (merged.assign(value=1)
         .pivot_table(index='PRIMARYID', columns='pt',
                      values='value', fill_value=0, aggfunc='max'))
    return X, Y, merged

# --------------------------- 主程序 ---------------------------

def main(args):
    clean = Path(args.clean_dir)
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)

    demo = pd.read_parquet(clean / "demo_clean.parquet")
    drug = pd.read_parquet(clean / "drug_clean.parquet")
    reac = pd.read_parquet(clean / "reac_clean.parquet")
    ext = pd.read_csv(args.external_csv)

    print("🔧 合并表格中...")
    X, Y, merged = build_model_table(demo, drug, reac, ext)

    print("📈 计算 PRR / ROR...")
    counts = disproportionality_counts(merged)
    metrics = prr_ror(counts)

    X.to_parquet(outdir / "X_patient_level.parquet", index=False)
    Y.to_parquet(outdir / "Y_multilabel_PT.parquet")
    merged.to_parquet(outdir / "merged_patient_drug_pt.parquet", index=False)
    metrics.to_csv(outdir / "signals_prr_ror.csv", index=False)

    print("\n✅ 已完成，结果保存在：", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clean_dir", required=True, help="含有 demo_clean/drug_clean/reac_clean.parquet 的文件夹")
    p.add_argument("--external_csv", required=True, help="你的 drug_list_ext.csv 路径")
    p.add_argument("--outdir", default="./outputs", help="输出文件夹")
    args = p.parse_args()
    main(args)
