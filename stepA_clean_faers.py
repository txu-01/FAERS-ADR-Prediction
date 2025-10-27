import pandas as pd
from pathlib import Path

# 你自己改成存放 FAERS ASCII 的文件夹路径
FAERS_DIR = Path(r"E:\MED\ASCII")

# 设置分隔符（如果失败可尝试 '|', ',', '\t'）
SEP = '$'

# 统一读取并清理的函数
def load_faers_table(table_prefix, usecols):
    files = list(FAERS_DIR.rglob(f"{table_prefix}*.TXT"))
    dfs = []
    for f in files:
        print(f"Loading {f.name} ...")
        try:
            df = pd.read_csv(f, sep=SEP, encoding='latin-1', dtype=str, low_memory=False)
            df.columns = df.columns.str.upper()
            keep = [c for c in usecols if c in df.columns]
            df = df[keep]
            dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to load {f.name}: {e}")
    if dfs:
        out = pd.concat(dfs, ignore_index=True).drop_duplicates()
        out = out.dropna(subset=[usecols[0]])  # drop rows missing PRIMARYID
        return out
    else:
        print(f"⚠️ No files found for {table_prefix}")
        return pd.DataFrame(columns=usecols)

# 读取三类表
demo = load_faers_table("DEMO", ["PRIMARYID", "CASEID", "AGE", "AGE_COD", "SEX", "FDA_DT"])
drug = load_faers_table("DRUG", ["PRIMARYID", "DRUGNAME", "ROLE_COD"])
reac = load_faers_table("REAC", ["PRIMARYID", "PT"])

# 基本信息检查
print("\n--- Loaded Shapes ---")
print("DEMO:", demo.shape)
print("DRUG:", drug.shape)
print("REAC:", reac.shape)

# 保存清理后的 parquet 文件（方便后续快速加载）
out_dir = FAERS_DIR / "cleaned"
out_dir.mkdir(exist_ok=True)

demo.to_parquet(out_dir / "demo_clean.parquet", index=False)
drug.to_parquet(out_dir / "drug_clean.parquet", index=False)
reac.to_parquet(out_dir / "reac_clean.parquet", index=False)

print("\n✅ All cleaned tables saved in:", out_dir)
