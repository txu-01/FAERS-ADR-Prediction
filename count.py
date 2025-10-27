import pandas as pd
from pathlib import Path

drug_file = Path(r"E:\MED\ASCII\cleaned\drug_clean.parquet")
drug_list_file = Path(r"E:\MED\drug_list_ext.csv")

# 读取
drug = pd.read_parquet(drug_file)
druglist = pd.read_csv(drug_list_file)

# 标准化药名
def std(s):
    return s.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)

# 确认药名列
drug_name_col = 'DRUGNAME_STD' if 'DRUGNAME_STD' in drug.columns else (
    'DRUGNAME' if 'DRUGNAME' in drug.columns else None
)
if drug_name_col is None:
    raise ValueError("❌ drug 表中未找到药物名称列")

drug['DRUG_STD'] = std(drug[drug_name_col])
druglist['DRUG_STD'] = std(druglist['drug_name'])

# 确认病例ID列
possible_case_cols = ['CASEID','caseid','PrimaryID','PRIMARYID','primaryid']
case_col = None
for c in possible_case_cols:
    if c in drug.columns:
        case_col = c
        break

if case_col is None:
    raise ValueError("❌ drug 表中未找到 CASEID 或 primaryid 列，请检查列名")

# 构建糖尿病药物列表
diab_drugs = set(druglist['DRUG_STD'].unique())

# 筛选糖尿病药物记录
diab_df = drug[drug['DRUG_STD'].isin(diab_drugs)].copy()

# 统计
n_records = len(diab_df)
n_cases = diab_df[case_col].nunique()

print("✅ 糖尿病药物相关用药记录条数：", n_records)
print("✅ 涉及的唯一报告/患者 CASEID 数量：", n_cases)
