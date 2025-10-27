import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams['font.family'] = 'DejaVu Sans'

# 输入输出路径
IN_FILE = Path(r"E:/MED/cleaned/drug_level_metrics_clean.csv")
OUT_FILE = Path(r"E:/MED/cleaned/drug_level_metrics_merged.csv")
PLOT_DIR = Path(r"E:/MED/cleaned/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 读取数据
df = pd.read_csv(IN_FILE)

# 确认主要列
print("原始药物条目数:", df['DRUGNAME_STD'].nunique())

# 如果存在 GENERIC_STD 就用它作为通用名
if 'GENERIC_STD' in df.columns:
    df['GENERIC_STD'] = df['GENERIC_STD'].fillna(df['DRUGNAME_STD'])
else:
    df['GENERIC_STD'] = df['DRUGNAME_STD']

# 合并通用名
agg_cols = ['n_reports','total_adr','unique_PT_count']
df_merged = (
    df.groupby('GENERIC_STD')
      .agg({
          'n_reports':'sum',
          'total_adr':'sum',
          'unique_PT_count':'mean',  # 平均或求和都可，这里选平均
          'mol_weight':'first',
          'logP':'first',
          'TPSA':'first',
          'HBD':'first',
          'HBA':'first'
      })
      .reset_index()
)
df_merged['mean_ADR_per_case'] = np.where(
    df_merged['n_reports']>0,
    df_merged['total_adr']/df_merged['n_reports'],
    np.nan
)

# 保存合并后表
df_merged.to_csv(OUT_FILE, index=False)
print(f"✅ 已保存合并后的药物表：{OUT_FILE}（药物数量 {len(df_merged)}）")

# ---------- 绘图函数 ----------
def scatter_with_labels(x, y, xlabel, ylabel, title, fname):
    sub = df_merged[[x, y, 'GENERIC_STD']].dropna()
    plt.figure(figsize=(7,6))
    sns.scatterplot(data=sub, x=x, y=y)
    for i, row in sub.iterrows():
        plt.text(row[x], row[y], row['GENERIC_STD'], fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / fname, dpi=300)
    plt.close()
    print(f"✅ 绘图完成: {fname}")

# ---------- 绘制散点图 ----------
scatter_with_labels('logP','mean_ADR_per_case','LogP','Mean ADR per Case',
                    'LogP vs Mean ADR per Case','scatter_logP_meanADR.png')

scatter_with_labels('TPSA','mean_ADR_per_case','TPSA','Mean ADR per Case',
                    'TPSA vs Mean ADR per Case','scatter_TPSA_meanADR.png')

scatter_with_labels('HBD','unique_PT_count','HBD','Unique PT Count',
                    'HBD vs Unique PT Count','scatter_HBD_uniquePT.png')

# ---------- Spearman相关性 ----------
props = ['logP','TPSA','HBD','HBA','mol_weight']
targets = ['mean_ADR_per_case','unique_PT_count']

print("\n📊 Spearman 相关性分析")
for p in props:
    for t in targets:
        sub = df_merged[[p,t]].dropna()
        if len(sub) > 3:
            rho, pval = spearmanr(sub[p], sub[t])
            print(f"{p} vs {t}: ρ={rho:.3f}, p={pval:.4f}, n={len(sub)}")
        else:
            print(f"{p} vs {t}: 数据不足进行分析")

print(f"\n图表已保存至: {PLOT_DIR.resolve()}")
