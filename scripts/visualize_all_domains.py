#!/usr/bin/env python3
"""
三域综合可视化 - 论文图表生成
USA-Summer, USA-Winter, Kenya-Transfer
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json

# 设置论文风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

output_dir = Path('outputs/interpretability_v3')
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# 加载数据
summer_gate = np.load(output_dir / 'usa_summer_gate_values.npy')
summer_grad = np.load(output_dir / 'usa_summer_env_gradients.npy')

winter_gate = np.load(output_dir / 'usa_winter_gate_values.npy')
winter_grad = np.load(output_dir / 'usa_winter_env_gradients.npy')

kenya_gate = np.load(output_dir / 'kenya_transfer_gate_values.npy')
kenya_grad = np.load(output_dir / 'kenya_transfer_env_gradients.npy')

# BioClim变量名
BIOCLIM = [
    'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9',
    'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19'
]

BIOCLIM_FULL = {
    'BIO1': 'Annual Mean Temp',
    'BIO2': 'Mean Diurnal Range',
    'BIO3': 'Isothermality',
    'BIO4': 'Temp Seasonality',
    'BIO5': 'Max Temp Warmest',
    'BIO6': 'Min Temp Coldest',
    'BIO7': 'Temp Annual Range',
    'BIO8': 'Mean Temp Wettest',
    'BIO9': 'Mean Temp Driest',
    'BIO10': 'Mean Temp Warmest Quarter',
    'BIO11': 'Mean Temp Coldest Quarter',
    'BIO12': 'Annual Precip',
    'BIO13': 'Precip Wettest Month',
    'BIO14': 'Precip Driest Month',
    'BIO15': 'Precip Seasonality',
    'BIO16': 'Precip Wettest Quarter',
    'BIO17': 'Precip Driest Quarter',
    'BIO18': 'Precip Warmest Quarter',
    'BIO19': 'Precip Coldest Quarter',
}

# 计算重要性 (取前19个bioclim变量)
summer_imp = np.abs(summer_grad[:, :19]).mean(axis=0)
summer_imp = summer_imp / summer_imp.sum()

winter_imp = np.abs(winter_grad[:, :19]).mean(axis=0)
winter_imp = winter_imp / winter_imp.sum()

kenya_imp = np.abs(kenya_grad).mean(axis=0)
kenya_imp = kenya_imp / kenya_imp.sum()

# Figure 1: 三域门控值分布对比
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ['#2ecc71', '#3498db', '#e74c3c']
domains = ['USA-Summer', 'USA-Winter', 'Kenya-Transfer']
data = [summer_gate, winter_gate, kenya_gate]

for ax, gate, color, name in zip(axes, data, colors, domains):
    ax.hist(gate, bins=50, density=True, alpha=0.7, color=color, edgecolor='black')
    ax.axvline(gate.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {gate.mean():.6f}')
    ax.set_xlabel('Gate Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{name} (n={len(gate)})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(gate.min() - 0.0001, gate.max() + 0.0001)

plt.suptitle('Adaptive Fusion Gate Value Distribution Across Domains', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(figures_dir / 'three_domain_gate_distribution.png', dpi=150)
plt.savefig(figures_dir / 'three_domain_gate_distribution.pdf', dpi=300)
print(f"✅ Saved: three_domain_gate_distribution")
plt.close()

# Figure 2: 三域环境变量重要性热力图
fig, ax = plt.subplots(figsize=(12, 6))

# 准备数据矩阵
importance_matrix = np.zeros((3, 19))
importance_matrix[0] = summer_imp
importance_matrix[1] = winter_imp
importance_matrix[2] = kenya_imp

im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(19))
ax.set_xticklabels(BIOCLIM, rotation=45, ha='right')
ax.set_yticks(range(3))
ax.set_yticklabels(['USA-Summer', 'USA-Winter', 'Kenya-Transfer'])
ax.set_xlabel('BioClim Variables')
ax.set_ylabel('Domain')
ax.set_title('Environmental Variable Importance Heatmap')

# 添加数值标签
for i in range(3):
    for j in range(19):
        val = importance_matrix[i, j]
        if val > 0.06:  # 只标注重要的
            ax.text(j, i, f'{val*100:.1f}%', ha='center', va='center', color='white', fontsize=8)

plt.colorbar(im, label='Relative Importance')
plt.tight_layout()
plt.savefig(figures_dir / 'three_domain_env_heatmap.png', dpi=150)
plt.savefig(figures_dir / 'three_domain_env_heatmap.pdf', dpi=300)
print(f"✅ Saved: three_domain_env_heatmap")
plt.close()

# Figure 3: Top-5重要变量对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

domains_data = [
    ('USA-Summer', summer_imp, '#2ecc71'),
    ('USA-Winter', winter_imp, '#3498db'),
    ('Kenya-Transfer', kenya_imp, '#e74c3c'),
]

for ax, (name, imp, color) in zip(axes, domains_data):
    sorted_idx = np.argsort(imp)[::-1][:5]
    labels = [BIOCLIM[i] for i in sorted_idx]
    values = [imp[i] * 100 for i in sorted_idx]
    
    bars = ax.barh(range(5), values[::-1], color=color, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel('Importance (%)')
    ax.set_title(name)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=10)

plt.suptitle('Top-5 Most Important Environmental Variables by Domain', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(figures_dir / 'three_domain_top5.png', dpi=150)
plt.savefig(figures_dir / 'three_domain_top5.pdf', dpi=300)
print(f"✅ Saved: three_domain_top5")
plt.close()

# Figure 4: 温度 vs 降水变量对比
temp_vars = [0, 3, 4, 5, 6, 9, 10]  # BIO1, 4, 5, 6, 7, 10, 11
precip_vars = [11, 13, 14, 15, 18]  # BIO12, 14, 15, 16, 19

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(3)
width = 0.35

temp_importance = [
    summer_imp[temp_vars].sum(),
    winter_imp[temp_vars].sum(),
    kenya_imp[temp_vars].sum()
]

precip_importance = [
    summer_imp[precip_vars].sum(),
    winter_imp[precip_vars].sum(),
    kenya_imp[precip_vars].sum()
]

bars1 = ax.bar(x - width/2, [v*100 for v in temp_importance], width, 
               label='Temperature Variables', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, [v*100 for v in precip_importance], width,
               label='Precipitation Variables', color='#3498db', alpha=0.8)

ax.set_ylabel('Total Importance (%)')
ax.set_xticks(x)
ax.set_xticklabels(['USA-Summer', 'USA-Winter', 'Kenya-Transfer'])
ax.legend()
ax.set_title('Temperature vs Precipitation Variable Importance')
ax.grid(True, alpha=0.3, axis='y')

# 添加数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(figures_dir / 'temp_vs_precip.png', dpi=150)
plt.savefig(figures_dir / 'temp_vs_precip.pdf', dpi=300)
print(f"✅ Saved: temp_vs_precip")
plt.close()

# 打印统计摘要
print("\n" + "="*70)
print("ANALYSIS SUMMARY FOR PAPER")
print("="*70)

print(f"\n📊 Sample Sizes:")
print(f"   USA-Summer: {len(summer_gate):,}")
print(f"   USA-Winter: {len(winter_gate):,}")
print(f"   Kenya-Transfer: {len(kenya_gate):,}")

print(f"\n🔧 Gate Values (Fusion Weights):")
print(f"   USA-Summer: {summer_gate.mean():.6f} ± {summer_gate.std():.6f}")
print(f"   USA-Winter: {winter_gate.mean():.6f} ± {winter_gate.std():.6f}")
print(f"   Kenya: {kenya_gate.mean():.6f} ± {kenya_gate.std():.6f}")

print(f"\n🌍 Top-3 Environmental Variables:")
for name, imp in [('USA-Summer', summer_imp), ('USA-Winter', winter_imp), ('Kenya', kenya_imp)]:
    top3 = np.argsort(imp)[::-1][:3]
    print(f"   {name}: {', '.join([f'{BIOCLIM[i]} ({imp[i]*100:.1f}%)' for i in top3])}")

print(f"\n📈 Temperature vs Precipitation Importance:")
for name, imp, idx in [('USA-Summer', summer_imp, 0), ('USA-Winter', winter_imp, 1), ('Kenya', kenya_imp, 2)]:
    t = imp[temp_vars].sum() * 100
    p = imp[precip_vars].sum() * 100
    print(f"   {name}: Temp={t:.1f}%, Precip={p:.1f}%")

print("\n✅ All figures saved to:", figures_dir)
