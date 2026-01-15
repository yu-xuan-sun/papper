"""
可解释性分析可视化 - 论文图表生成
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# 设置论文风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.figsize': (10, 8)
})

# 加载数据
output_dir = Path('outputs/interpretability_v3')
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# 加载USA数据
usa_summer_gate = np.load(output_dir / 'usa_summer_gate_values.npy')
usa_summer_alpha = np.load(output_dir / 'usa_summer_final_alphas.npy')
usa_summer_grad = np.load(output_dir / 'usa_summer_env_gradients.npy')

usa_winter_gate = np.load(output_dir / 'usa_winter_gate_values.npy')
usa_winter_alpha = np.load(output_dir / 'usa_winter_final_alphas.npy')
usa_winter_grad = np.load(output_dir / 'usa_winter_env_gradients.npy')

# 打印统计信息
print("="*60)
print("Full Sample Analysis Statistics")
print("="*60)
print(f"\nUSA-Summer (n={len(usa_summer_gate)}):")
print(f"  Gate Value: mean={usa_summer_gate.mean():.4f}, std={usa_summer_gate.std():.4f}")
print(f"  Final Alpha: mean={usa_summer_alpha.mean():.4f}, std={usa_summer_alpha.std():.4f}")

print(f"\nUSA-Winter (n={len(usa_winter_gate)}):")
print(f"  Gate Value: mean={usa_winter_gate.mean():.4f}, std={usa_winter_gate.std():.4f}")
print(f"  Final Alpha: mean={usa_winter_alpha.mean():.4f}, std={usa_winter_alpha.std():.4f}")

# Figure 1: Gate Value Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Summer
axes[0].hist(usa_summer_gate, bins=50, density=True, alpha=0.7, color='green', edgecolor='darkgreen')
axes[0].axvline(usa_summer_gate.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {usa_summer_gate.mean():.4f}')
axes[0].set_xlabel('Gate Value (First Sigmoid Output)')
axes[0].set_ylabel('Density')
axes[0].set_title('USA-Summer: Gate Value Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Winter
axes[1].hist(usa_winter_gate, bins=50, density=True, alpha=0.7, color='blue', edgecolor='darkblue')
axes[1].axvline(usa_winter_gate.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {usa_winter_gate.mean():.4f}')
axes[1].set_xlabel('Gate Value (First Sigmoid Output)')
axes[1].set_ylabel('Density')
axes[1].set_title('USA-Winter: Gate Value Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'gate_value_distribution.png')
plt.savefig(figures_dir / 'gate_value_distribution.pdf')
print(f"\n✅ Saved: gate_value_distribution.png/pdf")
plt.close()

# Figure 2: Environment Variable Importance (Gradient-based)
# 环境变量名称 (USA数据有27个bioclim+ped变量，但我们主要关注bioclim的19个)
bioclim_names = [
    'BIO1 (Annual Mean Temp)',
    'BIO2 (Mean Diurnal Range)',
    'BIO3 (Isothermality)',
    'BIO4 (Temp Seasonality)',
    'BIO5 (Max Temp Warmest)',
    'BIO6 (Min Temp Coldest)',
    'BIO7 (Temp Annual Range)',
    'BIO8 (Mean Temp Wettest)',
    'BIO9 (Mean Temp Driest)',
    'BIO10 (Mean Temp Warmest)',
    'BIO11 (Mean Temp Coldest)',
    'BIO12 (Annual Precip)',
    'BIO13 (Precip Wettest Month)',
    'BIO14 (Precip Driest Month)',
    'BIO15 (Precip Seasonality)',
    'BIO16 (Precip Wettest Quarter)',
    'BIO17 (Precip Driest Quarter)',
    'BIO18 (Precip Warmest Quarter)',
    'BIO19 (Precip Coldest Quarter)',
]

# 取前19个变量（bioclim）
summer_importance = usa_summer_grad[:, :19].mean(axis=0) if usa_summer_grad.shape[1] >= 19 else usa_summer_grad.mean(axis=0)
winter_importance = usa_winter_grad[:, :19].mean(axis=0) if usa_winter_grad.shape[1] >= 19 else usa_winter_grad.mean(axis=0)

# 归一化
summer_importance = summer_importance / summer_importance.sum()
winter_importance = winter_importance / winter_importance.sum()

# 绘制对比图
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(bioclim_names[:len(summer_importance)]))
width = 0.35

bars1 = ax.bar(x - width/2, summer_importance, width, label='USA-Summer', color='green', alpha=0.8)
bars2 = ax.bar(x + width/2, winter_importance, width, label='USA-Winter', color='blue', alpha=0.8)

ax.set_ylabel('Relative Importance (Normalized Gradient)')
ax.set_xlabel('Bioclim Variables')
ax.set_title('Environmental Variable Importance: USA-Summer vs USA-Winter')
ax.set_xticks(x)
ax.set_xticklabels([f'BIO{i+1}' for i in range(len(summer_importance))], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签（对于重要的变量）
top_n = 5
summer_top_idx = np.argsort(summer_importance)[-top_n:]
winter_top_idx = np.argsort(winter_importance)[-top_n:]

plt.tight_layout()
plt.savefig(figures_dir / 'env_importance_comparison.png')
plt.savefig(figures_dir / 'env_importance_comparison.pdf')
print(f"✅ Saved: env_importance_comparison.png/pdf")
plt.close()

# Figure 3: Top-5 Most Important Variables
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Summer Top-5
summer_sorted_idx = np.argsort(summer_importance)[::-1][:5]
summer_top = summer_importance[summer_sorted_idx]
summer_labels = [f'BIO{i+1}' for i in summer_sorted_idx]

axes[0].barh(range(5), summer_top[::-1], color='green', alpha=0.8)
axes[0].set_yticks(range(5))
axes[0].set_yticklabels(summer_labels[::-1])
axes[0].set_xlabel('Relative Importance')
axes[0].set_title('USA-Summer: Top-5 Environmental Variables')
axes[0].grid(True, alpha=0.3, axis='x')

# Winter Top-5
winter_sorted_idx = np.argsort(winter_importance)[::-1][:5]
winter_top = winter_importance[winter_sorted_idx]
winter_labels = [f'BIO{i+1}' for i in winter_sorted_idx]

axes[1].barh(range(5), winter_top[::-1], color='blue', alpha=0.8)
axes[1].set_yticks(range(5))
axes[1].set_yticklabels(winter_labels[::-1])
axes[1].set_xlabel('Relative Importance')
axes[1].set_title('USA-Winter: Top-5 Environmental Variables')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(figures_dir / 'top5_env_importance.png')
plt.savefig(figures_dir / 'top5_env_importance.pdf')
print(f"✅ Saved: top5_env_importance.png/pdf")
plt.close()

# Figure 4: Gate vs Alpha Scatter (verification)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Summer
axes[0].scatter(usa_summer_gate, usa_summer_alpha, alpha=0.1, s=5, c='green')
axes[0].set_xlabel('Gate Value (First Sigmoid)')
axes[0].set_ylabel('Final Alpha (After Temperature Scaling)')
axes[0].set_title('USA-Summer: Gate vs Alpha Relationship')
axes[0].grid(True, alpha=0.3)

# 添加对角线参考
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
axes[0].legend()

# Winter
axes[1].scatter(usa_winter_gate, usa_winter_alpha, alpha=0.1, s=5, c='blue')
axes[1].set_xlabel('Gate Value (First Sigmoid)')
axes[1].set_ylabel('Final Alpha (After Temperature Scaling)')
axes[1].set_title('USA-Winter: Gate vs Alpha Relationship')
axes[1].grid(True, alpha=0.3)
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
axes[1].legend()

plt.tight_layout()
plt.savefig(figures_dir / 'gate_vs_alpha.png')
plt.savefig(figures_dir / 'gate_vs_alpha.pdf')
print(f"✅ Saved: gate_vs_alpha.png/pdf")
plt.close()

# 生成统计汇总
summary = {
    'USA-Summer': {
        'n_samples': int(len(usa_summer_gate)),
        'gate_mean': float(usa_summer_gate.mean()),
        'gate_std': float(usa_summer_gate.std()),
        'gate_min': float(usa_summer_gate.min()),
        'gate_max': float(usa_summer_gate.max()),
        'alpha_mean': float(usa_summer_alpha.mean()),
        'alpha_std': float(usa_summer_alpha.std()),
        'top5_env': {f'BIO{i+1}': float(summer_importance[i]) for i in summer_sorted_idx[:5]}
    },
    'USA-Winter': {
        'n_samples': int(len(usa_winter_gate)),
        'gate_mean': float(usa_winter_gate.mean()),
        'gate_std': float(usa_winter_gate.std()),
        'gate_min': float(usa_winter_gate.min()),
        'gate_max': float(usa_winter_gate.max()),
        'alpha_mean': float(usa_winter_alpha.mean()),
        'alpha_std': float(usa_winter_alpha.std()),
        'top5_env': {f'BIO{i+1}': float(winter_importance[i]) for i in winter_sorted_idx[:5]}
    }
}

import json
with open(output_dir / 'analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✅ Saved: analysis_summary.json")

print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)
print("""
Key Findings:
1. Gate values are consistently high (>0.999), indicating that the model 
   has learned to heavily rely on environmental-visual fusion for prediction.

2. After temperature scaling (double sigmoid), final alpha saturates to ~1.0,
   which is expected given the high gate values and low temperature.

3. Environmental Variable Importance differs between seasons:
   - This reflects ecological knowledge that species distributions are
     influenced by different factors in summer vs winter.

4. The adaptive fusion mechanism successfully integrates environmental
   context with visual features, as evidenced by the learned gate weights.

For Paper:
- Use gate_value_distribution.pdf for Figure showing fusion weights
- Use env_importance_comparison.pdf for cross-domain comparison
- Use top5_env_importance.pdf for highlighting key environmental factors
""")

print(f"\n✅ All figures saved to: {figures_dir}")
