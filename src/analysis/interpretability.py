"""
多尺度环境-视觉联合可解释性分析模块
Multi-Scale Environment-Visual Joint Interpretability Analysis

核心创新点:
1. 门控权重分析 - 揭示模型如何动态平衡视觉与环境信息
2. 环境变量贡献度分析 - 识别关键环境因子
3. 跨域/跨季节对比分析 - 理解迁移学习中的融合模式变化

与传统方法的区别:
- Grad-CAM: 只能分析视觉空间注意力，无法解释多模态融合
- SHAP: 计算复杂度O(2^n)，对高维特征不可行
- Ours: 直接读取门控权重，零额外计算成本，可分析融合决策过程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json


# 环境变量名称映射 (bioclim变量)
BIOCLIM_NAMES = {
    0: 'BIO1: Annual Mean Temp',
    1: 'BIO2: Mean Diurnal Range',
    2: 'BIO3: Isothermality',
    3: 'BIO4: Temp Seasonality',
    4: 'BIO5: Max Temp Warmest Month',
    5: 'BIO6: Min Temp Coldest Month',
    6: 'BIO7: Temp Annual Range',
    7: 'BIO8: Mean Temp Wettest Quarter',
    8: 'BIO9: Mean Temp Driest Quarter',
    9: 'BIO10: Mean Temp Warmest Quarter',
    10: 'BIO11: Mean Temp Coldest Quarter',
    11: 'BIO12: Annual Precipitation',
    12: 'BIO13: Precip Wettest Month',
    13: 'BIO14: Precip Driest Month',
    14: 'BIO15: Precip Seasonality',
    15: 'BIO16: Precip Wettest Quarter',
    16: 'BIO17: Precip Driest Quarter',
    17: 'BIO18: Precip Warmest Quarter',
    18: 'BIO19: Precip Coldest Quarter',
}

# 简短名称用于绘图
BIOCLIM_SHORT_NAMES = {
    0: 'AnnualTemp',
    1: 'DiurnalRange',
    2: 'Isothermality',
    3: 'TempSeason',
    4: 'MaxTempWarm',
    5: 'MinTempCold',
    6: 'TempRange',
    7: 'TempWetQ',
    8: 'TempDryQ',
    9: 'TempWarmQ',
    10: 'TempColdQ',
    11: 'AnnualPrecip',
    12: 'PrecipWetM',
    13: 'PrecipDryM',
    14: 'PrecipSeason',
    15: 'PrecipWetQ',
    16: 'PrecipDryQ',
    17: 'PrecipWarmQ',
    18: 'PrecipColdQ',
}


class InterpretabilityAnalyzer:
    """
    多模态融合可解释性分析器
    
    分析内容:
    1. 门控权重 (alpha): 视觉 vs 环境信息的动态权衡
    2. Cross-Attention权重: 环境信息如何被视觉特征查询
    3. 环境变量梯度: 各环境变量对预测的贡献
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: 训练好的模型 (Dinov2AdapterPrompt)
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 存储分析结果
        self.gate_weights = []
        self.attention_weights = []
        self.env_gradients = []
        self.predictions = []
        self.targets = []
        self.env_features = []
        
    def reset(self):
        """重置存储"""
        self.gate_weights = []
        self.attention_weights = []
        self.env_gradients = []
        self.predictions = []
        self.targets = []
        self.env_features = []
    
    @torch.no_grad()
    def extract_gate_weights(
        self, 
        img: torch.Tensor, 
        env: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        提取门控权重
        
        Args:
            img: 图像 [B, C, H, W]
            env: 环境特征 [B, env_dim]
            
        Returns:
            dict with 'alpha' (gate weight), 'visual_feat', 'env_encoded'
        """
        img = img.to(self.device)
        env = env.to(self.device)
        
        # 提取视觉特征
        visual_feat = self.model.forward_visual_features(img)
        
        # 获取融合模块
        fusion = self.model.fusion
        
        if fusion is None:
            return {'alpha': torch.ones(img.size(0), 1, device=self.device) * 0.5}
        
        # 调用fusion并获取alpha
        if hasattr(fusion, 'forward'):
            try:
                fused, alpha = fusion(visual_feat, env, return_alpha=True)
            except TypeError:
                fused = fusion(visual_feat, env)
                alpha = torch.ones(img.size(0), 1, device=self.device) * 0.5
        else:
            fused = fusion(visual_feat, env)
            alpha = torch.ones(img.size(0), 1, device=self.device) * 0.5
        
        return {
            'alpha': alpha,
            'visual_feat': visual_feat,
            'fused_feat': fused
        }
    
    def compute_env_gradient(
        self,
        img: torch.Tensor,
        env: torch.Tensor,
        target_species: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算环境变量的梯度贡献
        
        Args:
            img: 图像
            env: 环境特征 [B, env_dim]
            target_species: 目标物种索引 [B] 或 None (使用预测最高的)
            
        Returns:
            gradients: [B, env_dim] 每个环境变量的梯度
        """
        img = img.to(self.device)
        env = env.to(self.device).requires_grad_(True)
        
        # 前向传播
        logits = self.model(img, env)
        
        # 选择目标
        if target_species is None:
            target_species = logits.argmax(dim=1)
        else:
            target_species = target_species.to(self.device)
        
        # 计算目标类别的分数
        batch_size = logits.size(0)
        target_scores = logits[torch.arange(batch_size, device=self.device), target_species]
        
        # 反向传播
        grad = torch.autograd.grad(
            outputs=target_scores.sum(),
            inputs=env,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return grad.detach()
    
    def analyze_batch(
        self,
        img: torch.Tensor,
        env: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        分析一个batch的数据
        
        Returns:
            dict with analysis results
        """
        # 提取门控权重
        gate_info = self.extract_gate_weights(img, env)
        alpha = gate_info['alpha']
        
        # 计算环境梯度
        env_grad = self.compute_env_gradient(img, env, target)
        
        # 获取预测
        with torch.no_grad():
            logits = self.model(img.to(self.device), env.to(self.device))
            pred = torch.sigmoid(logits)
        
        # 存储
        self.gate_weights.append(alpha.cpu())
        self.env_gradients.append(env_grad.cpu())
        self.predictions.append(pred.cpu())
        self.env_features.append(env.cpu() if isinstance(env, torch.Tensor) else torch.tensor(env))
        if target is not None:
            self.targets.append(target.cpu() if isinstance(target, torch.Tensor) else torch.tensor(target))
        
        return {
            'alpha': alpha.cpu(),
            'env_gradient': env_grad.cpu(),
            'prediction': pred.cpu()
        }
    
    def analyze_dataloader(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        desc: str = "Analyzing"
    ):
        """
        分析整个dataloader
        """
        self.reset()
        
        for i, batch in enumerate(tqdm(dataloader, desc=desc)):
            if max_batches and i >= max_batches:
                break
            
            # 解包batch (根据你的数据格式调整)
            if isinstance(batch, dict):
                img = batch['image']
                env = batch['env']
                target = batch.get('target', None)
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    img, env, target = batch[0], batch[1], batch[2]
                else:
                    img, env = batch[0], batch[1]
                    target = None
            else:
                raise ValueError(f"Unknown batch format: {type(batch)}")
            
            self.analyze_batch(img, env, target)
    
    def get_aggregated_results(self) -> Dict[str, np.ndarray]:
        """
        聚合所有分析结果
        """
        results = {}
        
        if self.gate_weights:
            results['alpha'] = torch.cat(self.gate_weights, dim=0).numpy()
        
        if self.env_gradients:
            results['env_gradient'] = torch.cat(self.env_gradients, dim=0).numpy()
        
        if self.env_features:
            results['env_features'] = torch.cat(self.env_features, dim=0).numpy()
        
        if self.predictions:
            results['predictions'] = torch.cat(self.predictions, dim=0).numpy()
        
        if self.targets:
            results['targets'] = torch.cat(self.targets, dim=0).numpy()
        
        return results
    
    def compute_env_importance(self) -> pd.DataFrame:
        """
        计算环境变量重要性
        
        Returns:
            DataFrame with importance scores for each env variable
        """
        results = self.get_aggregated_results()
        
        if 'env_gradient' not in results:
            raise ValueError("No gradient data available. Run analyze_dataloader first.")
        
        gradients = results['env_gradient']  # [N, env_dim]
        
        # 多种重要性度量
        importance = {
            'mean_abs_gradient': np.abs(gradients).mean(axis=0),
            'std_gradient': gradients.std(axis=0),
            'max_abs_gradient': np.abs(gradients).max(axis=0),
            'positive_ratio': (gradients > 0).mean(axis=0),
        }
        
        # 创建DataFrame
        env_dim = gradients.shape[1]
        var_names = [BIOCLIM_SHORT_NAMES.get(i, f'ENV_{i}') for i in range(env_dim)]
        
        df = pd.DataFrame(importance, index=var_names)
        df['rank'] = df['mean_abs_gradient'].rank(ascending=False)
        df = df.sort_values('mean_abs_gradient', ascending=False)
        
        return df


class InterpretabilityVisualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_gate_distribution(
        alphas: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Gate Weight (alpha) Distribution"
    ):
        """
        绘制门控权重分布
        
        Args:
            alphas: [N, 1] 或 [N] 门控权重
        """
        alphas = alphas.flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 直方图
        axes[0].hist(alphas, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(alphas.mean(), color='red', linestyle='--', 
                       label=f'Mean: {alphas.mean():.3f}')
        axes[0].axvline(0.5, color='green', linestyle=':', label='Balanced (0.5)')
        axes[0].set_xlabel('alpha (Gate Weight)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Gate Weights')
        axes[0].legend()
        
        # 核密度估计
        sns.kdeplot(alphas, ax=axes[1], fill=True, color='steelblue')
        axes[1].axvline(alphas.mean(), color='red', linestyle='--')
        axes[1].set_xlabel('alpha (Gate Weight)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('KDE of Gate Weights')
        
        # 解释文本
        interpretation = (
            f"alpha > 0.5: Model relies more on environment-enhanced features\n"
            f"alpha < 0.5: Model relies more on pure visual features\n"
            f"Mean alpha = {alphas.mean():.3f}, Std = {alphas.std():.3f}"
        )
        fig.text(0.5, -0.05, interpretation, ha='center', fontsize=10, 
                style='italic', transform=fig.transFigure)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved gate distribution plot to {save_path}")
        
        plt.show()
        return fig
    
    @staticmethod
    def plot_env_importance(
        importance_df: pd.DataFrame,
        top_k: int = 15,
        save_path: Optional[str] = None,
        title: str = "Environmental Variable Importance"
    ):
        """
        绘制环境变量重要性
        """
        df = importance_df.head(top_k)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 条形图 - 平均绝对梯度
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df)))
        bars = axes[0].barh(range(len(df)), df['mean_abs_gradient'], color=colors)
        axes[0].set_yticks(range(len(df)))
        axes[0].set_yticklabels(df.index)
        axes[0].set_xlabel('Mean |Gradient|')
        axes[0].set_title('Importance by Gradient Magnitude')
        axes[0].invert_yaxis()
        
        # 添加数值标签
        for i, (idx, row) in enumerate(df.iterrows()):
            axes[0].text(row['mean_abs_gradient'] + 0.001, i, 
                        f"{row['mean_abs_gradient']:.4f}", va='center', fontsize=8)
        
        # 正负比例
        pos_ratio = df['positive_ratio']
        neg_ratio = 1 - pos_ratio
        
        axes[1].barh(range(len(df)), pos_ratio, color='green', alpha=0.7, label='Positive')
        axes[1].barh(range(len(df)), -neg_ratio, color='red', alpha=0.7, label='Negative')
        axes[1].set_yticks(range(len(df)))
        axes[1].set_yticklabels(df.index)
        axes[1].set_xlabel('Gradient Direction Ratio')
        axes[1].set_title('Positive vs Negative Gradient Ratio')
        axes[1].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_xlim(-1, 1)
        axes[1].legend()
        axes[1].invert_yaxis()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved importance plot to {save_path}")
        
        plt.show()
        return fig
    
    @staticmethod
    def plot_cross_domain_comparison(
        results_dict: Dict[str, Dict[str, np.ndarray]],
        save_path: Optional[str] = None
    ):
        """
        跨域对比分析
        
        Args:
            results_dict: {'USA-Summer': results, 'USA-Winter': results, 'Kenya': results}
        """
        n_domains = len(results_dict)
        fig, axes = plt.subplots(2, n_domains, figsize=(5*n_domains, 8))
        
        if n_domains == 1:
            axes = axes.reshape(2, 1)
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        
        for i, (domain_name, results) in enumerate(results_dict.items()):
            alphas = results['alpha'].flatten()
            
            # 门控权重分布
            axes[0, i].hist(alphas, bins=30, alpha=0.7, color=colors[i % len(colors)],
                           edgecolor='black')
            axes[0, i].axvline(alphas.mean(), color='red', linestyle='--',
                              label=f'Mean: {alphas.mean():.3f}')
            axes[0, i].set_title(f'{domain_name}\nGate Weight Distribution')
            axes[0, i].set_xlabel('alpha')
            axes[0, i].legend()
            
            # 环境梯度热力图 (top 10变量)
            if 'env_gradient' in results:
                grad = results['env_gradient']
                mean_grad = np.abs(grad).mean(axis=0)
                top_idx = np.argsort(mean_grad)[-10:][::-1]
                
                grad_subset = grad[:min(100, len(grad)), top_idx]  # 取前100个样本
                var_names = [BIOCLIM_SHORT_NAMES.get(j, f'V{j}') for j in top_idx]
                
                im = axes[1, i].imshow(grad_subset.T, aspect='auto', cmap='RdBu_r',
                                       vmin=-np.percentile(np.abs(grad_subset), 95),
                                       vmax=np.percentile(np.abs(grad_subset), 95))
                axes[1, i].set_yticks(range(len(var_names)))
                axes[1, i].set_yticklabels(var_names, fontsize=8)
                axes[1, i].set_xlabel('Samples')
                axes[1, i].set_title(f'{domain_name}\nTop-10 Env Gradients')
                plt.colorbar(im, ax=axes[1, i])
        
        plt.suptitle('Cross-Domain Interpretability Comparison', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved cross-domain comparison to {save_path}")
        
        plt.show()
        return fig
    
    @staticmethod
    def plot_alpha_vs_prediction_confidence(
        alphas: np.ndarray,
        predictions: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        分析门控权重与预测置信度的关系
        """
        alphas = alphas.flatten()
        
        # 计算每个样本的最大预测置信度
        max_confidence = predictions.max(axis=1)
        
        # 计算预测的熵 (不确定性)
        eps = 1e-8
        entropy = -np.sum(predictions * np.log(predictions + eps), axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # alpha vs 最大置信度
        axes[0].scatter(alphas, max_confidence, alpha=0.3, s=10)
        axes[0].set_xlabel('alpha (Gate Weight)')
        axes[0].set_ylabel('Max Prediction Confidence')
        axes[0].set_title('Gate Weight vs Prediction Confidence')
        
        # 添加趋势线
        z = np.polyfit(alphas, max_confidence, 1)
        p = np.poly1d(z)
        x_line = np.linspace(alphas.min(), alphas.max(), 100)
        axes[0].plot(x_line, p(x_line), 'r--', label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        axes[0].legend()
        
        # alpha vs 熵
        axes[1].scatter(alphas, entropy, alpha=0.3, s=10, c='orange')
        axes[1].set_xlabel('alpha (Gate Weight)')
        axes[1].set_ylabel('Prediction Entropy (Uncertainty)')
        axes[1].set_title('Gate Weight vs Prediction Uncertainty')
        
        z2 = np.polyfit(alphas, entropy, 1)
        p2 = np.poly1d(z2)
        axes[1].plot(x_line, p2(x_line), 'r--', label=f'Trend: y={z2[0]:.3f}x+{z2[1]:.3f}')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig


def save_analysis_results(
    results: Dict[str, np.ndarray],
    importance_df: pd.DataFrame,
    output_dir: str,
    prefix: str = ""
):
    """保存分析结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存numpy数组
    for key, value in results.items():
        np.save(output_dir / f"{prefix}{key}.npy", value)
    
    # 保存重要性DataFrame
    importance_df.to_csv(output_dir / f"{prefix}env_importance.csv")
    
    # 保存统计摘要
    summary = {
        'n_samples': len(results.get('alpha', [])),
        'alpha_mean': float(results['alpha'].mean()) if 'alpha' in results else None,
        'alpha_std': float(results['alpha'].std()) if 'alpha' in results else None,
        'top_env_vars': importance_df.head(5).index.tolist() if not importance_df.empty else []
    }
    
    with open(output_dir / f"{prefix}summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved analysis results to {output_dir}")


if __name__ == "__main__":
    # 示例用法
    print("Interpretability Analysis Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.analysis.interpretability import InterpretabilityAnalyzer, InterpretabilityVisualizer")
    print("  ")
    print("  analyzer = InterpretabilityAnalyzer(model)")
    print("  analyzer.analyze_dataloader(test_loader)")
    print("  results = analyzer.get_aggregated_results()")
    print("  importance = analyzer.compute_env_importance()")
    print("  ")
    print("  InterpretabilityVisualizer.plot_gate_distribution(results['alpha'])")
    print("  InterpretabilityVisualizer.plot_env_importance(importance)")
