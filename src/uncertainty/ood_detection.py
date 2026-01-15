"""
Out-of-Distribution (OOD) Detection Module
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
import numpy as np
from tqdm import tqdm
import warnings


class OODDetector:
    """Out-of-Distribution Detector for species distribution models."""
    
    def __init__(
        self,
        model: nn.Module,
        method: str = 'energy',
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.method = method
        self.temperature = temperature
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._class_means = None
        self._precision_matrix = None
        self._is_fitted = False
        
        self.model.to(self.device)
        self.model.eval()
    
    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        get_env_fn: Optional[Callable] = None,
        show_progress: bool = True
    ) -> None:
        """Fit the OOD detector on training data (for Mahalanobis distance)."""
        print("Fitting OOD detector on training data...")
        
        all_features = []
        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                images, env = self._parse_batch(batch, get_env_fn)
                images = images.to(self.device)
                if env is not None:
                    env = env.to(self.device)
                
                features = self._extract_features(images, env)
                all_features.append(features.cpu())
        
        all_features = torch.cat(all_features, dim=0).numpy()
        self._fit_gaussian(all_features)
        self._is_fitted = True
        print(f"OOD detector fitted on {len(all_features)} samples")
    
    def _extract_features(self, images: torch.Tensor, env: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hasattr(self.model, 'forward_visual_features'):
            visual_feat = self.model.forward_visual_features(images)
            if env is not None and hasattr(self.model, 'fusion') and self.model.fusion is not None:
                try:
                    fused_feat = self.model.fusion(visual_feat, env)
                    return fused_feat
                except:
                    return visual_feat
            return visual_feat
        else:
            if env is not None:
                return self.model(images, env)
            else:
                return self.model(images)
    
    def _fit_gaussian(self, features: np.ndarray) -> None:
        self._class_means = np.mean(features, axis=0, keepdims=True)
        centered_features = features - self._class_means
        
        try:
            from sklearn.covariance import LedoitWolf
            cov_estimator = LedoitWolf().fit(centered_features)
            self._precision_matrix = cov_estimator.precision_
        except Exception as e:
            warnings.warn(f"LedoitWolf failed: {e}. Using pseudo-inverse.")
            cov = np.cov(centered_features.T)
            self._precision_matrix = np.linalg.pinv(cov)
        
        self._class_means = torch.from_numpy(self._class_means).float().to(self.device)
        self._precision_matrix = torch.from_numpy(self._precision_matrix).float().to(self.device)
    
    def _parse_batch(self, batch: dict, get_env_fn: Optional[Callable] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            images = batch.get('image') or batch.get('sat') or batch.get('images')
            if get_env_fn is not None:
                env = get_env_fn(batch)
            elif 'env' in batch:
                env = batch['env']
            elif 'bioclim' in batch and 'ped' in batch:
                bioclim = batch['bioclim']
                ped = batch['ped']
                if bioclim is not None and ped is not None:
                    env = torch.cat([bioclim, ped], dim=1)
                else:
                    env = bioclim if bioclim is not None else ped
            else:
                env = None
        else:
            images = batch[0]
            env = batch[1] if len(batch) > 1 else None
        return images, env
    
    @torch.no_grad()
    def compute_ood_scores(
        self,
        images: torch.Tensor,
        env: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute OOD scores for a batch of samples."""
        self.model.eval()
        images = images.to(self.device)
        if env is not None:
            env = env.to(self.device)
        
        if env is not None:
            logits = self.model(images, env)
        else:
            logits = self.model(images)
        
        results = {}
        
        # Maximum Softmax Probability (MSP)
        probs = torch.sigmoid(logits)
        msp_score = probs.max(dim=1).values
        results['msp'] = msp_score
        results['msp_ood_score'] = 1 - msp_score
        
        # Energy-based score
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        results['energy'] = energy
        results['energy_ood_score'] = energy
        
        # Entropy-based score
        probs_clamped = probs.clamp(1e-8, 1 - 1e-8)
        entropy = -(probs_clamped * torch.log(probs_clamped) + 
                   (1 - probs_clamped) * torch.log(1 - probs_clamped)).mean(dim=1)
        results['entropy'] = entropy
        results['entropy_ood_score'] = entropy
        
        # Mahalanobis distance (if fitted)
        if self._is_fitted and self._precision_matrix is not None:
            features = self._extract_features(images, env)
            mahal_dist = self._compute_mahalanobis_distance(features)
            results['mahalanobis'] = mahal_dist
            results['mahalanobis_ood_score'] = mahal_dist
        
        # Combined score
        if 'mahalanobis' in results:
            combined = (
                0.3 * results['msp_ood_score'] +
                0.3 * torch.sigmoid(results['energy'] / 10) +
                0.4 * torch.sigmoid(results['mahalanobis'] / 100)
            )
        else:
            combined = (
                0.5 * results['msp_ood_score'] +
                0.5 * torch.sigmoid(results['energy'] / 10)
            )
        results['combined_ood_score'] = combined
        
        return results
    
    def _compute_mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        if self._class_means is None or self._precision_matrix is None:
            raise ValueError("OOD detector not fitted. Call fit() first.")
        centered = features - self._class_means
        left = torch.matmul(centered, self._precision_matrix)
        mahal_dist = (left * centered).sum(dim=1)
        return mahal_dist.sqrt()
    
    @torch.no_grad()
    def detect_ood_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        threshold: Optional[float] = None,
        percentile: float = 95.0,
        get_env_fn: Optional[Callable] = None,
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Detect OOD samples in a dataloader."""
        all_scores = {'msp': [], 'energy': [], 'entropy': [], 'combined_ood_score': []}
        if self._is_fitted:
            all_scores['mahalanobis'] = []
        
        iterator = tqdm(dataloader, desc="OOD Detection") if show_progress else dataloader
        
        for batch in iterator:
            images, env = self._parse_batch(batch, get_env_fn)
            scores = self.compute_ood_scores(images, env)
            for key in all_scores:
                if key in scores:
                    all_scores[key].append(scores[key].cpu())
        
        results = {}
        for key, values in all_scores.items():
            if values:
                results[key] = torch.cat(values, dim=0)
        
        ood_scores = results['combined_ood_score']
        if threshold is None:
            threshold = torch.quantile(ood_scores, percentile / 100.0).item()
        
        results['is_ood'] = ood_scores > threshold
        results['ood_threshold'] = threshold
        results['n_ood'] = results['is_ood'].sum().item()
        results['ood_ratio'] = results['n_ood'] / len(ood_scores)
        
        print(f"OOD Detection: {results['n_ood']}/{len(ood_scores)} samples detected as OOD ({results['ood_ratio']*100:.1f}%)")
        return results
