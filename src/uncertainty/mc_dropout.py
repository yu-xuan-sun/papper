"""
MC Dropout for Epistemic Uncertainty Estimation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import tqdm


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout layers for MC Dropout inference."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def disable_mc_dropout(model: nn.Module) -> None:
    """Disable dropout layers (return to normal eval mode)."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.eval()


class MCDropoutEstimator:
    """Monte Carlo Dropout Estimator for uncertainty quantification."""
    
    def __init__(self, model: nn.Module, n_samples: int = 30, dropout_rate: Optional[float] = None):
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self._original_dropout_rates = {}
        
    def _set_dropout_rate(self, rate: float) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                self._original_dropout_rates[name] = module.p
                module.p = rate
    
    def _restore_dropout_rate(self) -> None:
        for name, module in self.model.named_modules():
            if name in self._original_dropout_rates:
                module.p = self._original_dropout_rates[name]
        self._original_dropout_rates.clear()
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        images: torch.Tensor,
        env: Optional[torch.Tensor] = None,
        apply_sigmoid: bool = True,
        return_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Perform MC Dropout inference with uncertainty estimation."""
        self.model.eval()
        enable_mc_dropout(self.model)
        
        if self.dropout_rate is not None:
            self._set_dropout_rate(self.dropout_rate)
        
        device = next(self.model.parameters()).device
        images = images.to(device)
        if env is not None:
            env = env.to(device)
        
        samples = []
        
        try:
            for _ in range(self.n_samples):
                if env is not None:
                    logits = self.model(images, env)
                else:
                    logits = self.model(images)
                
                if apply_sigmoid:
                    probs = torch.sigmoid(logits)
                else:
                    probs = logits
                    
                samples.append(probs)
            
            samples_tensor = torch.stack(samples, dim=0)
            mean_pred = samples_tensor.mean(dim=0)
            std_pred = samples_tensor.std(dim=0)
            
            eps = 1e-8
            mean_clamped = mean_pred.clamp(eps, 1 - eps)
            predictive_entropy = -(
                mean_clamped * torch.log(mean_clamped) + 
                (1 - mean_clamped) * torch.log(1 - mean_clamped)
            ).mean(dim=1)
            
            samples_clamped = samples_tensor.clamp(eps, 1 - eps)
            sample_entropies = -(
                samples_clamped * torch.log(samples_clamped) +
                (1 - samples_clamped) * torch.log(1 - samples_clamped)
            ).mean(dim=2)
            expected_entropy = sample_entropies.mean(dim=0)
            mutual_info = predictive_entropy - expected_entropy
            
            results = {
                'mean': mean_pred,
                'std': std_pred,
                'entropy': predictive_entropy,
                'mutual_info': mutual_info,
                'expected_entropy': expected_entropy,
                'epistemic_uncertainty': std_pred.mean(dim=1),
                'total_uncertainty': predictive_entropy,
            }
            
            if return_samples:
                results['samples'] = samples_tensor
                
            return results
            
        finally:
            if self.dropout_rate is not None:
                self._restore_dropout_rate()
            disable_mc_dropout(self.model)
