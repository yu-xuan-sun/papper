"""
Deep Ensemble for Uncertainty Estimation
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable
from tqdm import tqdm


class DeepEnsembleEstimator:
    """Deep Ensemble Estimator for uncertainty quantification."""
    
    def __init__(
        self,
        models: Optional[List[nn.Module]] = None,
        checkpoints: Optional[List[str]] = None,
        model_factory: Optional[Callable] = None,
        weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if models is not None:
            self.models = models
        elif checkpoints is not None and model_factory is not None:
            self.models = self._load_models_from_checkpoints(checkpoints, model_factory)
        else:
            self.models = []
        
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        n_models = len(self.models)
        if weights is not None:
            assert len(weights) == n_models
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / n_models] * n_models if n_models > 0 else []
    
    def _load_models_from_checkpoints(self, checkpoints: List[str], model_factory: Callable) -> List[nn.Module]:
        models = []
        for ckpt_path in checkpoints:
            print(f"Loading model from {ckpt_path}")
            model = model_factory()
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            models.append(model)
        print(f"Loaded {len(models)} models for ensemble")
        return models
    
    @property
    def n_models(self) -> int:
        return len(self.models)
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        images: torch.Tensor,
        env: Optional[torch.Tensor] = None,
        apply_sigmoid: bool = True,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models first.")
        
        images = images.to(self.device)
        if env is not None:
            env = env.to(self.device)
        
        predictions = []
        for model in self.models:
            model.eval()
            if env is not None:
                logits = model(images, env)
            else:
                logits = model(images)
            if apply_sigmoid:
                probs = torch.sigmoid(logits)
            else:
                probs = logits
            predictions.append(probs)
        
        predictions_tensor = torch.stack(predictions, dim=0)
        weights_tensor = torch.tensor(self.weights, device=self.device).view(-1, 1, 1)
        mean_pred = (predictions_tensor * weights_tensor).sum(dim=0)
        std_pred = predictions_tensor.std(dim=0)
        
        eps = 1e-8
        mean_clamped = mean_pred.clamp(eps, 1 - eps)
        predictive_entropy = -(
            mean_clamped * torch.log(mean_clamped) +
            (1 - mean_clamped) * torch.log(1 - mean_clamped)
        ).mean(dim=1)
        
        # Jensen-Shannon Divergence
        n_models = predictions_tensor.shape[0]
        mean_pred_for_js = predictions_tensor.mean(dim=0).clamp(eps, 1 - eps)
        H_avg = -(mean_pred_for_js * torch.log(mean_pred_for_js) +
                  (1 - mean_pred_for_js) * torch.log(1 - mean_pred_for_js)).mean(dim=1)
        preds_clamped = predictions_tensor.clamp(eps, 1 - eps)
        individual_entropies = -(preds_clamped * torch.log(preds_clamped) +
                                 (1 - preds_clamped) * torch.log(1 - preds_clamped)).mean(dim=2)
        avg_H = individual_entropies.mean(dim=0)
        js_divergence = H_avg - avg_H
        
        results = {
            'mean': mean_pred,
            'std': std_pred,
            'variance': predictions_tensor.var(dim=0),
            'entropy': predictive_entropy,
            'js_divergence': js_divergence,
            'epistemic_uncertainty': std_pred.mean(dim=1),
            'total_uncertainty': predictive_entropy,
        }
        
        if return_individual:
            results['individual_predictions'] = predictions_tensor
        
        return results
