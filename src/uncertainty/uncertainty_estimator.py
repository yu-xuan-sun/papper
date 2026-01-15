"""
Unified Uncertainty Estimator
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Callable

from .mc_dropout import MCDropoutEstimator
from .deep_ensemble import DeepEnsembleEstimator
from .ood_detection import OODDetector
from .metrics import compute_calibration_metrics


class UncertaintyEstimator:
    """
    Unified interface for uncertainty quantification.
    
    Supports multiple methods:
        - 'mc_dropout': Monte Carlo Dropout
        - 'ensemble': Deep Ensemble
        - 'combined': MC Dropout + Ensemble
        - 'ood': Out-of-Distribution detection
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        method: str = 'mc_dropout',
        n_samples: int = 30,
        models: Optional[List[nn.Module]] = None,
        checkpoints: Optional[List[str]] = None,
        model_factory: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):
        self.method = method
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if method == 'mc_dropout':
            if model is None:
                raise ValueError("Model required for mc_dropout method")
            self.mc_estimator = MCDropoutEstimator(model, n_samples=n_samples)
            self.ensemble_estimator = None
            self.ood_detector = None
            
        elif method == 'ensemble':
            self.mc_estimator = None
            self.ensemble_estimator = DeepEnsembleEstimator(
                models=models,
                checkpoints=checkpoints,
                model_factory=model_factory,
                device=self.device
            )
            self.ood_detector = None
            
        elif method == 'combined':
            base_model = model if model is not None else (models[0] if models else None)
            if base_model is not None:
                self.mc_estimator = MCDropoutEstimator(base_model, n_samples=n_samples)
            else:
                self.mc_estimator = None
            
            if models is not None or checkpoints is not None:
                self.ensemble_estimator = DeepEnsembleEstimator(
                    models=models,
                    checkpoints=checkpoints,
                    model_factory=model_factory,
                    device=self.device
                )
            else:
                self.ensemble_estimator = None
            self.ood_detector = None
            
        elif method == 'ood':
            if model is None:
                raise ValueError("Model required for ood method")
            self.mc_estimator = None
            self.ensemble_estimator = None
            self.ood_detector = OODDetector(model, device=self.device)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self._model = model
    
    @property
    def model(self) -> Optional[nn.Module]:
        return self._model
    
    def fit(self, dataloader, get_env_fn: Optional[Callable] = None) -> None:
        if self.ood_detector is not None:
            self.ood_detector.fit(dataloader, get_env_fn=get_env_fn)
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        images: torch.Tensor,
        env: Optional[torch.Tensor] = None,
        apply_sigmoid: bool = True,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        results = {}
        
        if self.method == 'mc_dropout' and self.mc_estimator is not None:
            mc_results = self.mc_estimator.predict_with_uncertainty(
                images, env, apply_sigmoid=apply_sigmoid
            )
            results['mean'] = mc_results['mean']
            results['std'] = mc_results['std']
            results['uncertainty'] = mc_results['epistemic_uncertainty']
            results['epistemic'] = mc_results['epistemic_uncertainty']
            results['entropy'] = mc_results['entropy']
            results['mutual_info'] = mc_results['mutual_info']
            
        elif self.method == 'ensemble' and self.ensemble_estimator is not None:
            ens_results = self.ensemble_estimator.predict_with_uncertainty(
                images, env, apply_sigmoid=apply_sigmoid
            )
            results['mean'] = ens_results['mean']
            results['std'] = ens_results['std']
            results['uncertainty'] = ens_results['epistemic_uncertainty']
            results['epistemic'] = ens_results['epistemic_uncertainty']
            results['entropy'] = ens_results['entropy']
            results['js_divergence'] = ens_results['js_divergence']
            
        elif self.method == 'combined':
            if self.mc_estimator is not None:
                mc_results = self.mc_estimator.predict_with_uncertainty(
                    images, env, apply_sigmoid=apply_sigmoid
                )
                results['mc_mean'] = mc_results['mean']
                results['mc_uncertainty'] = mc_results['epistemic_uncertainty']
            
            if self.ensemble_estimator is not None:
                ens_results = self.ensemble_estimator.predict_with_uncertainty(
                    images, env, apply_sigmoid=apply_sigmoid
                )
                results['ens_mean'] = ens_results['mean']
                results['ens_uncertainty'] = ens_results['epistemic_uncertainty']
            
            if 'mc_mean' in results and 'ens_mean' in results:
                results['mean'] = (results['mc_mean'] + results['ens_mean']) / 2
                results['uncertainty'] = (results['mc_uncertainty'] + results['ens_uncertainty']) / 2
            elif 'mc_mean' in results:
                results['mean'] = results['mc_mean']
                results['uncertainty'] = results['mc_uncertainty']
            else:
                results['mean'] = results['ens_mean']
                results['uncertainty'] = results['ens_uncertainty']
            
        elif self.method == 'ood' and self.ood_detector is not None:
            ood_results = self.ood_detector.compute_ood_scores(images, env)
            
            if self._model is not None:
                self._model.eval()
                images = images.to(self.device)
                if env is not None:
                    env = env.to(self.device)
                    logits = self._model(images, env)
                else:
                    logits = self._model(images)
                
                if apply_sigmoid:
                    results['mean'] = torch.sigmoid(logits)
                else:
                    results['mean'] = logits
            
            results['uncertainty'] = ood_results['combined_ood_score']
            results['energy'] = ood_results['energy']
            results['entropy'] = ood_results['entropy']
            
            if 'mahalanobis' in ood_results:
                results['mahalanobis'] = ood_results['mahalanobis']
        
        return results
    
    def evaluate_calibration(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        return compute_calibration_metrics(predictions, targets, uncertainties)
    
    def __repr__(self) -> str:
        return f"UncertaintyEstimator(method='{self.method}')"
