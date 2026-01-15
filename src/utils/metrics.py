"""Utility helpers for evaluation metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
	"""Ensure input tensors are shaped as [N, C]."""

	if tensor.ndim == 2:
		return tensor
	if tensor.ndim == 1:
		return tensor.view(1, -1)
	return tensor.view(tensor.size(0), -1)


def _topk_indices(tensor: torch.Tensor, k: int) -> torch.Tensor:
	"""Return the indices of the top-k entries for each row."""

	if k <= 0:
		raise ValueError("k must be positive when requesting top-k indices")
	return torch.topk(tensor, k=k, dim=1, largest=True).indices


def compute_topk_hit_rate(
	preds: torch.Tensor,
	targets: torch.Tensor,
	ks: Iterable[int] = (1, 5, 10, 30),
) -> Dict[str, torch.Tensor]:
	"""Compute the standard top-k hit rate against the argmax label.

	The metric answers: "Is the true argmax label contained in the model's
	top-k predictions?" and returns the average hit rate per ``k``.
	"""

	preds = _ensure_2d(preds)
	targets = _ensure_2d(targets)
	if preds.size(0) == 0:
		return {}

	_, num_classes = preds.shape
	max_k = max(ks, default=0)
	if max_k <= 0:
		return {}

	topk_idx = _topk_indices(preds, k=min(max_k, num_classes))
	target_argmax = torch.argmax(targets, dim=1)

	metrics: Dict[str, torch.Tensor] = {}
	for k in ks:
		kk = min(k, num_classes)
		if kk <= 0:
			continue
		hits = (topk_idx[:, :kk] == target_argmax.unsqueeze(1)).any(dim=1).float()
		metrics[f"top{k}_acc"] = hits.mean()
	return metrics


def compute_topk_set_overlap(
	preds: torch.Tensor,
	targets: torch.Tensor,
	ks: Iterable[int] = (5, 10, 30),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
	"""Compute overlap statistics between prediction and ground-truth top-k sets.

	Returns two dictionaries containing:

	* ``top{k}_set_acc`` – averaged fraction of the predicted top-k species that
	  also appear in the ground-truth top-k set (value in [0, 1]).
	* ``top{k}_set_match_count`` – averaged number of overlapping species.
	"""

	preds = _ensure_2d(preds)
	targets = _ensure_2d(targets)
	if preds.size(0) == 0:
		return {}, {}

	_, num_classes = preds.shape
	max_k = max(ks, default=0)
	if max_k <= 0 or num_classes == 0:
		return {}, {}

	topk = min(max_k, num_classes)
	pred_topk = _topk_indices(preds, topk)
	target_topk = _topk_indices(targets, topk)

	ratios: Dict[str, torch.Tensor] = {}
	counts: Dict[str, torch.Tensor] = {}

	for k in ks:
		kk = min(k, num_classes)
		if kk <= 0:
			continue

		pred_subset = pred_topk[:, :kk]
		target_subset = target_topk[:, :kk]
		matches = (pred_subset.unsqueeze(2) == target_subset.unsqueeze(1)).any(dim=2).float().sum(dim=1)

		ratios[f"top{k}_set_acc"] = (matches / kk).mean()
		counts[f"top{k}_set_match_count"] = matches.mean()

	return ratios, counts

