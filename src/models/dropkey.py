"""
DropKey: A novel regularization technique for Vision Transformers
Replaces standard Dropout on attention weights with dropping keys in the attention mechanism.
"""

import torch
import torch.nn as nn


class DropKey(nn.Module):
    """
    DropKey regularization for Vision Transformer attention layers.
    
    Unlike standard dropout which drops attention weights after softmax,
    DropKey drops keys before the attention computation.
    
    Args:
        p: Probability of dropping a key dimension (default: 0.1)
        inplace: If True, do operation in-place (default: False)
    """
    
    def __init__(self, p: float = 0.1, inplace: bool = False):
        super(DropKey, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DropKey to input tensor.
        
        Args:
            x: Input tensor, typically Keys in attention mechanism
               Shape: [B, num_heads, N, head_dim] or [B, N, C]
        
        Returns:
            Tensor with same shape as input
        """
        if not self.training or self.p == 0:
            return x
        
        return self.dropkey_fn(x)
    
    def dropkey_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropkey to last dimension (feature dimension)"""
        if self.p == 1:
            return x.mul(0) if self.inplace else torch.zeros_like(x)
        
        # Drop along last dimension (feature/head_dim)
        mask_shape = [1] * x.ndim
        mask_shape[-1] = x.shape[-1]
        
        keep_prob = 1 - self.p
        mask = torch.empty(mask_shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        mask = mask / keep_prob  # Inverted dropout scaling
        
        if self.inplace:
            return x.mul_(mask)
        else:
            return x * mask
    
    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'


# Test
if __name__ == "__main__":
    print("Testing DropKey...")
    dk = DropKey(p=0.1)
    dk.train()
    x = torch.randn(2, 8, 10, 64)
    y = dk(x)
    print(f"✓ Shape: {x.shape} -> {y.shape}")
    
    dk.eval()
    y_eval = dk(x)
    assert torch.allclose(y_eval, x)
    print(f"✓ Eval mode works")
    print("All tests passed! ✅")
