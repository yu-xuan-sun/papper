"""
Performance Boost Script - TTA and SWA evaluation
"""
import os
import sys
import glob
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

from src.utils.config_utils import load_opts


class TTAWrapper(nn.Module):
    """Test-Time Augmentation wrapper"""
    def __init__(self, model, augmentations=['none', 'hflip', 'vflip', 'hflip_vflip']):
        super().__init__()
        self.model = model
        self.augmentations = augmentations
    
    def _apply_aug(self, x, aug):
        if aug == 'none':
            return x
        elif aug == 'hflip':
            return torch.flip(x, dims=[3])  # horizontal flip
        elif aug == 'vflip':
            return torch.flip(x, dims=[2])  # vertical flip
        elif aug == 'hflip_vflip':
            return torch.flip(x, dims=[2, 3])
        return x
    
    def forward(self, x, env=None, loc=None):
        preds = []
        for aug in self.augmentations:
            x_aug = self._apply_aug(x, aug)
            with torch.no_grad():
                if env is not None:
                    pred = self.model(x_aug, env)
                else:
                    pred = self.model(x_aug)
            preds.append(pred)
        
        # Average predictions
        return torch.stack(preds).mean(dim=0)


def apply_swa(model, checkpoint_dir, top_k=5):
    """Apply Stochastic Weight Averaging"""
    print(f"\n[SWA] Searching for checkpoints in: {checkpoint_dir}")
    
    # Find best checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "best-*.ckpt")))
    if not ckpt_files:
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
    
    if len(ckpt_files) < 2:
        print(f"[SWA] Only {len(ckpt_files)} checkpoint(s) found, skipping SWA")
        return model
    
    ckpt_files = ckpt_files[:top_k]
    print(f"[SWA] Averaging {len(ckpt_files)} checkpoints:")
    for f in ckpt_files:
        print(f"  - {os.path.basename(f)}")
    
    # Average weights
    avg_state = {}
    for ckpt_path in ckpt_files:
        state = torch.load(ckpt_path, map_location='cpu')['state_dict']
        for key, val in state.items():
            clean_key = key.replace('model.', '', 1) if key.startswith('model.') else key
            if clean_key not in avg_state:
                avg_state[clean_key] = val.float() / len(ckpt_files)
            else:
                avg_state[clean_key] += val.float() / len(ckpt_files)
    
    # Load averaged weights
    model.load_state_dict(avg_state, strict=False)
    print(f"[SWA] Applied weight averaging from {len(ckpt_files)} checkpoints")
    return model


def evaluate(model, dataloader, device, metrics=['top30', 'mse']):
    """Evaluate model on dataloader"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["sat"].squeeze(1).to(device)  # Remove extra dim
                targets = batch['target'].to(device)
                env = batch.get('env')
                if env is not None:
                    env = env.to(device)
                    preds = model(images, env)
                else:
                    preds = model(images)
            else:
                images, targets = batch[0].to(device), batch[1].to(device)
                preds = model(images)
            
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    results = {}
    
    # Top-k accuracy
    _, pred_top = torch.topk(all_preds, k=30, dim=1)
    _, true_top = torch.topk(all_targets, k=30, dim=1)
    top_acc = 0
    for i in range(len(pred_top)):
        overlap = len(set(pred_top[i].tolist()) & set(true_top[i].tolist()))
        top_acc += overlap / 30
    results['top30'] = top_acc / len(all_preds)
    
    # MSE
    results['mse'] = ((all_preds - all_targets) ** 2).mean().item()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--swa_top_k', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    print("="*60)
    print("Performance Boost Script (TTA + SWA)")
    print("="*60)
    
    # Load config with defaults
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / args.config
    default_config = base_dir / "configs" / "defaults.yaml"
    
    print(f"Loading config: {args.config}")
    print(f"Default config: {default_config}")
    opts = load_opts(str(config_path), default=str(default_config), commandline_opts=None)
    opts.base_dir = str(base_dir)
    
    # Load model
    from src.trainer.trainer import EbirdTask
    
    print(f"Loading checkpoint: {args.checkpoint}")
    task = EbirdTask.load_from_checkpoint(args.checkpoint, map_location=args.device)
    model = task.model
    model = model.to(args.device)
    model.eval()
    print("Model loaded successfully")
    
    # Apply SWA
    if args.swa:
        checkpoint_dir = args.checkpoint_dir or str(Path(args.checkpoint).parent)
        model = apply_swa(model, checkpoint_dir, top_k=args.swa_top_k)
    
    # Apply TTA
    if args.tta:
        print("\n[TTA] Applying Test-Time Augmentation")
        model = TTAWrapper(model)
        print("[TTA] Enabled with augmentations: none, hflip, vflip, hflip_vflip")
    
    model = model.to(args.device)
    
    # Evaluation
    if args.eval:
        print("\n" + "="*60)
        print("Evaluating...")
        print("="*60)
        
        from src.trainer.trainer import EbirdDataModule
        datamodule = EbirdDataModule(opts)
        datamodule.setup('test')
        test_loader = datamodule.test_dataloader()
        
        print(f"Test set size: {len(test_loader.dataset)}")
        
        results = evaluate(model, test_loader, args.device)
        
        print("\nResults:")
        print("-"*40)
        for key, val in results.items():
            print(f"  {key}: {val:.4f}")
        print("-"*40)
        
        # Compare with baseline
        baseline_ckpt = args.checkpoint
        if args.tta or args.swa:
            print("\nNote: Compare these results with baseline evaluation")
            print("(without --tta and --swa flags) to see improvement")
    else:
        print("\nModel ready. Use --eval to run evaluation.")
        print("Example:")
        print(f"  python {sys.argv[0]} --checkpoint {args.checkpoint} --config {args.config} --tta --eval")

if __name__ == "__main__":
    main()
