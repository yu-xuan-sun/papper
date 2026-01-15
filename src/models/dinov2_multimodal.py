# src/models/dinov2_multimodal.py
"""
Dinov2-based multimodal model:
- Image encoder: DINOv2 ViT-S (via timm if available, otherwise torch.hub fallback)
- Env encoder: ResNet (timm)
- Prompt tokens: pooled-level learnable prompt (optional)
- Prototype bank: one prototype per species
- Hurdle head: presence branch (sigmoid) + conditional abundance branch (sigmoid)
Returns dict with presence_prob, conditional_abundance, pred, proto_sim
"""
import math
import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except Exception:  # pragma: no cover - timm might not be installed
    timm = None
    _HAS_TIMM = False

import torch.hub


def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self._drop_prob = float(drop_prob)

    @property
    def drop_prob(self) -> float:
        return self._drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self._drop_prob, self.training)


class PromptTokens(nn.Module):
    def __init__(self, n_prompts: int, embed_dim: int):
        super().__init__()
        self.prompts = nn.Parameter(torch.zeros(n_prompts, embed_dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.prompts.unsqueeze(0).expand(batch_size, -1, -1)


class AdapterBottleneck(nn.Module):
    def __init__(self, embed_dim: int, reduction_dim: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(embed_dim, reduction_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(reduction_dim, embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.up(self.act(self.down(x))))


class PromptAdapterBlock(nn.Module):
    def __init__(self,
                 base_block: nn.Module,
                 embed_dim: int,
                 n_prompts: int = 0,
                 prompt_dropout: float = 0.0,
                 adapter_reduction: Optional[int] = None,
                 adapter_dropout: float = 0.0,
                 train_layer_norm: bool = True):
        super().__init__()
        self.base_block = base_block
        self.embed_dim = embed_dim
        self.n_prompts = n_prompts
        self.prompt_dropout = nn.Dropout(prompt_dropout) if prompt_dropout > 0 else nn.Identity()
        if n_prompts > 0:
            self.prompt_params = nn.Parameter(torch.zeros(1, n_prompts, embed_dim))
            nn.init.trunc_normal_(self.prompt_params, std=0.02)
        else:
            self.register_buffer("prompt_params", None, persistent=False)

        drop_module = getattr(base_block, "drop_path", None)
        if isinstance(drop_module, nn.Module):
            self.block_drop_path = drop_module
        elif isinstance(drop_module, (int, float)) and float(drop_module) > 0:
            self.block_drop_path = DropPath(float(drop_module))
        else:
            self.block_drop_path = nn.Identity()

        self.adapter_enabled = adapter_reduction is not None and adapter_reduction > 0
        if self.adapter_enabled:
            self.pre_attn_norm = nn.LayerNorm(embed_dim)
            self.post_mlp_norm = nn.LayerNorm(embed_dim)
            self.attn_adapter = AdapterBottleneck(embed_dim, adapter_reduction, adapter_dropout)
            self.mlp_adapter = AdapterBottleneck(embed_dim, adapter_reduction, adapter_dropout)
            drop_prob_source = getattr(self.block_drop_path, "drop_prob", None)
            if drop_prob_source is not None:
                drop_prob = float(drop_prob_source)
            elif isinstance(drop_module, (int, float)):
                drop_prob = float(drop_module)
            else:
                drop_prob = 0.0
            self.attn_drop_path = DropPath(drop_prob)
            self.mlp_drop_path = DropPath(drop_prob)
        else:
            self.pre_attn_norm = None
            self.post_mlp_norm = None
            self.attn_adapter = None
            self.mlp_adapter = None
            self.attn_drop_path = None
            self.mlp_drop_path = None

        self.condition_cache: Optional[torch.Tensor] = None

        if not train_layer_norm:
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    for param in module.parameters():
                        param.requires_grad = False

    def set_condition(self, condition: Optional[torch.Tensor]) -> None:
        self.condition_cache = condition

    def _prepare_prompts(self, batch: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.n_prompts <= 0 or self.prompt_params is None:
            return None
        prompts = self.prompt_params.expand(batch, -1, -1)
        if self.condition_cache is not None:
            cond = self.condition_cache
            if cond.shape[0] != batch:
                cond = cond.expand(batch, -1)
            prompts = prompts + cond.unsqueeze(1)
        return self.prompt_dropout(prompts).to(device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch = x.shape[0]
        prompts = self._prepare_prompts(batch, device)

        if prompts is not None:
            cls_tokens, tokens = x[:, :1], x[:, 1:]
            x = torch.cat([cls_tokens, prompts, tokens], dim=1)

        residual = x
        x = self.base_block.norm1(x)
        x = residual + self.block_drop_path(self.base_block.attn(x))
        if self.adapter_enabled and self.attn_adapter is not None:
            adapter_input = self.pre_attn_norm(x) if self.pre_attn_norm is not None else x
            x = x + self.attn_drop_path(self.attn_adapter(adapter_input))

        residual = x
        x = self.base_block.norm2(x)
        x = residual + self.block_drop_path(self.base_block.mlp(x))
        if self.adapter_enabled and self.mlp_adapter is not None:
            adapter_input = self.post_mlp_norm(x) if self.post_mlp_norm is not None else x
            x = x + self.mlp_drop_path(self.mlp_adapter(adapter_input))

        if prompts is not None:
            cls_tokens = x[:, :1]
            remainder = x[:, 1 + self.n_prompts:]
            x = torch.cat([cls_tokens, remainder], dim=1)

        return x


class PromptAdapterVisionTransformer(nn.Module):
    def __init__(self,
                 base_vit: nn.Module,
                 n_prompts: int,
                 adapter_reduction: Optional[int],
                 prompt_dropout: float,
                 adapter_dropout: float,
                 train_layer_norm: bool = True):
        super().__init__()
        self.embed_dim = getattr(base_vit, "embed_dim", None)
        if self.embed_dim is None:
            raise ValueError("Base ViT must expose embed_dim to be wrapped by PromptAdapterVisionTransformer.")

        self.patch_embed = base_vit.patch_embed
        self.cls_token = base_vit.cls_token
        self.pos_embed = base_vit.pos_embed
        self.pos_drop = base_vit.pos_drop
        self.norm = base_vit.norm
        self.fc_norm = getattr(base_vit, "fc_norm", None)
        self.pre_logits = getattr(base_vit, "pre_logits", None)
        self.global_pool = getattr(base_vit, "global_pool", None)
        self.num_prefix_tokens = getattr(base_vit, "num_prefix_tokens", 1)
        self.no_embed_class = getattr(base_vit, "no_embed_class", False)

        wrapped_blocks = []
        for block in base_vit.blocks:
            wrapped_blocks.append(
                PromptAdapterBlock(
                    base_block=block,
                    embed_dim=self.embed_dim,
                    n_prompts=n_prompts,
                    prompt_dropout=prompt_dropout,
                    adapter_reduction=adapter_reduction,
                    adapter_dropout=adapter_dropout,
                    train_layer_norm=train_layer_norm,
                )
            )
        self.blocks = nn.ModuleList(wrapped_blocks)

    def set_condition(self, condition: Optional[torch.Tensor]) -> None:
        for block in self.blocks:
            block.set_condition(condition)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.no_embed_class:
            pos = self.pos_embed[:, :x.shape[1], :]
            x = x + pos
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            if self.pos_embed.shape[1] == x.shape[1]:
                x = x + self.pos_embed
            else:
                pos = self.pos_embed[:, :x.shape[1], :]
                x = x + pos

        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.global_pool == 'avg':
            feat = x[:, self.num_prefix_tokens:, :].mean(dim=1)
            if self.fc_norm is not None:
                feat = self.fc_norm(feat)
            return feat

        cls_out = x[:, 0]
        if self.fc_norm is not None:
            cls_out = self.fc_norm(cls_out)
        if self.pre_logits is not None:
            cls_out = self.pre_logits(cls_out)
        return cls_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class PrototypeBank(nn.Module):
    def __init__(self, num_species: int, emb_dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_species, emb_dim) * 0.01)

    def forward(self) -> torch.Tensor:
        return self.prototypes


class HurdleHead(nn.Module):
    def __init__(self, emb_dim: int, num_species: int, hidden_dim: int = 512):
        super().__init__()
        self.pres_fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_species)
        )
        self.cond_fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_species),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_logits = self.pres_fc(x)
        a_vals = self.cond_fc(x)
        p_probs = torch.sigmoid(p_logits)
        return p_probs, a_vals


class Dinov2Multimodal(nn.Module):
    def __init__(self,
                 num_species: int,
                 dinov2_name: str = "dinov2_vits14",
                 dinov2_pretrained: bool = True,
                 n_prompts: int = 8,
                 env_backbone: str = "resnet18",
                 proj_dim: int = 768,
                 freeze_dinov2: bool = True,
                 device: str = "cuda",
                 env_input_dim: int = 27,
                 use_grad_ckpt: bool = False,
                 use_channels_last: bool = False,
                 use_block_prompts: bool = False,
                 block_prompt_length: int = 2,
                 block_prompt_dropout: float = 0.0,
                 adapter_reduction: Optional[int] = None,
                 adapter_dropout: float = 0.0,
                 condition_prompts_on_env: bool = False,
                 train_block_layer_norm: bool = True,
                 use_global_prompt: bool = True,
                 train_prompts_when_frozen: bool = True):
        super().__init__()
        self.device = device
        self.use_channels_last = use_channels_last
        self.use_global_prompt = use_global_prompt
        self.use_block_prompts = use_block_prompts
        self.block_prompt_length = block_prompt_length if use_block_prompts else 0
        self.adapter_reduction = adapter_reduction if adapter_reduction and adapter_reduction > 0 else None
        self.block_prompt_dropout = block_prompt_dropout
        self.adapter_dropout = adapter_dropout
        self.condition_prompts_on_env = condition_prompts_on_env and self.block_prompt_length > 0
        self.train_block_layer_norm = train_block_layer_norm
        self.train_prompts_when_frozen = train_prompts_when_frozen

        self.img_encoder, img_emb_dim = self._load_dinov2(dinov2_name, pretrained=dinov2_pretrained)
        requires_wrapping = (self.block_prompt_length > 0) or (self.adapter_reduction is not None)
        if requires_wrapping and hasattr(self.img_encoder, "blocks"):
            self.img_encoder = PromptAdapterVisionTransformer(
                base_vit=self.img_encoder,
                n_prompts=self.block_prompt_length,
                adapter_reduction=self.adapter_reduction,
                prompt_dropout=self.block_prompt_dropout,
                adapter_dropout=self.adapter_dropout,
                train_layer_norm=self.train_block_layer_norm,
            )
        elif requires_wrapping:
            warnings.warn("Block-level prompts/adapters requested but backbone lacks transformer blocks; disabling these features.")
            self.use_block_prompts = False
            self.block_prompt_length = 0
            self.adapter_reduction = None
            self.condition_prompts_on_env = False

        if freeze_dinov2:
            for _, param in self.img_encoder.named_parameters():
                param.requires_grad = False

        if use_grad_ckpt and hasattr(self.img_encoder, "set_grad_checkpointing"):
            try:
                self.img_encoder.set_grad_checkpointing(enable=True)
            except Exception:
                pass

        self.n_prompts = n_prompts
        if self.use_global_prompt and n_prompts > 0:
            self.prompt = PromptTokens(n_prompts=n_prompts, embed_dim=img_emb_dim)
        else:
            self.prompt = None

        self._backbone_frozen = False

        self.img_proj = nn.Sequential(
            nn.Linear(img_emb_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim)
        )

        if _HAS_TIMM:
            try:
                self.env_encoder = timm.create_model(env_backbone, pretrained=False, num_classes=0, global_pool='avg')
                env_emb_dim = getattr(self.env_encoder, "num_features", None)
                if env_emb_dim is None:
                    env_emb_dim = getattr(self.env_encoder, "embed_dim", 256)
            except Exception:
                warnings.warn(f"timm failed to instantiate env_backbone={env_backbone}. Falling back to small MLP.")
                self.env_encoder = None
                env_emb_dim = 256
        else:
            warnings.warn("timm not available — using small MLP for env features.")
            self.env_encoder = None
            env_emb_dim = 256

        self.env_mlp = nn.Sequential(
            nn.Linear(env_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, env_emb_dim),
        )

        self.env_proj = nn.Sequential(
            nn.Linear(env_emb_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim)
        )

        if self.condition_prompts_on_env:
            self.env_prompt_mapper = nn.Sequential(
                nn.Linear(env_emb_dim, img_emb_dim),
                nn.Tanh()
            )
        else:
            self.env_prompt_mapper = None

        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim)
        )

        self.prototype_bank = PrototypeBank(num_species=num_species, emb_dim=proj_dim)
        self.hurdle_head = HurdleHead(emb_dim=proj_dim, num_species=num_species,
                                      hidden_dim=max(proj_dim // 2, 128))
        self.prototype_weight = nn.Parameter(torch.tensor(0.1))
        self.set_backbone_trainable(not freeze_dinov2,
                                    train_prompts=self.train_prompts_when_frozen if freeze_dinov2 else True)

    def _try_local_checkpoint(self, model: nn.Module, ckpt_paths):
        for path in ckpt_paths:
            if path is None:
                continue
            if os.path.exists(path):
                try:
                    state = torch.load(path, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                        state = state["state_dict"]
                    state = self._prepare_state_dict_for_vit(model, state)
                    model.load_state_dict(state, strict=False)
                    warnings.warn(f"Loaded local checkpoint for DINOv2 from: {path} (strict=False)")
                    return True
                except Exception as exc:
                    warnings.warn(f"Failed to load local checkpoint {path}: {exc}")
        return False

    def _prepare_state_dict_for_vit(self, model: nn.Module, state: dict) -> dict:
        cleaned = {}
        for key, value in state.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("model." ):
                new_key = new_key[len("model."):]
            cleaned[new_key] = value

        target_state = model.state_dict()
        filtered = {k: v for k, v in cleaned.items() if k in target_state}

        if "pos_embed" in filtered and hasattr(model, "pos_embed"):
            source = filtered["pos_embed"]
            target = model.pos_embed
            if source.shape != target.shape:
                filtered["pos_embed"] = self._resize_pos_embed_tensor(
                    source,
                    target,
                    getattr(model, "num_prefix_tokens", 1)
                )

        return filtered

    def _resize_pos_embed_to_target(self,
                                    posemb: torch.Tensor,
                                    target_posemb: torch.Tensor,
                                    num_prefix_tokens: int) -> torch.Tensor:
        if posemb.ndim != 2:
            posemb = posemb.view(1, posemb.shape[0], -1)
        if target_posemb.ndim != 2:
            target_posemb = target_posemb.view(1, target_posemb.shape[1], -1)

        posemb = posemb.float()
        target_posemb = target_posemb.float()

        prefix_src = posemb[:, :num_prefix_tokens]
        prefix_tgt = target_posemb[:, :num_prefix_tokens]
        grid_src = posemb[:, num_prefix_tokens:]
        grid_tgt = target_posemb[:, num_prefix_tokens:]

        dim = posemb.shape[-1]
        gs_src = int(math.sqrt(grid_src.shape[1]))
        gs_tgt = int(math.sqrt(grid_tgt.shape[1]))

        if gs_src * gs_src != grid_src.shape[1] or gs_tgt * gs_tgt != grid_tgt.shape[1]:
            return target_posemb

        grid_src = grid_src.reshape(1, gs_src, gs_src, dim).permute(0, 3, 1, 2)
        grid_src = F.interpolate(grid_src, size=(gs_tgt, gs_tgt), mode="bicubic", align_corners=False)
        grid_src = grid_src.permute(0, 2, 3, 1).reshape(1, gs_tgt * gs_tgt, dim)

        prefix = prefix_src
        if prefix.shape != prefix_tgt.shape:
            prefix = prefix_tgt

        posemb_resized = torch.cat([prefix, grid_src], dim=1)
        return posemb_resized.to(target_posemb.dtype)

    def _resize_pos_embed_tensor(self,
                                 posemb: torch.Tensor,
                                 target_posemb: torch.Tensor,
                                 num_prefix_tokens: int) -> torch.Tensor:
        try:
            from timm.models.vision_transformer import resize_pos_embed as timm_resize_pos_embed
            return timm_resize_pos_embed(posemb, target_posemb, num_prefix_tokens)
        except Exception:
            return self._resize_pos_embed_to_target(posemb, target_posemb, num_prefix_tokens)

    def _map_name_to_timm(self, name: str) -> str:
        mapping = {
            "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
            "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
            "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
            "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
            "vit_small_patch14_224.dinov2": "vit_small_patch14_dinov2.lvd142m",
        }
        return mapping.get(name, name)

    def _load_dinov2(self, name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_candidates = [
            os.path.join(ckpt_dir, f"{name}_pretrain.pth"),
            os.path.join(ckpt_dir, f"{name}_pretrain.pt"),
            os.path.join(ckpt_dir, f"{name}.pth"),
            os.path.join(ckpt_dir, f"{name}.pt"),
            os.path.join(ckpt_dir, "dinov2_vits14_pretrain.pth"),
            os.path.join(ckpt_dir, "dinov2_vits14_pretrain.pt"),
            os.path.join(ckpt_dir, "dinov2_vits14.pth"),
        ]

        timm_name = self._map_name_to_timm(name)
        if _HAS_TIMM:
            try:
                if pretrained:
                    model = timm.create_model(timm_name, pretrained=True, num_classes=0, global_pool='avg', img_size=224)
                else:
                    model = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool='avg', img_size=224)
                feat_dim = getattr(model, "num_features", getattr(model, "embed_dim", None))
                if feat_dim is None:
                    model.eval()
                    with torch.no_grad():
                        dummy = torch.zeros(1, 3, 224, 224)
                        out = model.forward_features(dummy) if hasattr(model, "forward_features") else model(dummy)
                        feat_dim = out.shape[1] if out.ndim == 2 else out.shape[1]
                return model, int(feat_dim)
            except Exception as exc:
                warnings.warn(f"timm create_model failed for {timm_name}: {exc}. Trying alternatives.")

        if _HAS_TIMM:
            try:
                model = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool='avg', img_size=224)
                if pretrained and self._try_local_checkpoint(model, ckpt_candidates):
                    feat_dim = getattr(model, "num_features", getattr(model, "embed_dim", None))
                    if feat_dim is None:
                        model.eval()
                        with torch.no_grad():
                            dummy = torch.zeros(1, 3, 224, 224)
                            out = model.forward_features(dummy) if hasattr(model, "forward_features") else model(dummy)
                            feat_dim = out.shape[1] if out.ndim == 2 else out.shape[1]
                    return model, int(feat_dim)
            except Exception as exc:
                warnings.warn(f"timm fallback instantiate failed for {timm_name}: {exc}")

        if pretrained:
            try:
                hub_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)

                class HubWrap(nn.Module):
                    def __init__(self, module: nn.Module):
                        super().__init__()
                        self.module = module

                    def forward(self, inp: torch.Tensor) -> torch.Tensor:
                        return self.module(inp)

                    def forward_features(self, inp: torch.Tensor) -> torch.Tensor:
                        out = self.module(inp)
                        if out.ndim > 2:
                            out = out.mean(dim=1)
                        return out

                wrapped = HubWrap(hub_model)
                wrapped.eval()
                with torch.no_grad():
                    feat = wrapped.forward_features(torch.zeros(1, 3, 224, 224))
                return wrapped, int(feat.shape[1])
            except Exception as exc:
                warnings.warn(f"torch.hub load failed for {name}: {exc}")

        if _HAS_TIMM:
            try:
                model = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool='avg', img_size=224)
                feat_dim = getattr(model, "num_features", getattr(model, "embed_dim", None))
                if feat_dim is None:
                    model.eval()
                    with torch.no_grad():
                        dummy = torch.zeros(1, 3, 224, 224)
                        out = model.forward_features(dummy) if hasattr(model, "forward_features") else model(dummy)
                        feat_dim = out.shape[1] if out.ndim == 2 else out.shape[1]
                warnings.warn("All pretrained DINOv2 loading attempts failed — using timm model with pretrained=False (random init).")
                return model, int(feat_dim)
            except Exception as exc:
                warnings.warn(f"Final timm fallback failed for {timm_name}: {exc}")

        raise RuntimeError(
            "Failed to load DINOv2 model. Provide a local checkpoint in ./checkpoints/ or set dinov2_pretrained=False."
        )

    def _encode_environment(self,
                             env: Optional[torch.Tensor],
                             batch_size: int,
                             device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if env is None:
            zero_feat = torch.zeros(batch_size, self.env_proj[0].in_features, device=device)
            return zero_feat, self.env_proj(zero_feat)

        if isinstance(env, dict):
            pieces = []
            if env.get("bioclim") is not None:
                pieces.append(env["bioclim"])
            if env.get("ped") is not None:
                pieces.append(env["ped"])
            if pieces:
                env = torch.cat(pieces, dim=1)
            else:
                env = None

            if env is None:
                zero_feat = torch.zeros(batch_size, self.env_proj[0].in_features, device=device)
                return zero_feat, self.env_proj(zero_feat)

        env = env.to(device)
        if env.ndim == 4 and self.env_encoder is not None:
            if hasattr(self.env_encoder, "forward_features"):
                env_feat = self.env_encoder.forward_features(env)
            else:
                env_feat = self.env_encoder(env)
            if env_feat.ndim > 2:
                env_feat = env_feat.mean(dim=1)
        elif env.ndim == 2 and getattr(self, "env_mlp", None) is not None:
            env_feat = self.env_mlp(env)
        elif env.ndim == 2:
            env_feat = env
        else:
            env_feat = env.view(batch_size, -1)

        env_feat = env_feat.to(device)
        env_emb = self.env_proj(env_feat)
        return env_feat, env_emb

    def _encode_image(self,
                      img: torch.Tensor,
                      prompt_condition: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_channels_last and img.is_cuda:
            img = img.to(memory_format=torch.channels_last)
        try:
            if isinstance(self.img_encoder, PromptAdapterVisionTransformer):
                self.img_encoder.set_condition(prompt_condition)
                img_feat = self.img_encoder.forward_features(img)
                self.img_encoder.set_condition(None)
            elif hasattr(self.img_encoder, "forward_features"):
                img_feat = self.img_encoder.forward_features(img)
            else:
                out = self.img_encoder(img)
                img_feat = out.mean(dim=1) if out.ndim > 2 else out
        except Exception:
            out = self.img_encoder(img)
            img_feat = out.mean(dim=1) if out.ndim > 2 else out

        if img_feat.ndim > 2:
            img_feat = img_feat.mean(dim=1)
        return img_feat

    def set_backbone_trainable(self,
                                train_backbone: bool,
                                train_prompts: Optional[bool] = None) -> None:
        if train_prompts is None:
            train_prompts = True if train_backbone else self.train_prompts_when_frozen

        for param in self.img_encoder.parameters():
            param.requires_grad = train_backbone

        if not train_backbone:
            if self.train_block_layer_norm:
                for module in self.img_encoder.modules():
                    if isinstance(module, nn.LayerNorm):
                        for param in module.parameters():
                            param.requires_grad = True
            if train_prompts:
                for name, param in self.img_encoder.named_parameters():
                    if "prompt" in name or "adapter" in name:
                        param.requires_grad = True

        if self.prompt is not None:
            self.prompt.prompts.requires_grad = train_prompts or train_backbone

        self._backbone_frozen = not train_backbone
        self._prompts_trainable = train_prompts

    def is_backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def forward(self,
                img: torch.Tensor,
                env: Optional[torch.Tensor]):
        if img is None:
            raise ValueError("forward() received img=None. Check dataloader output format.")

        batch_size = img.shape[0]
        device = img.device

        env_feat, env_emb = self._encode_environment(env, batch_size, device)
        prompt_condition = None
        if self.condition_prompts_on_env and self.env_prompt_mapper is not None:
            prompt_condition = self.env_prompt_mapper(env_feat)

        img_feat = self._encode_image(img, prompt_condition)

        if self.prompt is not None:
            prompts = self.prompt(batch_size)
            prompt_summary = prompts.mean(dim=1)
            if prompt_summary.shape[1] == img_feat.shape[1]:
                img_feat = img_feat + prompt_summary
            else:
                warnings.warn("Prompt dim != img_feat dim: skipping pooled prompt injection.")

        img_emb = self.img_proj(img_feat)
        env_emb = env_emb.to(img_emb.device)

        fused = torch.cat([img_emb, env_emb], dim=1)
        multimodal_emb = self.fusion_mlp(fused)

        prototypes = self.prototype_bank()
        emb_norm = F.normalize(multimodal_emb, dim=1)
        prot_norm = F.normalize(prototypes, dim=1)
        proto_sim = emb_norm @ prot_norm.t()

        p_probs, a_vals = self.hurdle_head(multimodal_emb)
        proto_sig = torch.sigmoid(proto_sim)
        gate = torch.sigmoid(self.prototype_weight)
        final_presence = (1.0 - gate) * p_probs + gate * proto_sig
        final_pred = final_presence * a_vals

        return {
            "presence_prob": final_presence,
            "conditional_abundance": a_vals,
            "pred": final_pred,
            "proto_sim": proto_sim,
        }
