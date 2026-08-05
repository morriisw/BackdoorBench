"""
Transformer-based model architectures to be evaluated (CIFAR-10 / GTSRB, 32x32 input).

ViT-Small: patch 4x4, dim 384, depth 12, heads 6, mlp_dim 1536, [CLS] token only.
DeiT-Tiny: patch 4x4, dim 192, depth 12, heads 3, mlp_dim 768, [CLS] + [DIST] tokens.
"""
import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer block.
    Pre-LN (LayerNorm before attention and MLP) is more stable than Post-LN
    and is the default in modern ViTs (DeiT, BEiT, etc.)

    Architecture per block:
      x = x + Attn(LN(x))     [residual self-attention]
      x = x + MLP(LN(x))      [residual feedforward]
    """
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        xn = self.norm1(x)
        x  = x + self.attn(xn, xn, xn)[0]
        x  = x + self.mlp(self.norm2(x))
        return x


class BaseVisionTransformer(nn.Module):
    """
    Shared backbone for ViT-Small and DeiT-Tiny on 32x32 images (CIFAR-10 / GTSRB).
    Set use_distillation_token=True to prepend an extra [DIST] token (DeiT-style).

    During training: returns (cls_logits, dist_logits) as a tuple so the custom
    training loop can apply separate CE losses (cls→true labels, dist→teacher labels).
    During evaluation: returns (cls_logits + dist_logits) / 2 as a single tensor
    for standard accuracy computation.
    """
    def __init__(
        self,
        image_size=32, patch_size=4, num_classes=10,
        dim=384, depth=12, heads=6, mlp_dim=1536,
        channels=3, dropout=0.1,
        use_distillation_token=False
    ):
        super().__init__()
        assert image_size % patch_size == 0
        assert dim % heads == 0

        self.use_distillation_token = use_distillation_token
        num_patches = (image_size // patch_size) ** 2      # number of patch tokens
        patch_dim   = channels * patch_size ** 2            # flattened pixel count per patch
        num_extra_tokens = 2 if use_distillation_token else 1  # [CLS] (+ [DIST])

        # Split image into patches, then project each flattened patch to `dim` (LN before and after stabilises from-scratch training)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # learnable classification token
        if use_distillation_token:
            self.dist_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # learnable distillation token (DeiT only)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_extra_tokens, dim) * 0.02)  # learnable position embeddings
        self.dropout = nn.Dropout(dropout)                             # embedding dropout

        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]  # stack of Pre-LN blocks
        )

        self.norm = nn.LayerNorm(dim)                       # final normalization before head(s)
        self.cls_head = nn.Linear(dim, num_classes)          # classification head on [CLS]
        if use_distillation_token:
            self.dist_head = nn.Linear(dim, num_classes)     # separate classification head on [DIST] (DeiT only)

    def forward(self, x):
        x = self.to_patch_embedding(x)                      # (B, num_patches, dim)
        b, n, _ = x.shape

        cls = repeat(self.cls_token, '() n d -> b n d', b=b)  # broadcast CLS token across batch
        tokens = [cls]
        if self.use_distillation_token:
            dist = repeat(self.dist_token, '() n d -> b n d', b=b)  # broadcast DIST token across batch
            tokens.append(dist)
        tokens.append(x)
        x = torch.cat(tokens, dim=1)                         # prepend [CLS] (+ [DIST]) to patch tokens

        x = self.dropout(x + self.pos_embedding[:, :x.shape[1]])  # add position info
        x = self.transformer(x)                             # (B, seq_len, dim)
        x = self.norm(x)

        cls_logits = self.cls_head(x[:, 0])
        if self.use_distillation_token:
            dist_logits = self.dist_head(x[:, 1])
            if self.training:
                return cls_logits, dist_logits
            return (cls_logits + dist_logits) / 2
        return cls_logits


def ViT_Small(num_classes=10):
    """
    ViT with patch size 4x4, 384 hidden dimensions, 12 transformer layers, 6 attention heads, 
    1536 MLP hidden dimensions, and 10 output classes (CIFAR-10).
    """
    return BaseVisionTransformer(
        patch_size=4, dim=384, depth=12, heads=6, mlp_dim=1536, 
        num_classes=num_classes, use_distillation_token=False
    )


def DeiT_Tiny(num_classes=10):
    """
    DeiT-Tiny/4: patch 4x4, dim 192, depth 12, heads 3, mlp_dim 768, with [CLS]+[DIST] tokens.
    """
    return BaseVisionTransformer(
        patch_size=4, dim=192, depth=12, heads=3, mlp_dim=768,
        num_classes=num_classes, use_distillation_token=True
    )