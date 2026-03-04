"""
Vision Mamba — Pure PyTorch implementation of a Selective State Space Model
for image classification.  No custom CUDA kernels required.

Architecture follows the Vision Mamba (Vim) paper:
  - Patch embedding (same as ViT)
  - Bidirectional Selective SSM blocks
  - Global average pooling → classification head

Reference: Zhu et al., "Vision Mamba: Efficient Visual Representation Learning
           with Bidirectional State Space Model", ICML 2024.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Core Mamba block (pure PyTorch, no custom CUDA)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model — the core of Mamba.

    Unlike classical SSMs with fixed A/B/C matrices, the Selective SSM
    makes B, C, and Delta *input-dependent* (selective), allowing the model
    to dynamically decide what to remember/forget.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection: x → (z, x_inner)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1-D depthwise convolution (causal)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters — input-dependent projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, delta

        # A is log-parameterized and *not* input-dependent
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) input sequence
        Returns:
            (B, L, D) output sequence
        """
        B, L, D = x.shape

        # Project input
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal 1-D convolution
        x_inner = rearrange(x_inner, 'b l d -> b d l')
        x_inner = self.conv1d(x_inner)[:, :, :L]  # trim to causal
        x_inner = rearrange(x_inner, 'b d l -> b l d')
        x_inner = F.silu(x_inner)

        # Compute input-dependent SSM parameters
        ssm_params = self.x_proj(x_inner)  # (B, L, 2*d_state + 1)
        B_param = ssm_params[..., :self.d_state]       # (B, L, d_state)
        C_param = ssm_params[..., self.d_state:2*self.d_state]  # (B, L, d_state)
        delta = F.softplus(ssm_params[..., -1])         # (B, L) — discretization step

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # SSM scan (sequential — pure PyTorch)
        y = self._ssm_scan(x_inner, A, B_param, C_param, delta)

        # Gated output
        y = y * F.silu(z)

        # Project back
        return self.out_proj(y)

    def _ssm_scan(self, x, A, B, C, delta):
        """
        Sequential SSM scan.

        Args:
            x: (B, L, d_inner)
            A: (d_inner, d_state)
            B: (B, L, d_state)
            C: (B, L, d_state)
            delta: (B, L)
        Returns:
            y: (B, L, d_inner)
        """
        batch, L, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        delta = delta.unsqueeze(-1)  # (B, L, 1)
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, 1, d_state) — broadcast with d_inner

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            # h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)  # (B, d_inner, d_state)
            # y = C * h + D * x
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            y_t = y_t + self.D * x[:, t]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class MambaBlock(nn.Module):
    """Pre-norm Mamba block with residual connection."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


# ---------------------------------------------------------------------------
# Bidirectional Vision Mamba block
# ---------------------------------------------------------------------------

class BidirectionalMambaBlock(nn.Module):
    """
    Process the patch sequence in both forward and backward directions,
    then fuse. This is the key innovation of Vision Mamba (Vim) over
    the original text-only Mamba.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.forward_block = MambaBlock(d_model, d_state, d_conv, expand)
        self.backward_block = MambaBlock(d_model, d_state, d_conv, expand)
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        fwd = self.forward_block(x)
        bwd = self.backward_block(x.flip(dims=[1])).flip(dims=[1])
        fused = self.fuse(torch.cat([fwd, bwd], dim=-1))
        return self.norm(fused)


# ---------------------------------------------------------------------------
# Full Vision Mamba model
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings (same as ViT)."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        return x


class VisionMamba(nn.Module):
    """
    Vision Mamba (Vim) for image classification.

    Architecture:
        Image → Patch Embedding → N × Bidirectional Mamba Blocks → Global Avg Pool → FC
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 100,
        d_model: int = 192,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)

        self.blocks = nn.ModuleList([
            BidirectionalMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, num_patches, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        return x

    def get_features(self, x):
        """Extract features before the classification head (for t-SNE / Grad-CAM)."""
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1)


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_vim_tiny(num_classes: int = 100, pretrained: bool = False, **kwargs):
    """Vision Mamba Tiny (~7M params)."""
    return VisionMamba(
        num_classes=num_classes,
        d_model=192,
        n_layers=12,
        d_state=16,
        d_conv=4,
        expand=2,
    )


def build_vim_small(num_classes: int = 100, pretrained: bool = False, **kwargs):
    """Vision Mamba Small (~26M params)."""
    return VisionMamba(
        num_classes=num_classes,
        d_model=384,
        n_layers=12,
        d_state=16,
        d_conv=4,
        expand=2,
    )
