import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x * norm)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    def get_1d(dim, size):
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
        omega = 1.0 / (10000 ** omega)
        pos = torch.arange(size, dtype=torch.float32)
        out = torch.einsum('p,o->po', pos, omega)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        return emb  # [size, dim]
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    emb_h = get_1d(half, grid_h)  # [H, half]
    emb_w = get_1d(half, grid_w)  # [W, half]
    emb_h = emb_h[:, None, :].repeat(1, grid_w, 1)  # [H, W, half]
    emb_w = emb_w[None, :, :].repeat(grid_h, 1, 1)  # [H, W, half]
    emb = torch.cat([emb_h, emb_w], dim=-1).view(-1, embed_dim)  # [H*W, D]
    return emb


# 支持非正方形 H×W 与非方形 patch_h×patch_w
class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_h=128, img_w=128, patch_h=16, patch_w=16,
                 in_chans=1, pca_dim=128, embed_dim=768, bias=True):
        super().__init__()
        assert img_h % patch_h == 0 and img_w % patch_w == 0
        self.img_size = (img_h, img_w)
        self.patch_size = (patch_h, patch_w)
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.proj1 = nn.Conv2d(in_chans, pca_dim,
                               kernel_size=(patch_h, patch_w),
                               stride=(patch_h, patch_w), bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_dim = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        # t shape: [B] in [0,1]
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=t.dtype, device=t.device) / half)
        args = t[:, None] * freqs[None, :] * math.pi * 2.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor):
        t = t.clamp(0, 1)
        t_freq = self.timestep_embedding(t, self.freq_dim)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor):
        return self.embedding(labels)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)  # [B, N, H, D]
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v  # [B, H, N, D]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        h = F.silu(x1) * x2
        return self.w3(self.drop(h))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_area, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_area * out_channels, bias=True)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden, drop=proj_drop)
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        s1, g1, b1, s2, g2, b2 = self.ada(c).chunk(6, dim=-1)  # shift1, scale1, gate1, shift2, scale2, gate2
        x = x + b1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, g1))
        x = x + b2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, g2))
        return x


class JiTMusic(nn.Module):
    def __init__(self,
                 input_h=128, input_w=128,
                 patch_h=16, patch_w=16,
                 in_channels=1,
                 hidden_size=768,
                 depth=12, num_heads=12, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0,
                 num_classes=1, bottleneck_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_h, self.input_w = input_h, input_w
        self.patch_h, self.patch_w = patch_h, patch_w
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(input_h, input_w, patch_h, patch_w,
                                               in_channels, bottleneck_dim, hidden_size, bias=True)

        gh = input_h // patch_h
        gw = input_w // patch_w
        self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_area=patch_h * patch_w, out_channels=self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        gh = self.input_h // self.patch_h
        gw = self.input_w // self.patch_w
        pe = get_2d_sincos_pos_embed(self.hidden_size, gh, gw)  # [N, D]
        self.pos_embed.data.copy_(pe.unsqueeze(0))
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(getattr(m, "bias", None)) if m.bias is not None else None
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(getattr(m, "bias", None)) if m.bias is not None else None
        self.apply(_init)

    def unpatchify(self, x):
        B, N, L = x.shape
        ph, pw, C = self.patch_h, self.patch_w, self.out_channels
        gh = self.input_h // ph
        gw = self.input_w // pw
        x = x.view(B, gh, gw, ph, pw, C).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, self.input_h, self.input_w)
        return x

    def forward(self, z, t, y):
        x = self.x_embedder(z) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y)
        for blk in self.blocks:
            x = blk(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)

def JiT_B_16(**kwargs):
    return JiTMusic(depth=12, hidden_size=768, num_heads=12,
                    bottleneck_dim=128, patch_h=16, patch_w=16, **kwargs)
def JiT_B_32(**kwargs):
    return JiTMusic(depth=12, hidden_size=768, num_heads=12,
                    bottleneck_dim=128, patch_h=32, patch_w=32, **kwargs)
def JiT_L_16(**kwargs):
    return JiTMusic(depth=24, hidden_size=1024, num_heads=16,
                    bottleneck_dim=128, patch_h=16, patch_w=16, **kwargs)
def JiT_L_32(**kwargs):
    return JiTMusic(depth=24, hidden_size=1024, num_heads=16,
                    bottleneck_dim=128, patch_h=32, patch_w=32, **kwargs)
def JiT_H_16(**kwargs):
    return JiTMusic(depth=32, hidden_size=1280, num_heads=16,
                    bottleneck_dim=256, patch_h=16, patch_w=16, **kwargs)
def JiT_H_32(**kwargs):
    return JiTMusic(depth=32, hidden_size=1280, num_heads=16,
                    bottleneck_dim=256, patch_h=32, patch_w=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}