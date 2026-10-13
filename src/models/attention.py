import torch
import torch.nn as nn
import numpy as np
from src.models.layers.rope import *
from src.models.layers.rmsnorm import RMSNorm
from src.models.layers.permutation import Permutation

class Attention(torch.nn.Module):
    def __init__(self, in_channels: int, head_channels: int, qk_norm: bool = True, gated: bool = True):
        assert in_channels % head_channels == 0
        super().__init__()
        self.gated = gated
        if gated:
            self.gate = torch.nn.Sequential(
                torch.nn.Linear(in_channels, in_channels),
                torch.nn.Sigmoid()
            )
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.head_channels = head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.q_norm = RMSNorm(head_channels) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_channels) if qk_norm else nn.Identity()
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward_spda(
        self, x: torch.Tensor, freq_cis, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        gate = None
        if self.gated:
            gate = self.gate(x)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)
            # 应用RoPE
            q = q.transpose(1, 2)  # (B, 1, H, head_dim)
            k = k.transpose(1, 2) # (B, 1, H, head_dim)
            if freq_cis.dim() == 2:
                rope = freq_cis[0:k.shape[1]]
            elif freq_cis.dim() == 3:
                rope = freq_cis[:,0:k.shape[1]]
            q, k = apply_rotary_emb(q, k, rope)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            q = q[:, :, -1, :].unsqueeze(-2)
        else:
            # 应用RoPE
            q = q.transpose(1, 2)  # (B, T, H, head_dim)
            k = k.transpose(1, 2)
            q, k = apply_rotary_emb(q, k, freq_cis)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        # print(k.shape, v.shape)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        if self.gated and gate is not None:
            x = gate * x
        x = self.proj(x)
        return x

    def forward(
        self, x: torch.Tensor, freq_cis, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        return self.forward_spda(x, freq_cis, mask, temp, which_cache)

class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        hidden_dim = int(np.ceil(2 * channels * expansion / 256 / 3)) * 256
        self.w1 = nn.Linear(channels, hidden_dim, bias=False)
        self.w3 = nn.Linear(channels, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.float()).type(x.dtype)
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4, permutation: Permutation = Permutation(100)):
        super().__init__()
        self.permutation = permutation
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, freq_cis, attn_mask: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, freq_cis, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


# ========== Optional-pos variant (keeps original Attention intact) ==========
class AttentionOptional(Attention):
    """兼容 freq_cis 可为空的注意力，用于无位置编码/可学习1D位置的场景。"""

    def forward_spda(
        self, x: torch.Tensor, freq_cis, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        gate = None
        if self.gated:
            gate = self.gate(x)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)
            q = q.transpose(1, 2)  # (B, T, H, head_dim)
            k = k.transpose(1, 2)  # (B, T, H, head_dim)
            if freq_cis is not None:
                if freq_cis.dim() == 2:
                    rope = freq_cis[0 : k.shape[1]]
                else:
                    rope = freq_cis[:, 0 : k.shape[1]]
                q, k = apply_rotary_emb(q, k, rope)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            q = q[:, :, -1, :].unsqueeze(-2)
        else:
            q = q.transpose(1, 2)  # (B, T, H, head_dim)
            k = k.transpose(1, 2)
            if freq_cis is not None:
                q, k = apply_rotary_emb(q, k, freq_cis)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        if self.gated and gate is not None:
            x = gate * x
        x = self.proj(x)
        return x


class AttentionBlockOptional(torch.nn.Module):
    """使用 AttentionOptional 的 Block，不改变原有 Block 行为。"""

    def __init__(self, channels: int, head_channels: int, expansion: int = 4, permutation: Permutation = Permutation(100)):
        super().__init__()
        self.permutation = permutation
        self.attention = AttentionOptional(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, freq_cis, attn_mask: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, freq_cis, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x
