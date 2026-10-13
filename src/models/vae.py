import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class BaseVAE(torch.nn.Module):
    def __init__(self, scale=1.0, shift=0.0, patch_size=2, img_size=32):
        super().__init__()
        self.model = torch.nn.Identity()
        self.scale = scale
        self.shift = shift
        self.patch_size = patch_size
        self.img_size = img_size

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C')"""
        u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W)"""
        u = x.transpose(1, 2)
        return torch.nn.functional.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)

    def encode(self, x):
        return x/self.scale+self.shift

    def decode(self, x):
        return (x-self.shift)*self.scale

class LatentVAE(BaseVAE):
    def __init__(self, precompute=False, weight_path: str = "stabilityai/sd-vae-ft-ema", local_files_only: bool = False):
        super().__init__()
        self.precompute = precompute
        self.model = None
        self.weight_path = weight_path

        from diffusers.models import AutoencoderKL
        setattr(self, "model", AutoencoderKL.from_pretrained(self.weight_path, local_files_only=local_files_only))
        self.scaling_factor = self.model.config.scaling_factor

    @torch.no_grad()
    def encode(self, x):
        assert self.model is not None
        if self.precompute:
            x = self.patchify(x)
            return x.mul_(self.scaling_factor)
        x = self.model.encode(x).latent_dist.sample()
        x = self.patchify(x)
        return x.mul_(self.scaling_factor)

    @torch.no_grad()
    def decode(self, x):
        assert self.model is not None
        x = self.unpatchify(x)
        return self.model.decode(x.div_(self.scaling_factor)).sample

class TrainableVAE(BaseVAE):
    """
    Diffusers AutoencoderKL 的封装：
    - 支持两种编码输出：
      1) mean-only（类似 AE）：posterior_mode="mean"，可在 mean 上加噪 mean_noise_std
      2) 标准 VAE：posterior_mode="sample"（训练常用），并返回 posteriors（支持 .kl()）
    - 默认 encode(x) 仍返回 Tensor，兼容旧 trainer；
      需要 posteriors 时用 encode(..., return_info=True)
    """

    def __init__(
        self,
        weight_path: str = "stabilityai/sd-vae-ft-ema",
        local_files_only: bool = False,
        posterior_mode: str = "sample",  # "sample" | "mean"
    ):
        super().__init__()
        self.model = None
        self.weight_path = weight_path
        self.local_files_only = local_files_only
        self.posterior_mode = posterior_mode
        self.layer_norm = nn.LayerNorm(4 * self.patch_size ** 2, elementwise_affine=True)

        from diffusers.models import AutoencoderKL
        setattr(self, "model", AutoencoderKL.from_pretrained(self.weight_path, local_files_only=local_files_only))
        self.scaling_factor = self.model.config.scaling_factor

    def encode(self, x):
        assert self.model is not None
        posteriors = self.model.encode(x).latent_dist  # 支持 .mean / .std / .kl()
        if self.posterior_mode == "sample":
            latents = posteriors.sample()
        elif self.posterior_mode == "mean":
            latents = posteriors.mean
            posteriors = None
        else:
            raise ValueError(f"Unsupported posterior_mode={self.posterior_mode}, expected 'sample' or 'mean'")

        latents = self.patchify(latents)
        latents = self.layer_norm(latents)
        latents = latents.mul_(self.scaling_factor)
        return latents, posteriors, None

    def decode(self, x, ids_restore=None):
        assert self.model is not None
        x = self.unpatchify(x)
        return self.model.decode(x.div_(self.scaling_factor)).sample

def uint82fp(x):
    x = x.to(torch.float32)
    x = (x - 127.5) / 127.5
    return x

def fp2uint8(x):
    x = torch.clip_((x + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)
    return x
