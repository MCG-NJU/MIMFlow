import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.trainer.e2e_trainer import E2ETrainer


class _ZeroPosterior:
    """占位 posterior，用于无 KL/熵约束的连续 tokenizer。"""

    def __init__(self, device: torch.device):
        self.device = device

    def kl(self):
        return torch.zeros((1), device=self.device)

    def entropy(self):
        return torch.zeros((1), device=self.device)


def _build_mlp(hidden_size: int, projector_dim: int, z_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def _mean_flat(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 1:
        return x
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class MAETokNFTrainer(E2ETrainer):
    """
    面向 MAETok（1D tokenizer）的 NF 端到端训练器。
    复用 E2ETrainer 的 GAN/感知重建损失与 token 噪声注入逻辑，
    但假定 encode 返回的即为 1D token 序列（无显式 posterior）。
    """

    def __init__(
        self,
        quantize_mode: str = "nf",
        nf_repa: bool = False,
        nf_repa_block: int | None = None,
        nf_repa_attn_index: int = 0,
        nf_repa_weight: float = 0.1,
        nf_repa_proj_dim: int = 1024,
        nf_repa_align: str = "attn_pool",
        nf_repa_temperature: float = 1.0,
        **kwargs,
    ):
        # 默认使用 "nf" 模式，避免对 posterior.kl() 的依赖
        super().__init__(quantize_mode=quantize_mode, **kwargs)
        self.nf_repa = nf_repa
        self.nf_repa_block = nf_repa_block
        self.nf_repa_attn_index = nf_repa_attn_index
        self.nf_repa_weight = nf_repa_weight
        self.nf_repa_proj_dim = nf_repa_proj_dim
        self.nf_repa_align = nf_repa_align
        self.nf_repa_temperature = nf_repa_temperature

        self.nf_repa_query_proj: nn.Module | None = None
        self.nf_repa_key_proj: nn.Module | None = None
        self.nf_repa_value_proj: nn.Module | None = None

    def _add_token_noise(self, latents: torch.Tensor) -> torch.Tensor:
        if self.gamma <= 0:
            return latents
        bsz, n_tokens, chans = latents.shape
        device = latents.device
        if self.noise_std > 0.0:
            noise_level_tensor = torch.full((bsz, 1, 1), self.noise_std, device=device)
        else:
            noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
        noise_level_tensor = noise_level_tensor.expand(-1, n_tokens, chans)
        noise = torch.randn(bsz, n_tokens, chans, device=device) * self.gamma
        t = torch.rand(bsz, 1, 1, device=device)
        if self.noise_mode == 'add':
            return latents + noise_level_tensor * noise
        elif self.noise_mode == 'linear':
            return (1 - t) * latents + t * noise
        elif self.noise_mode == 'slerp':
            eps = 1e-6
            lat_norm = torch.norm(latents, dim=-1, keepdim=True).clamp_min(eps)
            noise_norm = torch.norm(noise, dim=-1, keepdim=True).clamp_min(eps)
            lat_unit = latents / lat_norm
            noise_unit = noise / noise_norm

            dot = (lat_unit * noise_unit).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)

            coeff0 = torch.sin((1 - t) * omega) / sin_omega.clamp_min(eps)
            coeff1 = torch.sin(t * omega) / sin_omega.clamp_min(eps)
            dir_slerp = coeff0 * lat_unit + coeff1 * noise_unit

            # 保持幅值平滑变化：方向做球面插值，范数做线性插值
            out_norm = (1 - t) * lat_norm + t * noise_norm
            slerp_out = dir_slerp * out_norm

            # 小角度时退化为线性插值，避免数值不稳定
            lerp_out = (1 - t) * latents + t * noise
            small_angle = sin_omega.abs() < eps
            return torch.where(small_angle, lerp_out, slerp_out)
        else:
            raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")

    def _nan_stats(self, name: str, tensor: torch.Tensor) -> str:
        if tensor.numel() == 0:
            return f"{name}: empty"
        finite = torch.isfinite(tensor)
        finite_count = finite.sum().item()
        total = tensor.numel()
        if finite_count == 0:
            return f"{name}: all_non_finite total={total}"
        safe = tensor[finite]
        return (
            f"{name}: shape={tuple(tensor.shape)} "
            f"finite={finite_count}/{total} "
            f"min={safe.min().item():.6g} "
            f"max={safe.max().item():.6g} "
            f"mean={safe.mean().item():.6g} "
            f"std={safe.std().item():.6g}"
        )

    def _check_nan(self, name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all().item():
            msg = "[NaNDebug] Non-finite detected: " + self._nan_stats(name, tensor)
            raise RuntimeError(msg)

    def _check_range(self, name: str, tensor: torch.Tensor, lo: float, hi: float) -> None:
        if tensor.numel() == 0:
            return
        if not torch.isfinite(tensor).all().item():
            return
        out_of_range = (tensor < lo) | (tensor > hi)
        if out_of_range.any().item():
            safe = tensor[out_of_range]
            msg = (
                f"[NaNDebug] Range violation in {name}: "
                f"expected=[{lo}, {hi}] "
                f"violations={int(out_of_range.sum().item())}/{tensor.numel()} "
                f"min={safe.min().item():.6g} "
                f"max={safe.max().item():.6g}"
            )
            print(msg)

    def _ensure_nf_repa_modules(self, vae, token_dim: int, device: torch.device, dtype: torch.dtype) -> bool:
        if self.nf_repa_query_proj is not None:
            return True
        if not hasattr(vae, "repa_model") or vae.repa_model is None:
            return False
        repa_z_dim = vae.repa_model.embed_dim
        self.nf_repa_query_proj = _build_mlp(token_dim, self.nf_repa_proj_dim, repa_z_dim)
        self.nf_repa_key_proj = nn.Linear(repa_z_dim, repa_z_dim, bias=False)
        self.nf_repa_value_proj = nn.Linear(repa_z_dim, repa_z_dim, bias=False)
        self.nf_repa_query_proj.to(device=device, dtype=dtype)
        self.nf_repa_key_proj.to(device=device, dtype=dtype)
        self.nf_repa_value_proj.to(device=device, dtype=dtype)
        return True

    def _compute_nf_repa_loss(self, vae, nf_tokens: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not self.nf_repa or self.nf_repa_weight <= 0:
            return torch.tensor(0.0, device=nf_tokens.device)
        if not self._ensure_nf_repa_modules(
            vae, nf_tokens.shape[-1], nf_tokens.device, nf_tokens.dtype
        ):
            return torch.tensor(0.0, device=nf_tokens.device)

        with torch.no_grad():
            rescale_x = vae.scale(vae.de_scale(x))
            dino_tokens = vae.repa_model.forward_features(rescale_x)[
                :, vae.repa_model.num_prefix_tokens :
            ]
        dino_tokens = dino_tokens.to(dtype=nf_tokens.dtype)

        if self.nf_repa_align == "attn_pool":
            q = self.nf_repa_query_proj(nf_tokens)
            k = self.nf_repa_key_proj(dino_tokens)
            v = self.nf_repa_value_proj(dino_tokens)
            attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
            if self.nf_repa_temperature != 1.0:
                attn = attn / self.nf_repa_temperature
            attn = attn.softmax(dim=-1)
            z_hat = torch.matmul(attn, v)
        elif self.nf_repa_align == "avg_1d":
            dino_tokens = F.adaptive_avg_pool1d(
                dino_tokens.permute(0, 2, 1), nf_tokens.shape[1]
            ).permute(0, 2, 1)
            q = self.nf_repa_query_proj(nf_tokens)
            z_hat = self.nf_repa_key_proj(dino_tokens)
        else:
            raise ValueError(f"Unsupported nf_repa_align={self.nf_repa_align}")

        q = F.normalize(q, dim=-1)
        z_hat = F.normalize(z_hat, dim=-1)
        repa_loss = _mean_flat(-(q * z_hat).sum(dim=-1))
        repa_loss = repa_loss.mean()
        repa_loss *= self.nf_repa_weight
        return repa_loss

    def _impl_trainstep(self, vae, net, ema_net, raw_images, x, y, epoch, mask_ratio=None):
        # 兼容不同 encode 签名（MaskAEModel 无 mask_ratio）
        try:
            encode_out = vae.encode(x, mask_ratio=mask_ratio)
        except TypeError:
            encode_out = vae.encode(x)

        ids_restore = None
        mask = None
        nf_latents = encode_out[0] if isinstance(encode_out, tuple) else encode_out
        if isinstance(encode_out, tuple):
            if len(encode_out) >= 3 and torch.is_tensor(encode_out[2]):
                ids_restore = encode_out[2]
            if len(encode_out) >= 4 and torch.is_tensor(encode_out[3]):
                mask = encode_out[3]

        # token 级噪声注入
        z_ori = nf_latents.detach().clone()
        nf_latents = self._add_token_noise(nf_latents)
        x_latents = nf_latents

        nf_repa_tokens = None
        nf_repa_tokens_list = None
        handles = []
        if self.nf_repa and self.nf_repa_block is not None and hasattr(net, "blocks"):
            nf_repa_tokens_list = []
            block_idx = self.nf_repa_block
            attn_idx = self.nf_repa_attn_index
            if -len(net.blocks) <= block_idx < len(net.blocks):
                block = net.blocks[block_idx]
                if hasattr(block, "attn_blocks") and 0 <= attn_idx < len(block.attn_blocks):
                    attn_block = block.attn_blocks[attn_idx]
                    if hasattr(attn_block, "attention"):
                        def _capture_nf_tokens(_module, _input, output):
                            token_out = output
                            if hasattr(block, "permutation"):
                                token_out = block.permutation(token_out, inverse=True)
                            nf_repa_tokens_list.append(token_out)

                        handles.append(attn_block.attention.register_forward_hook(_capture_nf_tokens))

        try:
            z, outputs, logdets = net(nf_latents, y, ids_restore)
        finally:
            for handle in handles:
                handle.remove()
        # print(len(nf_repa_tokens_list))
        if nf_repa_tokens_list is not None and len(nf_repa_tokens_list) > 0:
            nf_repa_tokens = nf_repa_tokens_list[-1]
        nll_loss = net.get_loss(z, logdets)

        # 使用相同 token 进行解码，重建输入
        h, w = x.shape[2], x.shape[3]
        decoded, aux_loss = vae.decode_all(x_latents, input=x, h=h, w=w, mask=mask)
        targets = x * 0.5 + 0.5
        reconstructions = decoded * 0.5 + 0.5
        # reconstructions = (decoded * 0.5 + 0.5).clamp(0.0, 1.0)

        # NaN/Inf diagnostics for reconstruction path
        # self._check_nan("x", x)
        # self._check_nan("x_latents", x_latents)
        # self._check_nan("decoded", decoded)
        # self._check_nan("targets", targets)
        # self._check_nan("reconstructions", reconstructions)
        # self._check_range("decoded", decoded, -1.0, 1.0)

        mask_for_loss = mask
        zero_posterior = _ZeroPosterior(targets.device)
        extra_result_dict = {
            "posterior": zero_posterior,
            "mask": mask_for_loss,
        }
        if self.quantize_mode == "nf":
            extra_result_dict["nf_loss"] = nll_loss

        vae_loss, vae_loss_dict = self.reconstructions_loss(
            inputs=targets,
            reconstructions=reconstructions,
            extra_result_dict=extra_result_dict,
            epoch=epoch,
            mode="generator",
        )

        if self.reconstructions_loss.should_discriminator_be_trained(epoch):
            d_loss, d_loss_dict = self.reconstructions_loss(
                inputs=targets,
                reconstructions=reconstructions,
                extra_result_dict=zero_posterior,
                epoch=epoch,
                mode="discriminator",
            )
        else:
            d_loss = torch.tensor(0.0, device=targets.device)
            d_loss_dict = {
                "discriminator_loss": torch.tensor(0.0, device=targets.device),
                "logits_real": torch.tensor(0.0, device=targets.device),
                "logits_fake": torch.tensor(0.0, device=targets.device),
            }
        # print(nf_repa_tokens.shape)
        nf_repa_loss = self._compute_nf_repa_loss(vae, nf_repa_tokens if nf_repa_tokens is not None else z, x)
        total_loss = nll_loss + vae_loss + d_loss + sum(aux_loss) + nf_repa_loss

        device = targets.device
        z_ori_mean_log, z_ori_std_log = z_ori.mean(), z_ori.std()
        z_ori_min_log, z_ori_max_log = z_ori.min(), z_ori.max()

        out = {
            "epoch": torch.tensor(float(epoch), device=device),
            "loss": total_loss,
            "aux_loss/repa": aux_loss[0],
            "aux_loss/dino": aux_loss[1],
            "aux_loss/hog": aux_loss[2],
            "aux_loss/clip": aux_loss[3],
            "aux_loss/nf_repa": nf_repa_loss,
            # VAE (generator) loss
            "vae_loss/vae_loss": vae_loss,
            "vae_loss/reconstruction_loss": vae_loss_dict["reconstruction_loss"],
            "vae_loss/perceptual_loss": vae_loss_dict["perceptual_loss"],
            "vae_loss/kl_loss": vae_loss_dict.get(
                "kl_loss", vae_loss_dict.get("ent_loss", torch.tensor(0.0, device=device))
            ),
            "vae_loss/weighted_gan_loss": vae_loss_dict["weighted_gan_loss"],
            "vae_loss/discriminator_factor": vae_loss_dict.get(
                "discriminator_factor", torch.tensor(0.0, device=device)
            ),
            "vae_loss/gan_loss": vae_loss_dict["gan_loss"],
            "vae_loss/d_weight": vae_loss_dict.get("d_weight", torch.tensor(0.0, device=device)),
            "vae_loss/psnr": vae_loss_dict["psnr"],
            # Statistics
            "stats/z_ori_mean": z_ori_mean_log,
            "stats/z_ori_std": z_ori_std_log,
            "stats/z_ori_min": z_ori_min_log,
            "stats/z_ori_max": z_ori_max_log,
            # Normalizing flow loss
            "nf_loss/nf_loss": nll_loss,
            "nf_loss/logdets": logdets.mean(),
            "nf_loss/loss_z": nll_loss + logdets.mean(),
            # Discriminator loss
            "d_loss/d_loss": d_loss,
            "d_loss/logits_real": d_loss_dict["logits_real"],
            "d_loss/logits_fake": d_loss_dict["logits_fake"],
            "d_loss/lecam_loss": d_loss_dict.get("lecam_loss", torch.tensor(0.0, device=device)),
        }
        return out

