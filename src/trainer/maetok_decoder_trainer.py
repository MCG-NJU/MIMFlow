import torch
from src.trainer.base import BaseTrainer
from src.trainer.losses import ReconstructionLoss


class _ZeroPosterior:
    """占位 posterior，用于不需要 KL/熵约束的 decoder 微调。"""

    def __init__(self, device: torch.device):
        self.device = device

    def kl(self):
        return torch.zeros((1), device=self.device)

    def entropy(self):
        return torch.zeros((1), device=self.device)


class MAETokDecoderFinetuneTrainer(BaseTrainer):
    """
    仅微调 MAETok decoder：
    - 忽略 flow 模型
    - 使用重建 / 感知 / GAN 损失（可配置）
    """

    def __init__(
        self,
        noise_mode: str = "add",
        reconstruction_loss: str = "l2",
        reconstruction_weight: float = 1.0,
        masked_token_weight: float = 1.0,
        perceptual_loss: str = "lpips-convnext_s-1.0-0.1",
        perceptual_weight: float = 1.1,
        use_gan_loss: bool = True,
        discriminator_weight: float = 0.1,
        discriminator_start_epoch: int = 0,
        discriminator_type: str = "patchgan",
        discriminator_kwargs: dict | None = None,
        null_condition_p: float = 0.0,
        noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__(null_condition_p=null_condition_p, noise_std=noise_std)
        self.noise_mode = noise_mode
        # 复用 ReconstructionLoss 的 mask 加权与 loss 计算逻辑
        self._loss_helper = ReconstructionLoss(
            reconstruction_loss=reconstruction_loss,
            reconstruction_weight=reconstruction_weight,
            masked_token_weight=masked_token_weight,
            use_gan_loss=use_gan_loss,
            discriminator_weight=discriminator_weight,
            discriminator_start_epoch=discriminator_start_epoch,
            discriminator_type=discriminator_type,
            discriminator_kwargs=discriminator_kwargs,
            perceptual_loss=perceptual_loss,
            perceptual_weight=perceptual_weight,
            kl_weight=0.0,
            quantize_mode="nf",
        )
        self.reconstruction_weight = reconstruction_weight

    def _add_token_noise(self, latents: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, chans = latents.shape
        device = latents.device
        if self.noise_std > 0.0:
            noise_level_tensor = torch.full((bsz, 1, 1), self.noise_std, device=device)
        else:
            noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
        noise_level_tensor = noise_level_tensor.expand(-1, n_tokens, chans)
        noise = torch.randn(bsz, n_tokens, chans, device=device)
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

    def _impl_trainstep(self, vae, net, ema_net, raw_images, x, y, epoch, mask_ratio=None):
        # 兼容不同 encode 签名（部分模型不支持 mask_ratio）
        try:
            encode_out = vae.encode(x, mask_ratio=mask_ratio)
        except TypeError:
            encode_out = vae.encode(x)

        mask = None
        if isinstance(encode_out, tuple):
            quant = encode_out[0]
            if len(encode_out) >= 4 and torch.is_tensor(encode_out[3]):
                mask = encode_out[3]
        else:
            quant = encode_out

        if mask is not None and mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        quant = self._add_token_noise(quant)

        h, w = x.shape[2], x.shape[3]
        decoded = vae.decode(quant, x=x, h=h, w=w)
        if isinstance(decoded, tuple):
            decoded = decoded[0]

        targets = x * 0.5 + 0.5
        reconstructions = decoded * 0.5 + 0.5

        extra_result_dict = {
            "posterior": _ZeroPosterior(targets.device),
            "mask": mask,
        }
        gen_loss, gen_dict = self._loss_helper(
            inputs=targets,
            reconstructions=reconstructions,
            extra_result_dict=extra_result_dict,
            epoch=epoch,
            mode="generator",
        )

        if self._loss_helper.should_discriminator_be_trained(epoch):
            d_loss, d_dict = self._loss_helper(
                inputs=targets,
                reconstructions=reconstructions,
                extra_result_dict=extra_result_dict,
                epoch=epoch,
                mode="discriminator",
            )
        else:
            d_loss = torch.tensor(0.0, device=targets.device)
            d_dict = {
                "discriminator_loss": torch.tensor(0.0, device=targets.device),
                "logits_real": torch.tensor(0.0, device=targets.device),
                "logits_fake": torch.tensor(0.0, device=targets.device),
            }

        total_loss = gen_loss + d_loss

        out = {
            "epoch": torch.tensor(float(epoch), device=targets.device),
            "loss": total_loss,
            "recon_loss/total_loss": gen_dict["total_loss"].detach(),
            "recon_loss/reconstruction_loss": gen_dict["reconstruction_loss"],
            "recon_loss/perceptual_loss": gen_dict["perceptual_loss"],
            "recon_loss/ent_loss": gen_dict.get("ent_loss", torch.tensor(0.0, device=targets.device)),
            "recon_loss/weighted_gan_loss": gen_dict["weighted_gan_loss"],
            "recon_loss/discriminator_factor": gen_dict["discriminator_factor"],
            "recon_loss/d_weight": gen_dict["d_weight"],
            "recon_loss/gan_loss": gen_dict["gan_loss"],
            "recon_loss/psnr": gen_dict["psnr"],
            "d_loss/d_loss": d_loss,
            "d_loss/logits_real": d_dict["logits_real"],
            "d_loss/logits_fake": d_dict["logits_fake"],
        }
        return out
