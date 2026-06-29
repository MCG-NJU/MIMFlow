import time

import torch
import torch.nn as nn
from src.trainer.base import *
import copy
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from src.utils.no_grad import no_grad
from src.trainer.losses import ReconstructionLoss

class E2ETrainer(BaseTrainer):
    def __init__(
        self,
        null_condition_p: float = 0.1,
        noise_std: float = 0.05,
        noise_mode: str = "add",
        discriminator_weight: float = 0.1,
        discriminator_start_epoch: int = 0,
        use_gan_loss: bool = True,
        discriminator_type: str = "patchgan",
        discriminator_kwargs: dict | None = None,
        perceptual_loss: str = "lpips-convnext_s-1.0-0.1",
        perceptual_weight: float = 1.1,
        kl_weight: float = 1e-6,
        use_additive_noise: bool = True,
        gamma: float = 1.0,
        quantize_mode: str = "vae",
        masked_token_weight: float = 1.0,
        mask_after_encoder: bool = False,
        mask_nf: bool = False,
        **kwargs,
    ):
        # LightningCLI instantiates via YAML init_args -> kwargs; keep explicit args for clarity.
        super().__init__(null_condition_p=null_condition_p, noise_std=noise_std)
        self.noise_mode = noise_mode
        self.reconstructions_loss = ReconstructionLoss(
            discriminator_weight=discriminator_weight,
            discriminator_start_epoch=discriminator_start_epoch,
            use_gan_loss=use_gan_loss,
            discriminator_type=discriminator_type,
            discriminator_kwargs=discriminator_kwargs,
            perceptual_loss=perceptual_loss,
            perceptual_weight=perceptual_weight,
            kl_weight=kl_weight,
            quantize_mode=quantize_mode,
            masked_token_weight=masked_token_weight
        )
        self.quantize_mode = quantize_mode
        self.use_additive_noise = use_additive_noise
        self.gamma = gamma
        self.mask_after_encoder = mask_after_encoder
        self.mask_nf = mask_nf
        no_grad(self.reconstructions_loss.perceptual_loss)
        # for param_name, param in self.reconstructions_loss.named_parameters():
        #     print(param_name)
        # no_grad(self.reconstructions_loss)

    def _impl_trainstep(self, vae, net, ema_net, raw_images, x, y, epoch, mask_ratio=None):
        if self.mask_after_encoder:
            encode_out = vae.encode(x, mask_ratio=mask_ratio, mask_after_encoder=True)
            x_latents, posteriors, ids_restore, nf_latents = encode_out[:4]
            if self.mask_nf:
                nf_latents = x_latents
        else:
            try:
                x_latents, posteriors, ids_restore = vae.encode(x, mask_ratio=mask_ratio)
            except TypeError:
                x_latents, posteriors, ids_restore = vae.encode(x)
            # if self.mask_nf:
            nf_latents = x_latents
            # else:
            #     nf_latents, _, _ = vae.encode(x, mask_ratio=0)

        z_ori = nf_latents.detach().clone()
        device = nf_latents.device

        def _add_token_noise(latents: torch.Tensor) -> torch.Tensor:
            if self.gamma <= 0:
                return latents
            bsz, n_tokens, chans = latents.shape
            if self.noise_std > 0.0:
                noise_level_tensor = torch.full((bsz, 1, 1), self.noise_std, device=device)
            else:
                noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
            noise_level_tensor = noise_level_tensor.expand(-1, n_tokens, chans)
            noise = torch.randn(bsz, n_tokens, chans, device=device) * self.gamma
            if self.use_additive_noise:
                return latents + noise_level_tensor * noise
            return (1 - noise_level_tensor) * latents + noise_level_tensor * noise

        nf_latents = _add_token_noise(nf_latents)
        # decoder 分支与 flow 分支 tokens 可能不同；需要分别加噪
        if nf_latents is x_latents:
            x_latents = nf_latents
        else:
            x_latents = _add_token_noise(x_latents)
        # print(x_latents.shape)
        z, outputs, logdets = net(nf_latents, y, ids_restore)
        nll_loss = net.get_loss(z, logdets)

        decoded = vae.decode(x_latents, ids_restore)
        targets = x * 0.5 + 0.5
        reconstructions = decoded * 0.5 + 0.5

        mask_for_loss = None
        if ids_restore is not None:
            len_keep = x_latents.shape[1]
            total_tokens = ids_restore.shape[1]
            base_mask = torch.ones((ids_restore.shape[0], total_tokens), device=ids_restore.device)
            base_mask[:, :len_keep] = 0
            mask_for_loss = torch.gather(base_mask, dim=1, index=ids_restore)

        if self.quantize_mode == "vae":
            extra_result_dict = {
                "posterior": posteriors,
                "mask": mask_for_loss,
            }
        elif self.quantize_mode == "nf":
            assert self.gamma == 0.0, "gamma must be 0.0 for nf quantize mode"
            extra_result_dict = {
                "posterior": posteriors,
                "nf_loss": nll_loss,
                "mask": mask_for_loss,
            }
        else:
            raise ValueError(f"Unsupported quantize_mode={self.quantize_mode}, expected 'vae' or 'nf'")
        
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
                extra_result_dict=posteriors,
                epoch=epoch,
                mode="discriminator"
            )
            # print('='*10, 'discriminator')
        else:
            d_loss = torch.tensor(0.0, device=targets.device)
            # 记录零损失
            d_loss_dict = {
                "discriminator_loss": torch.tensor(0.0, device=targets.device),
                "logits_real": torch.tensor(0.0, device=targets.device),
                "logits_fake": torch.tensor(0.0, device=targets.device),
            }
        # print(type(nll_loss), type(g_loss), type(d_loss))
        total_loss = nll_loss + vae_loss + d_loss

        # Unify logging schema for easier cross-run comparison.
        # Note: LightningModel wraps these keys with "train/".
        z_ori_mean_log, z_ori_std_log, z_ori_min_log, z_ori_max_log = z_ori.mean(), z_ori.std(), z_ori.min(), z_ori.max()
        # z_mean_log, z_std_log, z_min_log, z_max_log = z.mean(), z.std(), z.min(), z.max()
        out = {
            "epoch": torch.tensor(float(epoch), device=device),
            "loss": total_loss,
            # VAE (generator) loss
            "vae_loss/vae_loss": vae_loss,
            "vae_loss/reconstruction_loss": vae_loss_dict["reconstruction_loss"],
            "vae_loss/perceptual_loss": vae_loss_dict["perceptual_loss"],
            "vae_loss/kl_loss": vae_loss_dict.get("kl_loss", vae_loss_dict.get('ent_loss',torch.tensor(0.0, device=device))),
            "vae_loss/weighted_gan_loss": vae_loss_dict["weighted_gan_loss"],
            "vae_loss/discriminator_factor": vae_loss_dict.get(
                "discriminator_factor", torch.tensor(0.0, device=device)
            ),
            "vae_loss/gan_loss": vae_loss_dict["gan_loss"],
            "vae_loss/d_weight": vae_loss_dict.get("d_weight", torch.tensor(0.0, device=device)),
            "vae_loss/psnr": vae_loss_dict['psnr'],
            # Statistics
            "stats/z_ori_mean": z_ori_mean_log,
            "stats/z_ori_std": z_ori_std_log,
            "stats/z_ori_min": z_ori_min_log,
            "stats/z_ori_max": z_ori_max_log,
            # "stats/z_mean": z_mean_log,
            # "stats/z_std": z_std_log,
            # "stats/z_min": z_min_log,
            # "stats/z_max": z_max_log,
            # Normalizing flow loss
            "nf_loss/nf_loss": nll_loss,
            "nf_loss/logdets": logdets.mean(),
            "nf_loss/loss_z": nll_loss+logdets.mean(),
            # Discriminator loss
            "d_loss/d_loss": d_loss,
            "d_loss/logits_real": d_loss_dict["logits_real"],
            "d_loss/logits_fake": d_loss_dict["logits_fake"],
            "d_loss/lecam_loss": d_loss_dict.get("lecam_loss", torch.tensor(0.0, device=device)),
        }
        return out