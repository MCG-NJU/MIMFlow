import torch
from pathlib import Path

from src.trainer.base import BaseTrainer


class MAETokNFLatentTrainer(BaseTrainer):
    """
    仅在预训练 MAETok 隐空间训练 NF：
    - MAETok 冻结，不参与反向传播；
    - 训练目标只保留 NF NLL loss。
    """

    def __init__(
        self,
        noise_mode: str = "add",
        noise_std: float = 0.0,
        gamma: float = 1.0,
        null_condition_p: float = 0.1,
        maetok_pretrained_path: str | None = None,
        maetok_pretrained_strict: bool = False,
        freeze_maetok_eval_mode: bool = True,
    ):
        super().__init__(null_condition_p=null_condition_p, noise_std=noise_std)
        self.noise_mode = noise_mode
        self.gamma = gamma

        # 通过 HuggingFace from_pretrained 从本地目录加载 MAETok 参数
        self.maetok_pretrained_path = maetok_pretrained_path
        self.maetok_pretrained_strict = maetok_pretrained_strict
        self.freeze_maetok_eval_mode = freeze_maetok_eval_mode

        self._maetok_loaded = False
        self._maetok_frozen = False

    def on_fit_start(self, vae) -> None:
        self._maybe_load_maetok(vae)
        self._freeze_maetok(vae)

    def _maybe_load_maetok(self, vae) -> None:
        if self._maetok_loaded:
            return
        self._maetok_loaded = True

        if not self.maetok_pretrained_path:
            return

        state_dict = self._load_state_dict_from_local_path(self.maetok_pretrained_path)
        msg = vae.load_state_dict(state_dict, strict=self.maetok_pretrained_strict)
        print(f"[MAETokNFLatentTrainer] loaded MAETok from local path: {self.maetok_pretrained_path}")
        print(f"[MAETokNFLatentTrainer] missing keys: {msg.missing_keys}")
        print(f"[MAETokNFLatentTrainer] unexpected keys: {msg.unexpected_keys}")

    def _load_state_dict_from_local_path(self, local_path: str) -> dict[str, torch.Tensor]:
        """
        从本地 HuggingFace 仓库目录加载权重，不重新构造模型，避免 from_pretrained
        在自定义构造器上出现重复 kwargs 的问题。
        """
        repo_dir = Path(local_path)
        if not repo_dir.exists():
            raise FileNotFoundError(f"maetok_pretrained_path does not exist: {local_path}")
        if not repo_dir.is_dir():
            raise NotADirectoryError(f"maetok_pretrained_path must be a directory: {local_path}")

        safetensors_path = repo_dir / "model.safetensors"
        pytorch_bin_path = repo_dir / "pytorch_model.bin"
        pytorch_pt_path = repo_dir / "pytorch_model.pt"

        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file as safe_load_file  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Found model.safetensors but failed to import safetensors. "
                    "Please install safetensors or provide pytorch_model.bin."
                ) from e
            state_dict = safe_load_file(str(safetensors_path))
            return state_dict

        if pytorch_bin_path.exists():
            obj = torch.load(str(pytorch_bin_path), map_location="cpu", weights_only=False)
            if not isinstance(obj, dict):
                raise ValueError(f"Unsupported checkpoint format in {pytorch_bin_path}")
            return obj.get("state_dict", obj)

        if pytorch_pt_path.exists():
            obj = torch.load(str(pytorch_pt_path), map_location="cpu", weights_only=False)
            if not isinstance(obj, dict):
                raise ValueError(f"Unsupported checkpoint format in {pytorch_pt_path}")
            return obj.get("state_dict", obj)

        raise FileNotFoundError(
            f"No supported weight file found in {local_path}. "
            "Expected one of: model.safetensors, pytorch_model.bin, pytorch_model.pt"
        )

    def _freeze_maetok(self, vae) -> None:
        if self._maetok_frozen:
            if self.freeze_maetok_eval_mode:
                vae.eval()
            return
        for param in vae.parameters():
            param.requires_grad = False
        if self.freeze_maetok_eval_mode:
            vae.eval()
        self._maetok_frozen = True

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
        if self.noise_mode == "add":
            return latents + noise_level_tensor * noise
        if self.noise_mode == "linear":
            return (1 - t) * latents + t * noise
        if self.noise_mode == "slerp":
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

            out_norm = (1 - t) * lat_norm + t * noise_norm
            slerp_out = dir_slerp * out_norm

            lerp_out = (1 - t) * latents + t * noise
            small_angle = sin_omega.abs() < eps
            return torch.where(small_angle, lerp_out, slerp_out)

        raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")

    def _impl_trainstep(self, vae, net, ema_net, raw_images, x, y, epoch, mask_ratio=None):
        # Fallback for non-Lightning entrypoints.
        # self._maybe_load_maetok(vae)
        # self._freeze_maetok(vae)

        with torch.no_grad():
            try:
                encode_out = vae.encode(x, mask_ratio=mask_ratio)
            except TypeError:
                encode_out = vae.encode(x)

        ids_restore = None
        nf_latents = encode_out[0] if isinstance(encode_out, tuple) else encode_out
        if isinstance(encode_out, tuple) and len(encode_out) >= 3 and torch.is_tensor(encode_out[2]):
            ids_restore = encode_out[2]

        nf_latents = self._add_token_noise(nf_latents)
        z, _, logdets = net(nf_latents, y, ids_restore)
        nll_loss = net.get_loss(z, logdets)

        device = nll_loss.device
        out = {
            "epoch": torch.tensor(float(epoch), device=device),
            "loss": nll_loss,
            "nf_loss/nf_loss": nll_loss,
            "nf_loss/logdets": logdets.mean(),
            "nf_loss/loss_z": nll_loss + logdets.mean(),
            "stats/z_mean": z.mean(),
            "stats/z_std": z.std(),
            "stats/z_min": z.min(),
            "stats/z_max": z.max(),
        }
        return out
