import argparse
import importlib
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision as tv
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.data.imagenet import CenterCrop
from src.models.vae import LatentVAE

'''
accelerate launch --num_processes 8 mse_loss.py \
  --config-path configs/mimflow_l_validate_samples.yaml \
  --ckpt-path /path/to/checkpoints/epoch=1-step=10010.ckpt \
  --imagenet-val-root /tmp/data/val \
  --latent-vae-weight-path /path/to/workspace/home/.cache/huggingface/offline/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a \
  --batch-size 16 \
  --num-workers 4 \
  --lowpass-kernel-size 21
'''

def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_from_config(node: Dict):
    cls = import_class(node["class_path"])
    return cls(**node.get("init_args", {}))


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_ckpt_vae(config_path: Path, ckpt_path: Path, map_location: str = "cpu"):
    cfg = load_yaml(config_path)
    vae = instantiate_from_config(cfg["model"]["vae"])

    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = ckpt.get("state_dict", ckpt)
    train_vae = bool(cfg["model"].get("train_vae", False))

    preferred_prefix = "ema_vae." if train_vae else "vae."
    weights = strip_prefix(state_dict, preferred_prefix)
    if len(weights) == 0:
        fallback_prefix = "vae." if preferred_prefix == "ema_vae." else "ema_vae."
        weights = strip_prefix(state_dict, fallback_prefix)
        print(f"[warn] '{preferred_prefix}' not found, fallback to '{fallback_prefix}'")

    msg = vae.load_state_dict(weights, strict=False)
    print(f"[ckpt_vae] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    resolution = int(cfg["data"].get("train_image_size", 256))
    nf_init = cfg.get("model", {}).get("nf_trainer", {}).get("init_args", {})
    noise_mode = str(nf_init.get("noise_mode", "add"))
    noise_std = float(nf_init.get("noise_std", 0.0))
    return vae, resolution, noise_mode, noise_std


def normalize_for_vae(x01: torch.Tensor) -> torch.Tensor:
    # [0,1] -> [-1,1]
    return x01 * 2.0 - 1.0


def sanitize_kernel(kernel_size: int) -> int:
    k = int(kernel_size)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return k


def split_low_high_freq(x01: torch.Tensor, lowpass_kernel_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    k = sanitize_kernel(lowpass_kernel_size)
    low01 = F.avg_pool2d(x01, kernel_size=k, stride=1, padding=k // 2)
    high01 = (x01 - low01 + 0.5).clamp(0, 1)
    return low01, high01


@torch.inference_mode()
def reconstruct_with_vae(
    vae: torch.nn.Module,
    x01: torch.Tensor,
    use_amp: bool = True,
    latent_noise_mode: Optional[str] = None,
    latent_noise_std: float = 0.0,
) -> torch.Tensor:
    vae_model = vae.module if hasattr(vae, "module") else vae
    x = normalize_for_vae(x01)
    amp_enabled = use_amp and x.is_cuda

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        enc_out = vae_model.encode(x)

    ids_restore = None
    if isinstance(enc_out, tuple):
        if len(enc_out) >= 3:
            latents, _, ids_restore = enc_out[:3]
        elif len(enc_out) == 2:
            latents, ids_restore = enc_out
        else:
            latents = enc_out[0]
    else:
        latents = enc_out

    if latent_noise_std > 0:
        if latent_noise_mode == "add":
            latents = latents + latent_noise_std * torch.randn_like(latents)
        elif latent_noise_mode in ("linear", "slerp"):
            # Mirror training behavior: both linear/slerp use random interpolation with Gaussian noise.
            noise = torch.randn_like(latents)
            t = torch.rand(latents.shape[0], 1, 1, device=latents.device)
            if latent_noise_mode == "linear":
                latents = (1 - t) * latents + t * noise
            else:
                eps = 1e-6
                lat_norm = torch.norm(latents, dim=-1, keepdim=True).clamp_min(eps)
                noise_norm = torch.norm(noise, dim=-1, keepdim=True).clamp_min(eps)
                lat_unit = latents / lat_norm
                noise_unit = noise / noise_norm
                dot = (lat_unit * noise_unit).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
                omega = torch.acos(dot)
                sin_omega = torch.sin(omega).clamp_min(eps)
                coeff0 = torch.sin((1 - t) * omega) / sin_omega
                coeff1 = torch.sin(t * omega) / sin_omega
                dir_slerp = coeff0 * lat_unit + coeff1 * noise_unit
                out_norm = (1 - t) * lat_norm + t * noise_norm
                slerp_out = dir_slerp * out_norm
                lerp_out = (1 - t) * latents + t * noise
                latents = torch.where((sin_omega.abs() < eps), lerp_out, slerp_out)
        else:
            raise ValueError(f"Unsupported latent_noise_mode: {latent_noise_mode}")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        try:
            recon = vae_model.decode(latents, ids_restore)
        except TypeError:
            recon = vae_model.decode(latents)
    return (recon * 0.5 + 0.5).clamp(0, 1)


@dataclass
class RunningMSE:
    sse: float = 0.0
    n: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        diff = (pred.float() - target.float()) ** 2
        self.sse += diff.sum().item()
        self.n += diff.numel()

    @property
    def mse(self) -> float:
        if self.n == 0:
            return float("nan")
        return self.sse / self.n


def build_dataloader(data_root: Path, resolution: int, batch_size: int, num_workers: int) -> DataLoader:
    transform = tv.transforms.Compose(
        [
            CenterCrop(resolution),
            tv.transforms.ToTensor(),
        ]
    )
    dataset = tv.datasets.ImageFolder(str(data_root), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction MSE on ImageNet val for two VAEs.")
    parser.add_argument("--config-path", type=str, required=True, help="Lightning config yaml path.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Lightning ckpt path.")
    parser.add_argument("--imagenet-val-root", type=str, required=True, help="ImageNet val root (ImageFolder format).")
    parser.add_argument("--latent-vae-weight-path", type=str, required=True, help="LatentVAE pretrained weight path.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lowpass-kernel-size", type=int, default=21)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full set.")
    parser.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast.")
    parser.add_argument(
        "--ckpt-latent-noise-std",
        type=float,
        default=None,
        help="Override ckpt VAE latent noise std. Default: read from config model.nf_trainer.init_args.noise_std",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    use_amp = not args.no_amp

    config_path = Path(args.config_path).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    val_root = Path(args.imagenet_val_root).expanduser().resolve()
    latent_vae_weight = Path(args.latent_vae_weight_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not val_root.exists():
        raise FileNotFoundError(val_root)
    if not latent_vae_weight.exists():
        raise FileNotFoundError(latent_vae_weight)

    ckpt_vae, resolution, ckpt_noise_mode, ckpt_noise_std = load_ckpt_vae(config_path, ckpt_path, map_location="cpu")
    if args.ckpt_latent_noise_std is not None:
        ckpt_noise_std = float(args.ckpt_latent_noise_std)
    if accelerator.is_main_process:
        print(f"[ckpt_vae] latent_noise_mode={ckpt_noise_mode}, latent_noise_std={ckpt_noise_std}")
    ckpt_vae = ckpt_vae.eval()

    latent_vae = LatentVAE(precompute=False, weight_path=str(latent_vae_weight)).eval()

    loader = build_dataloader(
        data_root=val_root,
        resolution=resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    ckpt_vae, latent_vae, loader = accelerator.prepare(ckpt_vae, latent_vae, loader)

    stats = {
        "ckpt_vae/orig": RunningMSE(),
        "ckpt_vae/low": RunningMSE(),
        "ckpt_vae/high": RunningMSE(),
        "latent_vae/orig": RunningMSE(),
        "latent_vae/low": RunningMSE(),
        "latent_vae/high": RunningMSE(),
    }

    seen_local = 0
    max_local_samples = ceil(args.max_samples / accelerator.num_processes) if args.max_samples > 0 else 0
    for batch_idx, (x01, _) in enumerate(loader):
        x01 = x01.to(device, non_blocking=True)
        low01, high01 = split_low_high_freq(x01, lowpass_kernel_size=args.lowpass_kernel_size)

        ckpt_recon_orig = reconstruct_with_vae(
            ckpt_vae,
            x01,
            use_amp=use_amp,
            latent_noise_mode=ckpt_noise_mode,
            latent_noise_std=ckpt_noise_std,
        )
        ckpt_recon_low = reconstruct_with_vae(
            ckpt_vae,
            low01,
            use_amp=use_amp,
            latent_noise_mode=ckpt_noise_mode,
            latent_noise_std=ckpt_noise_std,
        )
        ckpt_recon_high = reconstruct_with_vae(
            ckpt_vae,
            high01,
            use_amp=use_amp,
            latent_noise_mode=ckpt_noise_mode,
            latent_noise_std=ckpt_noise_std,
        )

        latent_recon_orig = reconstruct_with_vae(latent_vae, x01, use_amp=use_amp)
        latent_recon_low = reconstruct_with_vae(latent_vae, low01, use_amp=use_amp)
        latent_recon_high = reconstruct_with_vae(latent_vae, high01, use_amp=use_amp)

        stats["ckpt_vae/orig"].update(ckpt_recon_orig, x01)
        stats["ckpt_vae/low"].update(ckpt_recon_low, low01)
        stats["ckpt_vae/high"].update(ckpt_recon_high, high01)
        stats["latent_vae/orig"].update(latent_recon_orig, x01)
        stats["latent_vae/low"].update(latent_recon_low, low01)
        stats["latent_vae/high"].update(latent_recon_high, high01)

        seen_local += x01.size(0)
        if max_local_samples > 0 and seen_local >= max_local_samples:
            break

        if accelerator.is_main_process and (batch_idx + 1) % 20 == 0:
            approx_seen = seen_local * accelerator.num_processes
            print(f"[progress] batch={batch_idx + 1}, approx_seen={approx_seen}")

    accelerator.wait_for_everyone()
    global_seen = accelerator.reduce(torch.tensor(seen_local, device=device, dtype=torch.long), reduction="sum").item()

    reduced_metrics = {}
    for key, stat in stats.items():
        sse = accelerator.reduce(torch.tensor(stat.sse, device=device, dtype=torch.float64), reduction="sum")
        n = accelerator.reduce(torch.tensor(stat.n, device=device, dtype=torch.float64), reduction="sum")
        mse = (sse / n).item() if n.item() > 0 else float("nan")
        reduced_metrics[key] = mse

    if accelerator.is_main_process:
        print("\n=== Reconstruction MSE (lower is better) ===")
        print(f"Samples evaluated: {global_seen}")
        for key in [
            "ckpt_vae/orig",
            "ckpt_vae/low",
            "ckpt_vae/high",
            "latent_vae/orig",
            "latent_vae/low",
            "latent_vae/high",
        ]:
            print(f"{key:20s}: {reduced_metrics[key]:.8f}")


if __name__ == "__main__":
    main()
