import argparse
import contextlib
import importlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.imagenet import PixImageNet
'''
accelerate launch --num_processes 8 tools/extract_maetok_nf_features.py \
  --config configs/mimflow_l_phase1.yaml \
  --ckpt /path/to/checkpoints/0.6-0.8-std0.3/epoch=49-step=250250.ckpt \
  --data-root /tmp/data \
  --save-maetok-pre-proj \
  --split val \
  --output-root /tmp/features/highmask-fullimage-nocondition-gamma0-before-quant \
  --batch-size 128 \
  --num-workers 8 \
  --use-ema \
  --mixed-precision bf16 \
  --nf-blocks 2 \
  --nf-attn-index 0 \
  --nf-source attention \
  --undo-permutation \
  --maetok-pool mean \
  --nf-pool mean \
  --nf-conditioning none \
  --nf-noise-gamma 0
'''


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_from_cfg(cfg_node: Dict[str, Any]):
    cls = import_class(cfg_node["class_path"])
    init_args = cfg_node.get("init_args", {})
    return cls(**init_args)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pool_tokens(x: torch.Tensor, mode: str, token_index: int) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "mean":
        return x.mean(dim=1)
    if mode == "max":
        return x.max(dim=1).values
    if mode == "flatten":
        return x.flatten(1)
    if mode == "token":
        idx = token_index
        if idx < 0:
            idx = x.shape[1] + idx
        if idx < 0 or idx >= x.shape[1]:
            raise IndexError(f"token_index={token_index} out of range for seq_len={x.shape[1]}")
        return x[:, idx]
    raise ValueError(f"Unknown pool mode: {mode}")


@dataclass
class ShardWriter:
    output_dir: str
    split: str
    rank: int
    shard_size: int
    shard_id: int = 0
    sample_count: int = 0
    buffer: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    saved_files: List[str] = field(default_factory=list)

    def append(self, batch_dict: Dict[str, torch.Tensor]) -> None:
        for key, value in batch_dict.items():
            if key not in self.buffer:
                self.buffer[key] = []
            self.buffer[key].append(value.detach().cpu())
        self.sample_count += int(batch_dict["labels"].shape[0])
        if self.sample_count >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self.sample_count == 0:
            return
        save_obj = {}
        for key, chunks in self.buffer.items():
            save_obj[key] = torch.cat(chunks, dim=0)
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, f"{self.split}_rank{self.rank:03d}_shard_{self.shard_id:05d}.pt")
        torch.save(save_obj, out_path)
        self.saved_files.append(out_path)
        self.shard_id += 1
        self.sample_count = 0
        self.buffer = {}


def parse_int_list(text: str) -> List[int]:
    if text.strip() == "":
        return []
    return [int(x.strip()) for x in text.split(",")]


def add_nf_additive_noise(latents: torch.Tensor, noise_std: float, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        return latents
    bsz, n_tokens, chans = latents.shape
    if noise_std > 0.0:
        noise_level = torch.full((bsz, 1, 1), float(noise_std), device=latents.device, dtype=latents.dtype)
    else:
        noise_level = torch.rand((bsz, 1, 1), device=latents.device, dtype=latents.dtype)
    noise_level = noise_level.expand(-1, n_tokens, chans)
    noise = torch.randn_like(latents) * float(gamma)
    return latents + noise_level * noise


def load_maetok_nf(cfg: Dict[str, Any], ckpt_path: str, use_ema: bool, device: torch.device):
    vae = instantiate_from_cfg(cfg["model"]["vae"])
    nf_model = instantiate_from_cfg(cfg["model"]["model"])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    vae_prefix = "ema_vae." if use_ema else "vae."
    nf_prefix = "ema_model." if use_ema else "model."

    vae_state = {}
    nf_state = {}
    for key, value in state_dict.items():
        if key.startswith(vae_prefix):
            vae_state[key[len(vae_prefix) :]] = value
        if key.startswith(nf_prefix):
            nf_state[key[len(nf_prefix) :]] = value

    vae_msg = vae.load_state_dict(vae_state, strict=False)
    nf_msg = nf_model.load_state_dict(nf_state, strict=False)
    print(f"[load] vae missing={vae_msg.missing_keys}, unexpected={vae_msg.unexpected_keys}")
    print(f"[load] nf  missing={nf_msg.missing_keys}, unexpected={nf_msg.unexpected_keys}")

    vae.eval().to(device)
    nf_model.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False
    for p in nf_model.parameters():
        p.requires_grad = False
    return vae, nf_model


def main():
    parser = argparse.ArgumentParser(description="Stage-1: extract MAETok + NF intermediate features")
    parser.add_argument("--config", type=str, required=True, help="Path to maetok-nf yaml config")
    parser.add_argument("--ckpt", type=str, required=True, help="Trained Lightning checkpoint path")
    parser.add_argument("--output-root", type=str, required=True, help="Feature output root directory")
    parser.add_argument("--data-root", type=str, default=None, help="ImageNet root override (contains train/val)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--shard-size", type=int, default=20000, help="samples per saved shard")
    parser.add_argument("--use-ema", action="store_true", help="Load ema_vae/ema_model from checkpoint")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Accelerate mixed precision mode",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["none", "fp16", "bf16"],
        help="Autocast dtype",
    )
    parser.add_argument(
        "--nf-conditioning",
        type=str,
        default="labels",
        choices=["labels", "none"],
        help="Pass labels to NF forward or not",
    )
    parser.add_argument(
        "--nf-noise-std",
        type=float,
        default=None,
        help="Additive noise std for NF input. If omitted, use config model.nf_trainer.init_args.noise_std",
    )
    parser.add_argument(
        "--nf-noise-gamma",
        type=float,
        default=None,
        help="Additive noise gamma for NF input. If omitted, use config model.nf_trainer.init_args.gamma",
    )
    parser.add_argument(
        "--maetok-pool",
        type=str,
        default="mean",
        choices=["none", "mean", "max", "flatten", "token"],
    )
    parser.add_argument("--maetok-token-index", type=int, default=0)
    parser.add_argument(
        "--save-maetok-pre-proj",
        action="store_true",
        help="Also save MAETok encoder output before the last proj (quant_conv).",
    )
    parser.add_argument(
        "--nf-pool",
        type=str,
        default="mean",
        choices=["none", "mean", "max", "flatten", "token"],
    )
    parser.add_argument("--nf-token-index", type=int, default=0)
    parser.add_argument(
        "--nf-source",
        type=str,
        default="attention",
        choices=["attention", "attn_block"],
        help="Hook position for NF features",
    )
    parser.add_argument(
        "--nf-blocks",
        type=str,
        default="0,3,7",
        help="Comma separated NF block indices, e.g. 0,3,7",
    )
    parser.add_argument(
        "--nf-attn-index",
        type=int,
        default=0,
        help="Which attention layer to select inside each NF block",
    )
    parser.add_argument(
        "--undo-permutation",
        action="store_true",
        help="Apply block.permutation(..., inverse=True) on hooked NF features",
    )
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    cfg = load_yaml(args.config)
    cfg_data = cfg["data"]
    nf_cfg = cfg["model"]["nf_trainer"].get("init_args", {})
    nf_noise_std = float(nf_cfg.get("noise_std", 0.0) if args.nf_noise_std is None else args.nf_noise_std)
    nf_noise_gamma = float(nf_cfg.get("gamma", 1.0) if args.nf_noise_gamma is None else args.nf_noise_gamma)
    data_root = args.data_root or cfg_data["data_root"]
    if data_root.startswith("oss://"):
        raise ValueError("data_root is oss path. Please pass local --data-root for feature extraction.")
    split_root = os.path.join(data_root, args.split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    device = accelerator.device
    vae, nf_model = load_maetok_nf(cfg, args.ckpt, use_ema=args.use_ema, device=device)

    img_size = int(cfg_data["train_image_size"])
    dino_size = int(cfg_data.get("dino_image_size", 0))
    dataset = PixImageNet(root=split_root, resolution=img_size, dino_resolution=dino_size)
    base_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    vae, nf_model, loader = accelerator.prepare(vae, nf_model, base_loader)

    nf_blocks = parse_int_list(args.nf_blocks)
    if len(nf_blocks) == 0:
        raise ValueError("--nf-blocks cannot be empty")

    # hook_nf = accelerator.unwrap_model(nf_model)
    hook_nf = nf_model
    hook_features: Dict[str, torch.Tensor] = {}
    maetok_pre_proj_holder: Dict[str, torch.Tensor] = {}
    handles = []
    if args.save_maetok_pre_proj:
        if not hasattr(vae, "quant_conv"):
            raise AttributeError("Current VAE has no quant_conv, cannot capture pre-proj MAETok features.")

        def maetok_pre_proj_hook(_module, inputs, _output):
            if len(inputs) == 0:
                raise RuntimeError("quant_conv hook received empty inputs.")
            maetok_pre_proj_holder["feat"] = inputs[0].detach()

        handles.append(vae.quant_conv.register_forward_hook(maetok_pre_proj_hook))

    for block_idx in nf_blocks:
        if block_idx < 0:
            block_idx = len(hook_nf.blocks) + block_idx
        if block_idx < 0 or block_idx >= len(hook_nf.blocks):
            raise IndexError(f"NF block index out of range: {block_idx}")
        block = hook_nf.blocks[block_idx]
        if args.nf_attn_index < 0 or args.nf_attn_index >= len(block.attn_blocks):
            raise IndexError(
                f"nf_attn_index={args.nf_attn_index} out of range for block {block_idx} "
                f"(num_attn_blocks={len(block.attn_blocks)})"
            )

        def make_hook(k: str, b):
            def _hook(_module, _inputs, output):
                feat = output
                if args.undo_permutation:
                    feat = b.permutation(feat, inverse=True)
                hook_features[k] = feat.detach()

            return _hook

        if block_idx != 7:
            attn_index = args.nf_attn_index 
            attn_block = block.attn_blocks[attn_index]
            key = f"nf_b{block_idx}_a{attn_index}"

            if args.nf_source == "attention":
                handles.append(attn_block.attention.register_forward_hook(make_hook(key, block)))
            else:
                handles.append(attn_block.register_forward_hook(make_hook(key, block)))
        else:
            for attn_index in [2,6,10,14]:
                attn_block = block.attn_blocks[attn_index]
                key = f"nf_b{block_idx}_a{attn_index}"

                if args.nf_source == "attention":
                    handles.append(attn_block.attention.register_forward_hook(make_hook(key, block)))
                else:
                    handles.append(attn_block.register_forward_hook(make_hook(key, block)))

    out_dir = os.path.join(args.output_root, args.split)
    writer = ShardWriter(
        output_dir=out_dir,
        split=args.split,
        rank=accelerator.process_index,
        shard_size=args.shard_size,
    )
    processed = 0
    pbar = tqdm(loader, desc=f"extract-{args.split}", disable=not accelerator.is_local_main_process)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp_dtype, None)
    local_max_samples = 0
    if args.max_samples > 0:
        local_max_samples = (args.max_samples + accelerator.num_processes - 1) // accelerator.num_processes

    with torch.inference_mode():
        for _, norm_images, labels in pbar:
            labels = labels.long()

            hook_features.clear()
            amp_ctx = (
                torch.autocast(device_type=accelerator.device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else contextlib.nullcontext()
            )
            with amp_ctx:
                maetok_pre_proj_holder.clear()
                encode_out = vae.encode(norm_images)
                maetok_tokens = encode_out[0] if isinstance(encode_out, tuple) else encode_out
                nf_input_tokens = add_nf_additive_noise(maetok_tokens, nf_noise_std, nf_noise_gamma)
                cond_y = labels if args.nf_conditioning == "labels" else None
                _ = nf_model(nf_input_tokens, cond_y, None)

            batch_save = {
                "labels": labels.detach().cpu(),
                "maetok": pool_tokens(maetok_tokens.detach(), args.maetok_pool, args.maetok_token_index).cpu(),
            }
            if args.save_maetok_pre_proj:
                if "feat" not in maetok_pre_proj_holder:
                    raise RuntimeError("Failed to capture MAETok pre-proj feature from quant_conv hook.")
                batch_save["maetok_pre_proj"] = pool_tokens(
                    maetok_pre_proj_holder["feat"],
                    args.maetok_pool,
                    args.maetok_token_index,
                ).cpu()
            for feat_key, feat_tensor in hook_features.items():
                batch_save[feat_key] = pool_tokens(feat_tensor, args.nf_pool, args.nf_token_index).cpu()
            writer.append(batch_save)

            processed += int(labels.shape[0])
            pbar.set_postfix({"samples": processed})
            if local_max_samples > 0 and processed >= local_max_samples:
                break

    for h in handles:
        h.remove()
    writer.flush()
    os.makedirs(out_dir, exist_ok=True)
    rank_manifest = {
        "rank": accelerator.process_index,
        "saved_shards": writer.saved_files,
        "num_samples_local": processed,
    }
    with open(
        os.path.join(out_dir, f"{args.split}_rank{accelerator.process_index:03d}_manifest.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(rank_manifest, f, ensure_ascii=False, indent=2)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_saved = []
        total_local = 0
        for rank in range(accelerator.num_processes):
            manifest_path = os.path.join(out_dir, f"{args.split}_rank{rank:03d}_manifest.json")
            if not os.path.exists(manifest_path):
                continue
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            all_saved.extend(m.get("saved_shards", []))
            total_local += int(m.get("num_samples_local", 0))

        meta = {
            "config": args.config,
            "ckpt": args.ckpt,
            "split": args.split,
            "data_root": data_root,
            "use_ema": bool(args.use_ema),
            "mixed_precision": args.mixed_precision,
            "num_processes": accelerator.num_processes,
            "nf_conditioning": args.nf_conditioning,
            "nf_noise_std": nf_noise_std,
            "nf_noise_gamma": nf_noise_gamma,
            "maetok_pool": args.maetok_pool,
            "maetok_token_index": args.maetok_token_index,
            "save_maetok_pre_proj": bool(args.save_maetok_pre_proj),
            "nf_pool": args.nf_pool,
            "nf_token_index": args.nf_token_index,
            "nf_source": args.nf_source,
            "nf_blocks": nf_blocks,
            "nf_attn_index": args.nf_attn_index,
            "undo_permutation": bool(args.undo_permutation),
            "saved_shards": sorted(all_saved),
            "num_samples_total_approx": total_local,
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[done] saved {len(all_saved)} shards under: {out_dir}")


if __name__ == "__main__":
    main()
