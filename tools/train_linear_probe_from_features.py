import argparse
import glob
import json
import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

'''
maetok,nf_b0_a0,nf_b2_a0,nf_b4_a0,nf_b6_a0,nf_b7_a6
accelerate launch --num_processes 8 tools/train_linear_probe_from_features.py \
  --feature-root /tmp/features/highmask-fullimage-nocondition-gamma0-before-quant \
  --feature-keys maetok_pre_proj \
  --num-classes 1000 \
  --epochs 90 \
  --batch-size 2048 \
  --val-batch-size 2048 \
  --optimizer sgd \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --use-bn \
  --standardize \
  --save-dir ./linear_probe_ckpt/maetok_nf_probe
'''

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_keys(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def list_shards(root: str, split: str) -> List[str]:
    split_dir = os.path.join(root, split)
    paths = sorted(glob.glob(os.path.join(split_dir, f"{split}_rank*_shard_*.pt")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No shard file found under {split_dir}")
    return paths


def compose_features(shard: Dict[str, torch.Tensor], feature_keys: List[str], normalize_3d: bool) -> torch.Tensor:
    parts = []
    for key in feature_keys:
        if key not in shard:
            raise KeyError(f"feature key '{key}' not found in shard. Available keys: {list(shard.keys())}")
        feat = shard[key]
        if normalize_3d and feat.dim() > 2:
            feat = feat.flatten(1)
        parts.append(feat.float())
    return torch.cat(parts, dim=1)


def build_head(input_dim: int, num_classes: int, use_bn: bool, dropout: float) -> nn.Module:
    layers = []
    print('build head:', input_dim, use_bn, dropout)
    # exit()
    if use_bn:
        layers.append(nn.BatchNorm1d(input_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(input_dim, num_classes))
    return nn.Sequential(*layers)


@torch.no_grad()
def estimate_mean_std(train_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0)
    var = ((train_x - mean) * (train_x - mean)).mean(dim=0)
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return mean, std


@torch.no_grad()
def load_split_tensors(
    shard_paths: List[str], feature_keys: List[str], flatten_3d: bool, desc: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for path in tqdm(shard_paths, desc=desc):
        shard = torch.load(path, map_location="cpu")
        x = compose_features(shard, feature_keys, normalize_3d=flatten_3d)
        y = shard["labels"].long()
        features.append(x)
        labels.append(y)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def run_eval(
    accelerator: Accelerator,
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_correct = torch.tensor(0, device=accelerator.device)
    total_samples = torch.tensor(0, device=accelerator.device)
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            pred = logits.argmax(dim=1)
            bs = yb.shape[0]
            total_loss += loss.detach() * bs
            total_correct += (pred == yb).sum()
            total_samples += bs
    total_loss = accelerator.gather_for_metrics(total_loss).sum()
    total_correct = accelerator.gather_for_metrics(total_correct).sum()
    total_samples = accelerator.gather_for_metrics(total_samples).sum()
    avg_loss = (total_loss / total_samples.clamp_min(1)).item()
    acc = (total_correct.float() / total_samples.clamp_min(1)).item()
    return float(avg_loss), float(acc)


def main():
    parser = argparse.ArgumentParser(description="Stage-2: train linear probe on extracted features")
    parser.add_argument("--feature-root", type=str, required=True, help="Root dir from stage-1 output")
    parser.add_argument(
        "--feature-keys",
        type=str,
        required=True,
        help="Comma separated feature names, e.g. maetok,nf_b3_a0,nf_b7_a0",
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=2048)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--use-bn", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--flatten-3d",
        action="store_true",
        help="Flatten [B,T,C] to [B,T*C]. Set if stage-1 pool=none",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Fit train-set feature mean/std and apply z-score before linear head",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda / cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./linear_probe_ckpt")
    parser.add_argument("--eval-interval", type=int, default=10, help="Run validation every N epochs")
    args = parser.parse_args()

    accelerator = Accelerator(cpu=args.device == "cpu")
    set_seed(args.seed)
    device = accelerator.device
    feature_keys = parse_keys(args.feature_keys)
    train_shards = list_shards(args.feature_root, "train")
    val_shards = list_shards(args.feature_root, "val")

    train_x, train_y = load_split_tensors(
        train_shards,
        feature_keys,
        flatten_3d=args.flatten_3d,
        desc="load-train-features" if accelerator.is_local_main_process else "load-train-features-worker",
    )
    val_x, val_y = load_split_tensors(
        val_shards,
        feature_keys,
        flatten_3d=args.flatten_3d,
        desc="load-val-features" if accelerator.is_local_main_process else "load-val-features-worker",
    )

    input_dim = int(train_x.shape[1])
    model = build_head(input_dim, args.num_classes, use_bn=args.use_bn, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mean = None
    std = None
    if args.standardize:
        mean, std = estimate_mean_std(train_x)
        train_x = (train_x - mean) / std
        val_x = (val_x - mean) / std

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    best_acc = 0.0
    last_val_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0, device=device)
        total_samples = torch.tensor(0, device=device)

        pbar = tqdm(
            train_loader,
            desc=f"epoch-{epoch}-train",
            disable=not accelerator.is_local_main_process,
        )
        for xb, yb in pbar:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                bs = yb.shape[0]
                total_correct += (pred == yb).sum()
                total_loss += loss.detach() * bs
                total_samples += bs

        scheduler.step()
        gathered_loss = accelerator.gather_for_metrics(total_loss).sum()
        gathered_correct = accelerator.gather_for_metrics(total_correct).sum()
        gathered_samples = accelerator.gather_for_metrics(total_samples).sum()
        train_loss = (gathered_loss / gathered_samples.clamp_min(1)).item()
        train_acc = (gathered_correct.float() / gathered_samples.clamp_min(1)).item()

        should_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch == args.epochs - 1) or (epoch == 0)
        if should_eval:
            val_loss, val_acc = run_eval(
                accelerator=accelerator,
                model=model,
                val_loader=val_loader,
                criterion=criterion,
            )
            last_val_acc = val_acc
        else:
            val_loss, val_acc = None, None

        log_row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(log_row)
        if accelerator.is_main_process:
            if should_eval:
                print(
                    f"[epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}"
                )
            else:
                print(
                    f"[epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val=skipped, lr={optimizer.param_groups[0]['lr']:.6f}"
                )

            unwrapped_model = model
            ckpt = {
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "feature_keys": feature_keys,
                "input_dim": input_dim,
                "epoch": epoch,
                "val_acc": val_acc if val_acc is not None else last_val_acc,
                "history": history,
            }
            if mean is not None and std is not None:
                ckpt["feature_mean"] = mean.cpu()
                ckpt["feature_std"] = std.cpu()

            torch.save(ckpt, os.path.join(args.save_dir, "latest.pt"))
            if should_eval and val_acc is not None and val_acc >= best_acc:
                best_acc = val_acc
                torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

    if accelerator.is_main_process:
        with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[done] best val acc = {best_acc:.4f}, checkpoints in: {args.save_dir}")


if __name__ == "__main__":
    main()
