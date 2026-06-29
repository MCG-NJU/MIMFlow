from __future__ import annotations

from typing import Dict, Tuple

import torch


@torch.no_grad()
def all_reduce_scalar_dict_mean(
    values: Dict[str, torch.Tensor],
    *,
    group=None,
) -> Dict[str, torch.Tensor]:
    """
    Reduce a dict of scalar tensors across ranks with ONE all_reduce (mean).

    This is a performance-oriented alternative to calling a reduce for every metric key.
    It assumes each value is a scalar (0-dim tensor) or a tensor with exactly 1 element.
    """
    if not values:
        return {}

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return values

    world_size = torch.distributed.get_world_size(group=group)
    if world_size <= 1:
        return values

    # keep deterministic order
    keys = sorted(values.keys())

    tensors = []
    # Track devices to surface mixed-device inputs early
    devices: Dict[str, list[str]] = {}
    for k in keys:
        v = values[k]
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        v = v.detach()
        if v.numel() != 1:
            raise ValueError(f"all_reduce_scalar_dict_mean expects scalar tensors, got {k} with shape {tuple(v.shape)}")
        device_str = str(v.device)
        devices.setdefault(device_str, []).append(k)
        tensors.append(v.reshape(1).to(dtype=torch.float32))

    if len(devices) > 1:
        detail = "; ".join(f"{dev}: {', '.join(keys)}" for dev, keys in devices.items())
        raise RuntimeError(f"all_reduce_scalar_dict_mean expects all tensors on the same device, but got mixed devices -> {detail}")

    stacked = torch.cat(tensors, dim=0)
    torch.distributed.all_reduce(stacked, op=torch.distributed.ReduceOp.SUM, group=group)
    stacked.div_(float(world_size))

    out: Dict[str, torch.Tensor] = {}
    for i, k in enumerate(keys):
        out[k] = stacked[i]
    return out


