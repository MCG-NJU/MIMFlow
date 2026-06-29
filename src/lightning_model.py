from typing import Callable, Iterable, Any, Optional, Union, Sequence, Dict
import time
import torch
import copy
import os
import math
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.models.vae import BaseVAE, fp2uint8
from src.trainer.base import BaseTrainer
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params
from src.callbacks.simple_ema import SimpleEMA
from src.utils.dist_log_sync import all_reduce_scalar_dict_mean

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]

def count_parameters(model):
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 计算可训练参数量 (requires_grad=True 的参数)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    return total_params, trainable_params

class LightningModel(pl.LightningModule):
    def __init__(self,
                 vae: nn.Module,
                 model: nn.Module,
                 nf_trainer: BaseTrainer,
                 metric: Optional[nn.Module] = None,
                 ema_tracker: Optional[EMACallable] = None,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 stage1_ckpt_path: Optional[str] = None,
                 stage1_strict: bool = False,
                 stage1_init_mode: str = "separate",
                 precompute_metric_data: Optional[Dict[str, str]] = None,
                 recon_metric: Optional[nn.Module] = None,
                 precompute_recon_metric_data: Optional[Dict[str, str]] = None,
                 nan_debug: Optional[Dict[str, Any]] = None,
                 # denoising controls
                 denoising_mode: str = "flow",  # "flow" (current) | "engine" (cond-uncond diff)
                 engine_denoising_chunk_size: int = 16,
                 cfg: float = 0,
                 denoising_lr: float = 1.0,
                 interval_guidance: float = 0.0,
                 save_val_images: bool = False,
                 train_vae: bool = False,
                 train_model: bool = True,
                 vae_train_mode: str = "full",
                 compute_val_recon: bool = False,
                 compute_test_acc: bool = False,
                 fixed_mask_seed: Optional[int] = 0,
                 # mask control for DeTok training
                 mask_strategy: str = "fixed",  # "fixed" | "random"
                 mask_ratio: Optional[float] = 0.0,
                 ):
        super().__init__()
        self.vae = vae
        self.train_vae = train_vae
        self.train_model = train_model
        self.vae_train_mode = vae_train_mode
        self.model = model
        # ckpt = torch.load('/tmp/flowback/imagenet_ema_model_2_1024_8_[2, 2, 2, 2, 2, 2, 2, 20]_0.30_ep80.pth', map_location='cpu', weights_only=True)
        # self.model.load_state_dict(ckpt, strict=False)
        # ckpt = torch.load('/path/to/workspace/train/init_tarflow.pth', map_location='cpu')
        # self.model.load_state_dict(ckpt,strict=False)

        # weights = torch.load('/path/to/workspace/home/models/detok/detok-BB-gamm3.0-m0.7-decoder_tuned.pth', weights_only=False, map_location='cpu')
        # if 'model_ema' in weights:
        #     weights = weights["model_ema"]
        #     msg = self.vae.load_state_dict(weights, strict=False)
        #     print(f"[Tokenizer] Missing keys: {msg.missing_keys}")
        #     print(f"[Tokenizer] Unexpected keys: {msg.unexpected_keys}")
        #     print("[Tokenizer] Loaded EMA tokenizer.")
        # else:
        #     if args.use_ema_tokenizer:
        #         logger.warning("EMA tokenizer is not in the checkpoint, using the model weights")
        #     weights = weights["model"] if "model" in weights else weights
        #     msg = self.vae.load_state_dict(weights, strict=True)
        #     print(f"[Tokenizer] Missing keys: {msg.missing_keys}")
        #     print(f"[Tokenizer] Unexpected keys: {msg.unexpected_keys}")

        if self.train_vae:
            self.ema_vae = copy.deepcopy(self.vae)
        self.ema_model = copy.deepcopy(self.model)
        self.metric = metric
        self.nf_trainer = nf_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.stage1_ckpt_path = stage1_ckpt_path
        self.stage1_strict = stage1_strict
        self.stage1_init_mode = stage1_init_mode
        self.nan_debug = nan_debug or {}
        self.denoising_mode = denoising_mode
        self.engine_denoising_chunk_size = engine_denoising_chunk_size
        self.cfg = cfg
        self.noise_std = nf_trainer.noise_std
        self.denoising_lr = denoising_lr
        self.interval_guidance = interval_guidance
        self.save_val_images = save_val_images
        self.strict_loading = False
        self.fixed_mask_seed = fixed_mask_seed
        self.mask_strategy = mask_strategy
        self.mask_ratio = mask_ratio
        self._mask_viz_logged = False
        self._fixed_mask_cache = None
        self._nan_debug_hooks = []
        self._nan_debug_triggered = False
        self._nan_debug_module_name_map = {}
        if self.mask_strategy == "random":
            self.mask_ratio = -1
        self._maybe_register_fixed_mask()
        # from fvcore.nn import FlopCountAnalysis, parameter_count_table
        # inputs = torch.randn(1, 128, 64)
        # flops = FlopCountAnalysis(self.model, inputs)
        # print(f"Total FLOPs: {flops.total() / 1e9:.2f} G")
        # print(parameter_count_table(model))
        # exit()

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"未使用的参数: {name}")
                
    def _vae_decode(self, latents: torch.Tensor, ids_restore: Any = None) -> torch.Tensor:
        """
        Make VAE decode compatible with both:
        - BaseVAE/LatentVAE: decode(x)
        - TrainableVAE: decode(x, ids_restore=None)
        """
        vae_for_decode = (
            self.ema_vae
            if getattr(self, "train_vae", False) and hasattr(self, "ema_vae")
            else self.vae
        )
        try:
            return vae_for_decode.decode(latents, ids_restore)
        except TypeError:
            return vae_for_decode.decode(latents)
    
    def _maybe_register_fixed_mask(self) -> None:
        """
        若配置为固定 mask，则在初始化时生成一次固定的 ids_restore/mask，
        并注册到 VAE 编码器与 NF（含 EMA）模型中，保证训练与采样使用相同位置。
        """
        if self.mask_strategy != "fixed" or self.mask_ratio is None or self.mask_ratio <= 0:
            return
        encoder = getattr(self.vae, "encoder", None)
        if encoder is None or not hasattr(encoder, "seq_len"):
            return

        seq_len = encoder.seq_len
        len_keep = int(math.ceil(seq_len * (1 - self.mask_ratio)))
        g = torch.Generator()
        if self.fixed_mask_seed is not None:
            g.manual_seed(int(self.fixed_mask_seed))
        noise = torch.rand(seq_len, generator=g)
        ids_shuffle = torch.argsort(noise)
        ids_keep = ids_shuffle[:len_keep]
        ids_restore = torch.argsort(ids_shuffle)
        mask = torch.ones(seq_len, dtype=torch.float)
        mask[:len_keep] = 0
        mask = mask[ids_restore]

        # 注册到 VAE（含 EMA）
        for vae_model in (self.vae, getattr(self, "ema_vae", None)):
            if vae_model is None:
                continue
            enc = getattr(vae_model, "encoder", None)
            if enc is None:
                continue
            setter = getattr(enc, "set_fixed_mask", None)
            if callable(setter):
                setter(ids_keep, ids_restore, mask)
            else:
                enc.fixed_ids_keep = ids_keep
                enc.fixed_ids_restore = ids_restore
                enc.fixed_mask = mask

        # 注册到 NF（含 EMA）
        for nf_model in (self.model, getattr(self, "ema_model", None)):
            if nf_model is None:
                continue
            setter = getattr(nf_model, "set_fixed_mask", None)
            if callable(setter):
                setter(ids_restore)
            elif hasattr(nf_model, "fixed_ids_restore"):
                nf_model.fixed_ids_restore = ids_restore
        grid_size = getattr(encoder, "grid_size", int(math.sqrt(seq_len)))
        try:
            self._fixed_mask_cache = (mask.view(grid_size, grid_size), grid_size)
        except Exception:
            self._fixed_mask_cache = None

    def _set_vae_train_mode(self) -> None:
        if not self.train_vae:
            no_grad(self.vae)
            return
        if self.vae_train_mode == "full":
            return
        if self.vae_train_mode != "decoder":
            raise ValueError(
                f"Unsupported vae_train_mode={self.vae_train_mode!r}, expected 'full' or 'decoder'."
            )

        no_grad(self.vae)
        decoder = getattr(self.vae, "decoder", None)
        post_quant_conv = getattr(self.vae, "post_quant_conv", None)
        for module in (decoder, post_quant_conv):
            if module is None:
                continue
            module.train()
            for param in module.parameters():
                param.requires_grad = True
    
    def setup(self, stage: str):
        seed = torch.initial_seed()
        process_seed = seed + self.trainer.global_rank
        pl.seed_everything(process_seed)
        current_torch_seed = torch.initial_seed()
        print(f"[setup Rank {self.global_rank}] Seed in setup(): {current_torch_seed}")

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        copy_params(src_model=self.model, dst_model=self.ema_model)

        # print(self.metric.device, self.trainer.global_rank, self.trainer.world_size)
        # self.denoiser = torch.compile(self.denoiser)
        # disable grad for conditioner and vae
        if self.train_vae:
            copy_params(src_model=self.vae, dst_model=self.ema_vae)
            no_grad(self.ema_vae)
        self._set_vae_train_mode()
        if not self.train_model:
            no_grad(self.model)
        if self.metric is not None:
            no_grad(self.metric)
        no_grad(self.ema_model)
        self._log_fixed_mask()

    def on_fit_start(self) -> None:
        self._maybe_enable_nan_debug()
        if self.stage1_ckpt_path:
            self._load_stage1_weights(self.stage1_ckpt_path)
        trainer_on_fit_start = getattr(self.nf_trainer, "on_fit_start", None)
        if callable(trainer_on_fit_start):
            trainer_on_fit_start(self.vae)

    def on_fit_end(self) -> None:
        self._disable_nan_debug()

    def _log_fixed_mask(self) -> None:
        """将固定 mask 以图片形式记录到 logger（仅 rank0）。"""
        if self._mask_viz_logged:
            return
        cache = self._fixed_mask_cache
        if cache is None:
            return
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "global_rank", 0) != 0:
            return
        mask_map, grid_size = cache
        try:
            import wandb  # type: ignore
            img = mask_map.detach().cpu().float().numpy()
            logger.experiment.log({"fixed_mask": wandb.Image(img, caption=f"grid={grid_size}")})
            self._mask_viz_logged = True
        except Exception as e:
            self.print(f"[MaskViz] log failed: {e}")

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        if self.train_vae:
            ema_tracker = self.ema_tracker(self.model, self.ema_model)
            ema_tracker2 = self.ema_tracker(self.vae, self.ema_vae)
            return [ema_tracker, ema_tracker2]
        else:
            ema_tracker = self.ema_tracker(self.model, self.ema_model)
            return [ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.model.parameters())
        params_trainer = filter_nograd_tensors(self.nf_trainer.parameters())
        # for name, param in self.nf_trainer.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)
        # print('='*20)
        # print(params_trainer)
        if self.train_vae:
            params_vae = filter_nograd_tensors(self.vae.parameters())
            optimizer: torch.optim.Optimizer = self.optimizer([*params_trainer, *params_denoiser, *params_vae])
        else:
            optimizer: torch.optim.Optimizer = self.optimizer([*params_trainer, *params_denoiser])
        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )

    def _load_stage1_weights(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        init_mode = getattr(self, "stage1_init_mode", "separate")
        if init_mode not in ("separate", "ema", "model"):
            raise ValueError(
                f"Invalid stage1_init_mode={init_mode!r}. Expected 'separate', 'ema', or 'model'."
            )
        need_ema = init_mode in ("separate", "ema")
        need_model = init_mode in ("separate", "model")
        prefixes = []
        if need_model:
            prefixes.extend(["vae.", "model."])
        if need_ema:
            prefixes.extend(["ema_vae.", "ema_model."])
        filtered = {k: v for k, v in state_dict.items() if any(k.startswith(p) for p in prefixes)}
        msg = self.load_state_dict(filtered, strict=self.stage1_strict)
        if init_mode == "ema":
            self.model.load_state_dict(self.ema_model.state_dict(), strict=False)
            if getattr(self, "train_vae", False) and hasattr(self, "ema_vae"):
                self.vae.load_state_dict(self.ema_vae.state_dict(), strict=False)
        elif init_mode == "model":
            self.ema_model.load_state_dict(self.model.state_dict(), strict=False)
            if getattr(self, "train_vae", False) and hasattr(self, "ema_vae"):
                self.ema_vae.load_state_dict(self.vae.state_dict(), strict=False)
        if getattr(self.trainer, "is_global_zero", True):
            self.print(f"[stage1] loaded weights from {ckpt_path}")
            self.print(f"[stage1] missing_keys={msg.missing_keys}")
            self.print(f"[stage1] unexpected_keys={msg.unexpected_keys}")

    def training_step(self, batch, batch_idx):
        raw_images, x, y = batch
        # dic = torch.load(f'../flowback/debug_batch_{self.global_rank}.pt')
        # x = dic['x'].cuda()
        # y = dic['y'].cuda()
        # with torch.no_grad():
        #     x = self.vae.encode(x)
        # self.print('enter train step:', time.time())
        loss = self.nf_trainer(
            self.vae,
            self.model,
            self.ema_model,
            raw_images,
            x,
            y,
            self.current_epoch,
            mask_ratio=self.mask_ratio,
        )

        # Performance: pack all scalar metrics and do ONE all_reduce instead of N reductions.
        # This keeps "all metrics recorded" while cutting down distributed sync overhead.
        # train_metrics = {f"train/{k}": v for k, v in loss.items()}
        train_metrics = all_reduce_scalar_dict_mean(loss)
        # Log per-step only to avoid Lightning auto-appending _epoch/_step suffix pairs.
        self.log_dict(train_metrics, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False)
        # print('=================')
        # print(self.global_rank, loss)
        # exit()
        # self.print('finish train step:', time.time())
        return loss['loss']

    def _maybe_enable_nan_debug(self) -> None:
        if not self.nan_debug.get("enabled", False):
            return
        if not self._nan_debug_rank_allowed():
            return
        if self.nan_debug.get("enable_autograd_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
        self._register_nan_debug_hooks()

    def _disable_nan_debug(self) -> None:
        if not self._nan_debug_hooks:
            return
        for handle in self._nan_debug_hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._nan_debug_hooks = []
        self._nan_debug_module_name_map = {}

    def _nan_debug_rank_allowed(self) -> bool:
        if self.nan_debug.get("all_ranks", False):
            return True
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return True
        return getattr(trainer, "global_rank", 0) == 0

    def _register_nan_debug_hooks(self) -> None:
        roots = self._nan_debug_get_roots()
        if not roots:
            return
        self._nan_debug_module_name_map = self._build_nan_debug_name_map(roots)
        check_forward = self.nan_debug.get("check_forward", True)
        for _, root in roots:
            for _, module in root.named_modules():
                if check_forward:
                    self._nan_debug_hooks.append(
                        module.register_forward_hook(self._nan_debug_forward_hook)
                    )

    def _nan_debug_get_roots(self):
        names = self.nan_debug.get("modules", ["vae", "model", "nf_trainer"])
        roots = []
        for name in names:
            mod = getattr(self, name, None)
            if isinstance(mod, nn.Module):
                roots.append((name, mod))
        return roots

    def _build_nan_debug_name_map(self, roots):
        name_map = {}
        for prefix, root in roots:
            for name, module in root.named_modules():
                full_name = f"{prefix}.{name}" if name else prefix
                name_map[id(module)] = full_name
        return name_map

    def _nan_debug_forward_hook(self, module, inputs, output):
        if self._nan_debug_triggered:
            return
        if self._nan_debug_has_nan(inputs):
            self._nan_debug_report("forward-input", module)
            return
        if self._nan_debug_has_nan(output):
            self._nan_debug_report("forward-output", module)
            return
        if self.nan_debug.get("check_backward", True):
            self._nan_debug_attach_grad_hooks(output, module)

    def _nan_debug_backward_hook(self, module, grad_input, grad_output):
        if self._nan_debug_triggered:
            return
        if self._nan_debug_has_nan(grad_output):
            self._nan_debug_report("backward-grad_output", module)
            return
        if self._nan_debug_has_nan(grad_input):
            self._nan_debug_report("backward-grad_input", module)
            return

    def _nan_debug_attach_grad_hooks(self, output, module: nn.Module) -> None:
        for tensor in self._nan_debug_iter_tensors(output):
            if tensor is None or not torch.is_tensor(tensor):
                continue
            if not tensor.requires_grad:
                continue
            try:
                tensor.register_hook(lambda grad, m=module: self._nan_debug_grad_hook(m, grad))
            except Exception:
                continue

    def _nan_debug_grad_hook(self, module: nn.Module, grad: torch.Tensor):
        if self._nan_debug_triggered:
            return
        if grad is None:
            return
        if not torch.is_tensor(grad):
            return
        if grad.numel() == 0:
            return
        if not torch.isfinite(grad).all().item():
            self._nan_debug_report("backward-grad_output", module)

    def _nan_debug_report(self, stage: str, module: nn.Module) -> None:
        self._nan_debug_triggered = True
        module_name = self._nan_debug_module_name_map.get(id(module), module.__class__.__name__)
        msg = f"[NaNDebug] {stage} NaN/Inf detected at module={module_name}"
        self.print(msg)
        if self.nan_debug.get("fail_fast", True):
            raise RuntimeError(msg)

    def _nan_debug_has_nan(self, obj) -> bool:
        for tensor in self._nan_debug_iter_tensors(obj):
            if tensor is None:
                continue
            if tensor.numel() == 0:
                continue
            if not torch.isfinite(tensor).all().item():
                return True
        return False

    def _nan_debug_iter_tensors(self, obj):
        if torch.is_tensor(obj):
            yield obj
            return
        if isinstance(obj, dict):
            for v in obj.values():
                yield from self._nan_debug_iter_tensors(v)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                yield from self._nan_debug_iter_tensors(v)
            return

    def _check_fixed_ids_restore_sync(self) -> None:
        """
        Compare self.model.fixed_ids_restore across ranks to ensure the fixed mask
        is identical on every process.
        """
        if (
            self.mask_strategy != "fixed"
            or self.mask_ratio is None
            or self.mask_ratio <= 0
        ):
            return
        ids_restore = getattr(self.model, "fixed_ids_restore", None)
        if ids_restore is None or not torch.is_tensor(ids_restore) or ids_restore.numel() == 0:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return

        ids_cpu = ids_restore.detach().to(dtype=torch.long, device="cpu")
        gathered = [None for _ in range(world_size)]  # type: ignore[var-annotated]
        torch.distributed.all_gather_object(gathered, ids_cpu)

        ref = gathered[0]
        mismatched_ranks = [rank for rank, tensor in enumerate(gathered) if not torch.equal(ref, tensor)]
        if mismatched_ranks:
            msg = (
                f"[Rank {self.global_rank}] fixed_ids_restore mismatch across ranks; "
                f"mismatched ranks={mismatched_ranks}, ref_shape={getattr(ref, 'shape', None)}"
            )
            raise RuntimeError(msg)

    def predict_step(self, batch, batch_idx):
        # self._check_fixed_ids_restore_sync()
        noise, y, metadata= batch
        batch_size = noise.size(0)

        if y is not None and isinstance(y, torch.Tensor) and y.dtype != torch.long:
            # keep labels consistent with most class-condition codepaths
            y = y.long()

        ids_restore = None
        if self.mask_strategy == 'fixed' and self.mask_ratio > 0:
            noise, mask, ids_restore, rope_visible= self.vae.encoder.mae_random_masking(noise, self.mask_ratio)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latents = self.ema_model.reverse(
                noise,
                ids_restore,
                y,
                self.cfg,
                attn_temp=1,
                annealed_guidance=True,
                interval_guidance=self.interval_guidance,
            )
        assert isinstance(latents, torch.Tensor)

        if self.denoising_lr > 0 and self.denoising_mode == "engine":
            if y is None:
                raise ValueError("denoising_mode='engine' requires class labels y (conditional/unconditional diff).")
            with torch.enable_grad():
                latents_cpu = latents.cpu()
                denoised_latents = []
                db = int(self.engine_denoising_chunk_size) if self.engine_denoising_chunk_size else batch_size
                db = max(1, min(db, batch_size))
                for start in range(0, batch_size, db):
                    x = torch.clone(latents_cpu[start: start + db]).detach().cuda()
                    x.requires_grad_(True)
                    y_ = y[start: start + db]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        z_c, _, logdets_c = self.ema_model(x, y_)
                        z_u, _, logdets_u = self.ema_model(x, None)
                    loss = self.ema_model.get_loss(z_c, logdets_c) - self.ema_model.get_loss(z_u, logdets_u)
                    grad = torch.autograd.grad(loss, [x])[0]
                    # engine.py uses a fixed per-sample scale (16*16*64*16) for ImageNet-64;
                    # generalize it to the current latent tensor shape.
                    scale = x.numel() * self.noise_std ** 2
                    x.data.add_(grad, alpha=-self.denoising_lr * scale)
                    denoised_latents.append(x.detach().cpu())
                latents = torch.cat(denoised_latents, dim=0).cuda()
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return self._vae_decode(latents, ids_restore)

        if self.denoising_lr > 0 and self.denoising_mode == "flow" and self.noise_std > 0:
            with torch.enable_grad():
                latents_cpu = latents.cpu()
                # This should be the theoretical optimal denoising lr
                base_lr = self.denoising_lr * noise.numel() * self.noise_std ** 2 / noise.shape[0]
                denoised_latents = []
                # keep the original "split into 4 chunks" idea, but make it robust for small/odd batch sizes
                chunk = max(1, batch_size // 4)
                for start in range(0, batch_size, chunk):
                    x = torch.clone(latents_cpu[start: start + chunk]).detach().cuda()
                    x.requires_grad_(True)
                    if x.shape[0] == 0:
                        continue
                    lr = base_lr * x.shape[0]
                    y_ = y[start: start + chunk] if y is not None else None
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        z, _, logdets = self.ema_model(x, y_)
                    loss = self.ema_model.get_loss(z, logdets)
                    grad = torch.autograd.grad(loss, [x])[0]
                    x.data.add_(grad, alpha=-lr)
                    denoised_latents.append(x.detach().cpu())
                latents = torch.cat(denoised_latents, dim=0).cuda()
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return self._vae_decode(latents, ids_restore)

        if self.denoising_lr > 0 and self.denoising_mode not in ("flow", "engine"):
            raise ValueError(
                f"Unknown denoising_mode={self.denoising_mode!r}. Expected 'flow' or 'engine'."
            )

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return self._vae_decode(latents, ids_restore)

    def on_validation_epoch_start(self) -> None:
        pass
        # self.max_save_images_num = 1000
        # # setup image saver
        # save_dir = os.path.join(self.save_dir, f"epoch_{self.current_epoch}")
        # self.image_saver = ImageSaver(save_dir,
        #         rank=self.trainer.global_rank,
        #         max_save_num=self.max_save_images_num
        # )

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
        if self.nf_trainer.noise_mode == 'add':
            return latents + noise_level_tensor * noise
        elif self.nf_trainer.noise_mode == 'linear':
            return (1 - t) * latents + t * noise
        elif self.nf_trainer.noise_mode == 'slerp':
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

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            samples = self.predict_step(batch, batch_idx)
            samples = fp2uint8(samples)
            return samples

        return None

    def on_test_epoch_start(self) -> None:
        pass

    def test_step(self, batch, batch_idx):
        """Return reconstructions only; metrics are computed outside this repo."""
        raw_images, x, y = batch

        # 选择推理用的 VAE（如有 EMA 优先用 EMA）
        vae_for_eval = self.ema_vae if getattr(self, "train_vae", False) and hasattr(self, "ema_vae") else self.vae

        # 根据配置决定是否传入 mask_ratio
        encode_kwargs = {}
        if self.mask_strategy == "fixed" and self.mask_ratio is not None:
            encode_kwargs["mask_ratio"] = self.mask_ratio

        # 编码阶段
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            try:
                encode_out = vae_for_eval.encode(x, **encode_kwargs)
            except TypeError:
                # 兼容不支持 mask_ratio 的 VAE
                encode_out = vae_for_eval.encode(x)

        ids_restore = None
        if isinstance(encode_out, tuple):
            if len(encode_out) >= 3:
                latents, _, ids_restore = encode_out[:3]
            elif len(encode_out) == 2:
                latents, ids_restore = encode_out
            else:
                latents = encode_out[0]
        else:
            latents = encode_out
        latents = self._add_token_noise(latents)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            try:
                reconstructions = vae_for_eval.decode(latents, ids_restore)
            except TypeError:
                reconstructions = vae_for_eval.decode(latents)

        return reconstructions

    def on_validation_epoch_end(self) -> None:
        self.trainer.strategy.barrier()
        torch.cuda.empty_cache()

    def on_test_epoch_end(self) -> None:
        self.trainer.strategy.barrier()
        torch.cuda.empty_cache()

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        if self.train_vae:
            self.vae.state_dict(
                destination=destination,
                prefix=prefix + "vae.",
                keep_vars=keep_vars)
            self.ema_vae.state_dict(
                destination=destination,
                prefix=prefix + "ema_vae.",
                keep_vars=keep_vars
            )
        self.model.state_dict(
            destination=destination,
            prefix=prefix + "model.",
            keep_vars=keep_vars)
        self.ema_model.state_dict(
            destination=destination,
            prefix=prefix + "ema_model.",
            keep_vars=keep_vars)
        self.nf_trainer.state_dict(
            destination=destination,
            prefix=prefix + "nf_trainer.",
            keep_vars=keep_vars)
        return destination
