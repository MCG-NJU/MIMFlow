import time
import glob
from typing import Any, Union, List, Tuple

# from src.utils.patch_bugs import *

import os
import torch
from lightning import Trainer, LightningModule, seed_everything
from src.lightning_data import DataModule
from src.lightning_model import LightningModel
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
import lightning.pytorch as pl

import logging
logger = logging.getLogger("lightning.pytorch")

class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime('%y%m%d%H%M')
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        # 确保保存目录存在（部分 launcher 环境不会提前创建）
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.parser.save(
            self.config, file_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )

class ReWriteRootDirCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp:str, std:float):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")
        parser.add_argument('--cfg_range', type=List[Union[int, float]], default=None,
                            help='List of numeric values (int or float) to validate.')
        parser.add_argument('--ckpt_dir', type=str, default=None,
                            help='Directory containing checkpoints to evaluate sequentially.')
        parser.add_argument('--ckpt_glob', type=str, default="*.ckpt",
                            help='Glob pattern for checkpoints inside ckpt_dir (default: *.ckpt).')
        parser.add_argument('--ckpt_limit', type=int, default=None,
                            help='Optional max number of checkpoints to evaluate (after sorting).')
        parser.add_argument('--ckpt_sort', type=str, default='mtime', choices=['mtime', 'name'],
                            help='How to sort checkpoints when scanning ckpt_dir.')
    
    def before_instantiate_classes(self) -> None:
        """在实例化类之前配置路径和 logger"""
        # 获取 trainer 配置
        subcommand = self.config.get("subcommand")

        # 获取当前子命令的配置
        if subcommand:
            config = self.config[subcommand]
        else:
            config = self.config

        # 获取 trainer 配置
        config_trainer = config.get("trainer", {})

        # 1. 构建实验目录名称
        tags = self._get(self.config, "tags", default={})
        dirname = "_".join([f"{k}_{v}" for k, v in tags.items()]) if tags else "default_exp"

        # 2. 设置 default_root_dir
        base_dir = config_trainer.get("default_root_dir", os.path.join(os.getcwd(), "workdirs"))
        # 评估/预测使用单独的日志目录，避免与训练混在一起
        is_eval_like = subcommand in ("validate", "test", "predict")
        default_root_dir = os.path.join(base_dir, 'test', dirname) if is_eval_like else os.path.join(base_dir, dirname)

        # # 3. 检查目录是否存在（仅在 fit 时检查）
        # is_resume = self._get(self.config_init, "ckpt_path", default=None)
        # if os.path.exists(default_root_dir) and "debug" not in default_root_dir:
        #     if (os.listdir(default_root_dir) and
        #         self.subcommand not in ["predict", "validate"] and
        #         not is_resume):
        #         raise FileExistsError(f"{default_root_dir} already exists")

        # 4. 更新 trainer 的 default_root_dir
        config_trainer["default_root_dir"] = default_root_dir

        # 5. 配置 TensorBoardLogger
        # 组织清晰的日志目录结构：
        # workdirs/
        # └── exp_name/
        #     ├── logs/           # TensorBoard 日志
        #     ├── checkpoints/    # 模型检查点
        #     └── configs/        # 配置文件

        if "logger" in config_trainer:
            logger_config = config_trainer["logger"]

            # 如果是 TensorBoardLogger
            if logger_config.get("class_path", "").endswith("TensorBoardLogger"):
                init_args = logger_config.get("init_args", {})

                # 设置 save_dir 为实验根目录（训练或评估已区分）
                init_args["save_dir"] = default_root_dir

                # 设置 name 为 'logs'，这样日志会保存在 default_root_dir/logs/version_x
                init_args["name"] = "logs"

                # 设置版本号（可选）
                if "version" not in init_args:
                    init_args["version"] = None  # 自动递增版本号

                logger_config["init_args"] = init_args

        # # 创建目录（如果是主进程）
        # if self.trainer is None or getattr(self.trainer, 'is_global_zero', True):
        #     os.makedirs(default_root_dir, exist_ok=True)
        #     os.makedirs(os.path.join(default_root_dir, "logs"), exist_ok=True)
        #     os.makedirs(os.path.join(default_root_dir, "checkpoints"), exist_ok=True)

        # logger.info(f"Experiment directory: {default_root_dir}")
        # logger.info(f"  - Logs: {os.path.join(default_root_dir, 'logs')}")
        # logger.info(f"  - Checkpoints: {os.path.join(default_root_dir, 'checkpoints')}")

    def _run_subcommand(self, subcommand: str):
        # 捕获 validate/test 子命令的特殊情况
        print('=' * 80)
        self.subcommand = subcommand
        sub_cfg = self.config[subcommand]
        print('cfg range:', getattr(sub_cfg, "cfg_range", None))
        has_ckpt_scan = (
            getattr(sub_cfg, "ckpt_dir", None) is not None
            or (getattr(sub_cfg, "ckpt_path", None) and os.path.isdir(sub_cfg.ckpt_path))
        )
        if subcommand in ("validate", "test") and has_ckpt_scan:
            self._run_multi_ckpt_eval(subcommand)
        elif subcommand == 'validate' and getattr(sub_cfg, "cfg_range", None):
            self._run_multi_validation()
        else:
            # 对于其他所有情况，使用默认行为
            super()._run_subcommand(subcommand)

    def _run_multi_validation(self):
        config = self.config['validate']
        cfg_range = config.cfg_range

        logging.info(f"Starting multi-validation for cfg with range {cfg_range}")

        ckpt_path = self.config[self.subcommand].ckpt_path
        if not ckpt_path:
            raise ValueError("`--ckpt_path` is required for multi-validation.")

        import numpy as np
        for value in np.linspace(*cfg_range):
            print("=" * 70)
            print(value)
            logging.info(f"Running validation for cfg = {value}")
            self.model.cfg = value

            self.trainer.validate(model=self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)
            # 将 cfg 作为横轴，使用 log_metrics 直接记录，前缀改为 cfg_scan/ 避免与默认 val 指标混淆
            metrics = {}
            for k, v in self.trainer.callback_metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[f"cfg_scan/{k}"] = v.detach().cpu().item()
                else:
                    metrics[f"cfg_scan/{k}"] = float(v)
            if self.trainer.logger is not None:
                self.trainer.logger.log_metrics(metrics, step=10*value)
            print("=" * 70 + "\n")

    def _resolve_ckpt_paths(self, config: Any) -> Tuple[List[str], str, str]:
        """解析需要评估的 checkpoint 列表。"""
        ckpt_dir = getattr(config, "ckpt_dir", None)
        ckpt_path = getattr(config, "ckpt_path", None)
        search_dir = ckpt_dir if ckpt_dir is not None else (ckpt_path if ckpt_path and os.path.isdir(ckpt_path) else None)
        if search_dir is None:
            return [], "", ""

        ckpt_glob = getattr(config, "ckpt_glob", "*.ckpt") or "*.ckpt"
        pattern = os.path.join(search_dir, ckpt_glob)
        ckpt_paths = sorted(glob.glob(pattern))

        ckpt_sort = getattr(config, "ckpt_sort", "mtime")
        if ckpt_sort == "mtime":
            ckpt_paths = sorted(ckpt_paths, key=os.path.getmtime)
        elif ckpt_sort == "name":
            ckpt_paths = sorted(ckpt_paths)

        ckpt_limit = getattr(config, "ckpt_limit", None)
        if ckpt_limit is not None:
            ckpt_paths = ckpt_paths[:ckpt_limit]

        return ckpt_paths, search_dir, pattern

    def _log_metrics_with_prefix(self, prefix: str, step: float) -> None:
        """为多轮评估打上前缀，避免覆盖默认 val/test 指标。"""
        if self.trainer.logger is None:
            return
        metrics = {}
        for k, v in self.trainer.callback_metrics.items():
            key = f"{prefix}/{k}"
            if isinstance(v, torch.Tensor):
                metrics[key] = v.detach().cpu().item()
            else:
                try:
                    metrics[key] = float(v)
                except Exception:
                    continue
        if metrics:
            self.trainer.logger.log_metrics(metrics, step=step)

    def _run_multi_ckpt_eval(self, subcommand: str):
        """遍历目录下的 ckpt 逐个进行 validate/test。"""
        config = self.config[subcommand]
        ckpt_paths, search_dir, pattern = self._resolve_ckpt_paths(config)
        if len(ckpt_paths) == 0:
            raise FileNotFoundError(f"No checkpoints found under '{search_dir}' with pattern '{pattern}'. "
                                    "请确保提供 --ckpt_dir 或 --ckpt_path=目录。")

        logging.info(f"Starting multi-{subcommand} over {len(ckpt_paths)} checkpoints from {search_dir} (pattern={pattern})")
        run_fn = self.trainer.validate if subcommand == "validate" else self.trainer.test
        cfg_range = getattr(config, "cfg_range", None)

        import numpy as np
        for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
            print("=" * 70)
            print(f"[{subcommand}] ckpt {ckpt_idx + 1}/{len(ckpt_paths)}: {ckpt_path}")

            if cfg_range:
                for cfg_idx, cfg_value in enumerate(np.linspace(*cfg_range)):
                    print(f"  - cfg = {cfg_value}")
                    logging.info(f"Running {subcommand} for cfg={cfg_value} ckpt={ckpt_path}")
                    self.model.cfg = cfg_value
                    run_fn(model=self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)
                    self._log_metrics_with_prefix("ckpt_scan", step=ckpt_idx * 1000 + cfg_idx)
            else:
                logging.info(f"Running {subcommand} for ckpt={ckpt_path}")
                run_fn(model=self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)
                self._log_metrics_with_prefix("ckpt_scan", step=ckpt_idx)
            print("=" * 70 + "\n")

if __name__ == "__main__":
    cli = ReWriteRootDirCli(LightningModel, DataModule,
                       auto_configure_optimizers=False,
                       save_config_callback=ReWriteRootSaveConfigCallback,
                       save_config_kwargs={"overwrite": True})
