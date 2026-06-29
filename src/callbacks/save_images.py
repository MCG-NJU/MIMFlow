import lightning.pytorch as pl
from lightning.pytorch import Callback


import os.path
import shutil
import tempfile
import uuid
import numpy
import torch
from PIL import Image
from typing import Sequence, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info

def process_fn(image, path):
    Image.fromarray(image).save(path)

# 验证时 dataloader_idx: 0=生成(samples), 1=重建(reconstructions)
SUBDIR_BY_DATALOADER = {0: "generated", 1: "reconstruction"}


class SaveImagesHook(Callback):
    def __init__(
        self,
        save_dir="val",
        max_save_num: Optional[int] = 50,
        compressed=True,
        save_pngs: bool = True,
        selected_classes: Optional[Sequence[int]] = None,
        per_class_max_save_num: Optional[int] = None,
    ):
       self.save_dir = save_dir
       self.max_save_num = max_save_num
       self.compressed = compressed
       self.save_pngs = save_pngs
       self.selected_classes = set(int(v) for v in selected_classes) if selected_classes else None
       self.per_class_max_save_num = per_class_max_save_num

    def save_start(self, target_dir):
        self.target_dir = target_dir
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_name = self.tmp_dir.name
        self.executor_pool = ThreadPoolExecutor(max_workers=8)
        self.samples = {}  # subdir -> list of arrays
        self._have_saved_num = {}  # subdir -> int
        self._have_saved_num_per_class = {}  # subdir -> {class_id: int}
        rank_zero_info(f"Save images to {self.target_dir} (generated / reconstruction), write via tmp_dir")

    def _subdir_for(self, dataloader_idx: int) -> str:
        return SUBDIR_BY_DATALOADER.get(dataloader_idx, f"dataloader_{dataloader_idx}")

    def save_image(self, images, filenames, target_subdir: str, labels=None):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        count = self._have_saved_num.get(target_subdir, 0)
        per_class_count = self._have_saved_num_per_class.setdefault(target_subdir, {})

        label_list = None
        if not self.save_pngs:
            return
        if labels is not None:
            if torch.is_tensor(labels):
                if labels.ndim == 1:
                    label_list = labels.detach().cpu().tolist()
            else:
                try:
                    label_list = [int(v) for v in labels]
                except Exception:
                    label_list = None

        for idx, (sample, filename) in enumerate(zip(images, filenames)):
            if isinstance(filename, Sequence):
                filename = filename[0]

            label = None
            if label_list is not None and idx < len(label_list):
                label = int(label_list[idx])

            if self.max_save_num is not None and count >= self.max_save_num:
                break
            if self.selected_classes is not None and label not in self.selected_classes:
                continue
            if label is not None and self.per_class_max_save_num is not None:
                cls_cnt = per_class_count.get(label, 0)
                if cls_cnt >= self.per_class_max_save_num:
                    continue
                per_class_count[label] = cls_cnt + 1
            if label is None and self.per_class_max_save_num is not None:
                break

            path = os.path.join(self.tmp_dir_name, target_subdir, filename)
            self.executor_pool.submit(process_fn, sample, path)
            count += 1
        self._have_saved_num[target_subdir] = count

    def process_batch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        samples: STEP_OUTPUT,
        batch: Any,
        dataloader_idx: int = 0,
    ) -> None:
        if samples is None:
            return
        subdir = self._subdir_for(dataloader_idx)
        if subdir not in self.samples:
            self.samples[subdir] = []
            self._have_saved_num[subdir] = 0
            self._have_saved_num_per_class[subdir] = {}
            os.makedirs(os.path.join(self.tmp_dir_name, subdir), exist_ok=True)
        b, c, h, w = samples.shape
        y = None
        metadata = None
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            if dataloader_idx == 0:
                _, y, metadata = batch[:3]
            else:
                _, _, y = batch[:3]
        if torch.is_tensor(metadata):
            metadata = [(f"{uuid.uuid4().hex}.png",) for _ in range(b)]
        elif metadata is None:
            metadata = [(f"{uuid.uuid4().hex}.png",) for _ in range(b)]
        all_samples = pl_module.all_gather(samples).view(-1, c, h, w)
        self.save_image(samples, metadata, subdir, labels=y)
        if trainer.is_global_zero:
            all_samples = all_samples.permute(0, 2, 3, 1).cpu().numpy()
            self.samples[subdir].append(all_samples)

    def save_end(self):
        if self.compressed:
            for subdir, arr_list in self.samples.items():
                if len(arr_list) > 0:
                    samples = numpy.concatenate(arr_list)
                    npz_path = os.path.join(self.tmp_dir_name, subdir, "samples.npz")
                    numpy.savez(npz_path, arr_0=samples)
        self.executor_pool.shutdown(wait=True)
        # 从 tmp_dir 拷贝到目标目录
        if self.target_dir:
            os.makedirs(self.target_dir, exist_ok=True)
            for name in os.listdir(self.tmp_dir_name):
                src = os.path.join(self.tmp_dir_name, name)
                dst = os.path.join(self.target_dir, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        self.tmp_dir.cleanup()
        self.samples = {}
        self._have_saved_num = {}
        self._have_saved_num_per_class = {}
        self.target_dir = None
        self.executor_pool = None

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, f"iter_{trainer.global_step}_cfg{pl_module.cfg}")
        self.save_start(target_dir)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, outputs, batch, dataloader_idx)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_end()

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, "predict")
        self.save_start(target_dir)

    # def on_predict_batch_end(
    #     self,
    #     trainer: "pl.Trainer",
    #     pl_module: "pl.LightningModule",
    #     samples: Any,
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,
    # ) -> None:
    #     return self.process_batch(trainer, pl_module, samples, batch)

    # def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     self.save_end()
