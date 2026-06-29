from typing import Any
import os
import torch
import copy
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from src.data.randn import RandomNDataset
from torch.utils.data import DistributedSampler

class StaticDistributedSampler(DistributedSampler):
    """
    A DistributedSampler that ignores the .set_epoch() call.
    This ensures that the data shuffling order is the same for every epoch,
    which is useful for reproducing the behavior of code that forgets to
    call .set_epoch().

    WARNING: This is generally a bad practice for model training as it
    reduces data randomness and can lead to poorer generalization.
    Use it only for debugging or specific replication purposes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("WARNING: Using StaticDistributedSampler. Data order will be IDENTICAL across epochs.")

    def set_epoch(self, epoch: int) -> None:
        # 关键：重写这个方法，让它什么都不做
        # 默认的 DistributedSampler 会使用 epoch 来改变随机种子
        # 我们让它永远使用初始的 epoch (通常是 0)
        print(f"StaticDistributedSampler: Ignoring set_epoch({epoch}). The shuffle order remains fixed.")
        # super().set_epoch(epoch) # 我们注释掉或删除这一行
        pass

def collate_fn(batch):
    new_batch = copy.deepcopy(batch)
    new_batch = list(zip(*new_batch))
    for i in range(len(new_batch)):
        if isinstance(new_batch[i][0], torch.Tensor):
            try:
                new_batch[i] = torch.stack(new_batch[i], dim=0)
            except:
                print("Warning: could not stack tensors")
    return new_batch

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_root,
                 cache_folder,
                 test_nature_root,
                 test_gen_root,
                 test_batch_size=32,
                 train_image_size=64,
                 dino_image_size=256,
                 train_batch_size=64,
                 train_num_workers=8,
                train_prefetch_factor=2,
                train_dataset: str = None,
                eval_batch_size=32,
                eval_num_workers=4,
                eval_max_num_instances=50000,
                pred_batch_size=32,
                pred_num_workers=4,
                pred_seeds:str=None,
                num_classes=1000,
                latent_shape=(3,64,64),
                val_recon: bool = False,
                val_recon_num_workers: int | None = None,
    ):
        super().__init__()
        pred_seeds = list(map(lambda x: int(x), pred_seeds.strip().split(","))) if pred_seeds is not None else None

        self.data_root = data_root
        self.cache_folder = cache_folder
        self.train_image_size = train_image_size
        self.dino_image_size = dino_image_size
        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor

        self.test_nature_root = test_nature_root
        self.test_gen_root = test_gen_root
        self.test_batch_size = test_batch_size
        self.eval_max_num_instances = eval_max_num_instances
        self.pred_seeds = pred_seeds
        self.num_classes = num_classes
        self.latent_shape = latent_shape

        self.eval_batch_size = eval_batch_size
        self.pred_batch_size = pred_batch_size

        self.pred_num_workers = pred_num_workers
        self.eval_num_workers = eval_num_workers
        self.val_recon = val_recon
        self.val_recon_num_workers = val_recon_num_workers

        self._train_dataloader = None

    def setup(self, stage: str) -> None:
        keys = ["CUDA_VISIBLE_DEVICES","RANK","WORLD_SIZE","LOCAL_RANK","NODE_RANK","MASTER_ADDR","MASTER_PORT","LOCAL_PROCESS_RANK"]
        print({k: os.environ.get(k) for k in keys})
        if stage == "fit":
           assert self.train_dataset is not None

           if self.data_root.startswith("oss://"):
               raise ValueError(
                   "OSS paths are not supported in the open-source data module. "
                   "Please sync the dataset locally and pass --data.data_root /path/to/imagenet."
               )

           if self.train_dataset == "pix_imagenet":
               from src.data.imagenet import PixImageNet
               self.train_dataset = PixImageNet(
                   root=os.path.join(self.data_root, "train"),
                   resolution=self.train_image_size,
                   dino_resolution=self.dino_image_size,
               )
           elif self.train_dataset == "latent_imagenet":
               from src.data.imagenet import LatentDataset
               self.train_dataset = LatentDataset(
                   root=os.path.join(self.data_root, "train"),
                   cache_root=os.path.join(self.data_root, self.cache_folder),
                   resolution=self.dino_image_size,
               )
           else:
               raise NotImplementedError("no such dataset")
           if self.val_recon:
               from src.data.imagenet import PixImageNet
               self.val_recon_dataset = PixImageNet(
                   root=os.path.join(self.data_root, "val"),
                   resolution=self.train_image_size,
                   dino_resolution=self.dino_image_size,
               )

        if self.val_recon and stage == "validate":
            if self.data_root.startswith("oss://"):
               raise ValueError(
                   "OSS paths are not supported in the open-source data module. "
                   "Please sync the dataset locally and pass --data.data_root /path/to/imagenet."
               )
            from src.data.imagenet import PixImageNet
            self.val_recon_dataset = PixImageNet(
                root=os.path.join(self.data_root, "val"),
                resolution=self.train_image_size,
                dino_resolution=self.dino_image_size,
            )

        if stage == "test":
            if self.data_root.startswith("oss://"):
               raise ValueError(
                   "OSS paths are not supported in the open-source data module. "
                   "Please sync the dataset locally and pass --data.data_root /path/to/imagenet."
               )
            from src.data.imagenet import PixImageNet
            self.test_dataset = PixImageNet(
                root=os.path.join(self.data_root, "val"),
                resolution=self.train_image_size,
                dino_resolution=self.dino_image_size,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        # sampler = StaticDistributedSampler(self.train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        self._train_dataloader = DataLoader(
            self.train_dataset,
            self.train_batch_size,
            timeout=6000,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
            sampler=sampler,
            pin_memory=True,
        )
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        self.eval_dataset = RandomNDataset(
            latent_shape=self.latent_shape,
            num_classes=self.num_classes,
            max_num_instances=self.eval_max_num_instances,
        )
        # fixed_noise = torch.randn(self.eval_max_num_instances, *self.latent_shape, device='cpu')
        # fixed_y = torch.arange(self.num_classes).repeat(self.eval_max_num_instances // self.num_classes + 1)[:self.eval_max_num_instances]
        # self.eval_dataset = torch.utils.data.TensorDataset(fixed_noise, fixed_y)
        sampler = DistributedSampler(self.eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        gen_loader = DataLoader(
            self.eval_dataset,
            self.eval_batch_size,
            num_workers=self.eval_num_workers,
            prefetch_factor=2,
            sampler=sampler,
        )
        if not self.val_recon:
            return gen_loader

        if not hasattr(self, "val_recon_dataset"):
            raise RuntimeError("val_recon=True but val_recon_dataset not initialized in setup().")
        recon_sampler = DistributedSampler(
            self.val_recon_dataset, num_replicas=world_size, rank=global_rank, shuffle=False
        )
        recon_loader = DataLoader(
            self.val_recon_dataset,
            self.test_batch_size,
            num_workers=self.val_recon_num_workers or self.eval_num_workers,
            prefetch_factor=2,
            sampler=recon_sampler,
            pin_memory=True,
        )
        return [gen_loader, recon_loader]
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        sampler = DistributedSampler(self.test_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        # sampler = StaticDistributedSampler(self.test_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        self._test_dataloader = DataLoader(
            self.test_dataset,
            self.test_batch_size,
            timeout=6000,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
            sampler=sampler,
            pin_memory=True,
        )
        return self._test_dataloader

    # def predict_dataloader(self) -> EVAL_DATALOADERS:
    #     global_rank = self.trainer.global_rank
    #     world_size = self.trainer.world_size
    #     self.pred_dataset = RandomNDataset(
    #         seeds= self.pred_seeds,
    #         max_num_instances=50000,
    #         num_classes=self.num_classes,
    #         selected_classes=self.pred_selected_classes,
    #         latent_shape=self.latent_shape,
    #     )
    #     from torch.utils.data import DistributedSampler
    #     sampler = DistributedSampler(self.pred_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    #     return DataLoader(self.pred_dataset, batch_size=self.pred_batch_size,
    #                       num_workers=self.pred_num_workers,
    #                       prefetch_factor=4,
    #                       collate_fn=collate_fn,
    #                       sampler=sampler
    #            )
