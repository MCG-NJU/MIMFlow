import os.path
import random

import torch
from torch.utils.data import Dataset



class RandomNDataset(Dataset):
    def __init__(self, latent_shape=(4, 64, 64), num_classes=1000, selected_classes:list=None, seeds=None, max_num_instances=50000, ):
        self.selected_classes = selected_classes
        if selected_classes is not None:
            num_classes = len(selected_classes)
            max_num_instances = 10*num_classes
        self.num_classes = num_classes
        self.seeds = seeds
        if seeds is not None:
            self.max_num_instances = len(seeds)*num_classes
            self.num_seeds = len(seeds)
        else:
            self.num_seeds = (max_num_instances + num_classes - 1)  // num_classes
            self.max_num_instances = self.num_seeds*num_classes

        self.latent_shape = latent_shape


    def __getitem__(self, idx):
        label = idx // self.num_seeds
        if self.selected_classes:
            label = self.selected_classes[label]
        seed = random.randint(0, 1<<31) #idx % self.num_seeds
        if self.seeds is not None:
            seed = self.seeds[idx % self.num_seeds]

        # cls_dir = os.path.join(self.root, f"{label}")
        filename = f"{label}_{seed}.png",
        generator = torch.Generator().manual_seed(seed)
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)
        return latent, label, filename
    def __len__(self):
        return self.max_num_instances
