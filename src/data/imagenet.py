import torch
from PIL import Image
import numpy as np
import torchvision as tv
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize

class CenterCrop:
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        def center_crop_arr(pil_image, image_size):
            """
            Center cropping implementation from ADM.
            https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
            """
            while min(*pil_image.size) >= 2 * image_size:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.BOX
                )

            scale = image_size / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )

            arr = np.array(pil_image)
            crop_y = (arr.shape[0] - image_size) // 2
            crop_x = (arr.shape[1] - image_size) // 2
            return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

        return center_crop_arr(image, self.size)

class LatentDataset(ImageFolder):
    def __init__(self, root, cache_root, resolution=256):
        super().__init__(root)
        self.transform = CenterCrop(resolution)
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.cache_root = cache_root

    def load_latent(self, latent_path, flip=False):
        pk_data = torch.load(latent_path)
        if flip:
            mean = pk_data['fliped_mean']
            logvar = pk_data['fliped_logvar']
        else:
            mean = pk_data['mean'].to(torch.float32)
            logvar = pk_data['logvar'].to(torch.float32)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        latent = mean + torch.randn_like(mean) * std
        return latent

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        latent_path = image_path.replace(self.root, self.cache_root) + ".pt"

        raw_image = Image.open(image_path).convert('RGB')
        raw_image = self.transform(raw_image)
        raw_image = to_tensor(raw_image)
        flip = False
        # flip = True if np.random.rand() > 0.5 else False
        # if flip:
        #     raw_image = raw_image.flip(-1)
        if self.cache_root is not None:
            latent = self.load_latent(latent_path, flip)
        else:
            latent = self.normalize(raw_image)
        return raw_image, latent, target

class PixImageNet(ImageFolder):
    def __init__(self, root, resolution=256, dino_resolution=0):
        super().__init__(root)
        self.transform = CenterCrop(resolution)
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if dino_resolution > 0:
            self.dino_transform = tv.transforms.Compose(
                [
                    CenterCrop(dino_resolution),
                    tv.transforms.ToTensor(),
                ]
            )
        else:
            self.dino_transform = None

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        raw_image = Image.open(image_path).convert('RGB')
        image = self.transform(raw_image)
        image = to_tensor(image)

        normalized_image = self.normalize(image)
        if self.dino_transform is not None:
            raw_image = self.dino_transform(raw_image)
        else:
            raw_image = image
        # if np.random.rand() > 0.5:
        #     raw_image = raw_image.flip(-1)
        #     normalized_image = normalized_image.flip(-1)
        return raw_image, normalized_image, target
