from src.trainer.base import *
import torch
import copy
import timm
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from src.utils.no_grad import no_grad

class DINOv2(nn.Module):
    def __init__(self, weight_path:str):
        super(DINOv2, self).__init__()
        self.encoder = torch.hub.load('/path/to/workspace/home/.cache/torch/hub/facebookresearch_dinov2_main', weight_path, source='local', skip_validation=True)
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()

    def fetch_pos(self, h, w):
        key = (h, w)
        if key in self.precomputed_pos_embed:
            return self.precomputed_pos_embed[key]
        value = timm.layers.pos_embed.resample_abs_pos_embed(
            self.pos_embed.data, [h, w],
        )
        self.precomputed_pos_embed[key] = value
        return value

    def forward(self, x):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, (int(224*h/256), int(224*w/256)), mode='bicubic')
        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        pos_embed_data = self.fetch_pos(patch_num_h, patch_num_w)
        self.encoder.pos_embed.data = pos_embed_data
        feature = self.encoder.forward_features(x)['x_norm_patchtokens']
        return feature

class RepaTrainer(BaseTrainer):
    def __init__(
            self,
            align_blocks: list[int] = [1, 2, 3, 4],
            align_attn_layer: int = 1,
            repa_type: str = 'reverse',
            repa_loss_weight: float = 0.1,
            num_proj: int = 1,
            proj_model_dim=1024,
            proj_hidden_dim=1024,
            proj_encoder_dim=768,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.align_blocks = align_blocks
        self.align_attn_layer = align_attn_layer
        self.repa_type = repa_type
        self.num_proj = num_proj
        self.blocks_per_projector = len(self.align_blocks) // num_proj
        self.repa_loss_weight = repa_loss_weight
        self.encoder = DINOv2('dinov2_vitb14')
        no_grad(self.encoder)

        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_model_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            ) for _ in range(self.num_proj)
        ])

    def _impl_trainstep(self, vae, net, ema_net, raw_images, x, y, epoch, mask_ratio=None):
        x = vae.encode(x)
        def create_hook(feature_list):
            def hook(module, input, output):
                feature_list.append(module.permutation(output))
            return hook

        src_features = []
        handles = []
        for block_idx in self.align_blocks:
            handles.append(net.blocks[block_idx - 1].attn_blocks[self.align_attn_layer - 1].register_forward_hook(create_hook(src_features)))

        z, outputs, logdets = net(x, y)
        loss_nf = net.get_loss(z, logdets)

        if self.repa_type == 'detach':
            src_features.clear()
            net.repa_forward(y, outputs, self.align_blocks)
        elif self.repa_type == 'reverse':
            src_features.clear()
            net.repa_reverse(outputs, y, self.align_blocks)
        elif self.repa_type == 'detach_forward':
            src_features.clear()
            net.detach_forward(y, outputs, self.align_blocks)
        assert len(src_features) == len(self.align_blocks)
        # print(len(args.align_blocks), len(projectors_ddp))
        with torch.no_grad():
            dst_feature = self.encoder(raw_images)  # [32, 256, 768]
            # print(dst_feature.shape)

        repa_loss = 0
        for i, sf in enumerate(src_features):
            # 使用对应的projector处理每个source feature
            sf_projected = self.projectors[i // self.blocks_per_projector](sf)
            cos_sim = torch.nn.functional.cosine_similarity(sf_projected, dst_feature, dim=-1)
            cos_loss = 1 - cos_sim
            repa_loss += cos_loss.mean()
        repa_loss = repa_loss / len(src_features)

        loss = loss_nf + self.repa_loss_weight * repa_loss
        for handle in handles:
            handle.remove()
        out = dict(
            loss=loss,
            repa_loss=repa_loss,
            loss_nf=loss_nf,
            logdets=logdets.mean(),
        )
        return out
