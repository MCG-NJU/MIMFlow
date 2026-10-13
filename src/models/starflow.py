import torch
from src.models.attention import *
import torch.nn.functional as F
from src.models.layers.permutation import Permutation, PermutationIdentity, PermutationFlip

class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
        tanh_alpha: float = 20.0,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        if num_classes:
            self.class_embed = torch.nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2)
        else:
            self.class_embed = None
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion, permutation) for _ in range(num_layers)]
        )
        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = torch.nn.Linear(channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))
        self.tanh_alpha = tanh_alpha

    def classify_forward(self, x: torch.Tensor, logits: torch.Tensor, freq_cis: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x)
        x_in = x
        x = self.proj_in(x)
        freq_cis = self.permutation(freq_cis, dim=0)

        class_weights = self.class_embed.squeeze(1)  # (1000, C)
        class_features = torch.matmul(logits, class_weights)  # (B, 1000) @ (1000, C) = (B, C)
        class_features = class_features.unsqueeze(1)
        x = x + class_features

        for block in self.attn_blocks:
            x = block(x, freq_cis, self.attn_mask)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        a = torch.tensor(self.tanh_alpha)
        xa = a * F.tanh(xa / a)
        # xa = torch.clip(xa, -80, None)
        scale = (-xa.float()).exp().type(xa.dtype)
        # scale = F.softplus(-xa)
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, freq_cis: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x) # [B, T, C]
        x_in = x
        x = self.proj_in(x)
        freq_cis = self.permutation(freq_cis, dim=0)

        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, freq_cis, self.attn_mask)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        a = torch.tensor(self.tanh_alpha)
        xa = a * F.tanh(xa / a)
        # xa = torch.clip(xa, -80, None)
        scale = F.softplus(-xa)
        return self.permutation((x_in - xb) * scale, inverse=True), torch.log(scale).mean(dim=[1, 2])
        scale = (-xa.float()).exp().type(xa.dtype)
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

    def cache_reverse(self, x: torch.Tensor, x_in: torch.Tensor, freq_cis: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x = self.permutation(x)
        # pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = self.permutation(x_in)
        x = self.proj_in(x)
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, freq_cis, self.attn_mask)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (xa.float()).exp().type(xa.dtype)
        return self.permutation(xb + scale * x_in, inverse=True)

    def reverse_step(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        i: int,  # 移除pos_embed参数
        y: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]
        x = self.proj_in(x_in)
        # 移除位置编码的添加
        # x = x + pos_embed[i : i + 1]

        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, freq_cis, attn_temp=attn_temp, which_cache=which_cache)  # here we use kv caching, so no attn_mask
        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        freq_cis: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        x = self.permutation(x)
        freq_cis = self.permutation(freq_cis, dim=0)
        # 移除位置编码相关代码
        # pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        T = x.size(1)
        for i in range(x.size(1) - 1):
            # 更新函数调用，移除pos_embed参数
            za, zb = self.reverse_step(x, freq_cis, i, y, which_cache='cond')
            if guidance > 0 and guide_what:
                za_u, zb_u = self.reverse_step(x, freq_cis, i, None, attn_temp=attn_temp, which_cache='uncond')
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance

                # CFG proposed by STARFlow
                # eps = 1e-8
                # zb_u_safe = torch.where(torch.abs(zb_u) < eps, eps * torch.sign(zb_u), zb_u)
                # s = (zb / zb_u_safe)**2
                # s = torch.clamp(s, 0, 1)
                # if 'a' in guide_what:
                #     za = za + g * s * (za - za_u) / (1 + g - g * s)
                # if 'b' in guide_what:
                #     zb = zb / torch.sqrt(1 + g - g * s)

                # default CFG
                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            a = torch.tensor(self.tanh_alpha)
            za = a * F.tanh(za / a)
            # softplus
            scale = F.softplus(-za[:,0])
            x[:, i + 1] = x[:, i + 1] / scale + zb[:, 0]
            # scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            # x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        layers_per_block: list[int],
        nvp: bool = True,
        num_classes: int = 0,
        head_dim: int = 64
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        permutations = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)]
        self.head_dim = head_dim

        blocks = []
        for i in range(len(layers_per_block)):
            blocks.append(
                MetaBlock(
                    in_channels * patch_size**2,
                    channels,
                    self.num_patches,
                    permutations[i % 2],
                    layers_per_block[i],
                    head_dim=head_dim,
                    nvp=nvp,
                    num_classes=num_classes if i == len(layers_per_block) - 1 else 0,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer('var', torch.ones(self.num_patches, in_channels * patch_size**2))
        self.precompute_pos = dict()
        self.fetch_pos(img_size // patch_size, img_size // patch_size, 'cuda')

    def fetch_pos(self, height, width, device):
        # if (height, width) in self.precompute_pos:
        #     return self.precompute_pos[(height, width)].to(device)
        # else:
        pos = precompute_freqs_cis_2d(self.head_dim, height, width).to(device)
            # self.precompute_pos[(height, width)] = pos
        return pos

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C')"""
        u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W)"""
        u = x.transpose(1, 2)
        return torch.nn.functional.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)

    def classify_forward(self, x:torch.Tensor, logits: torch.Tensor):
        x = self.vae.encode(x)
        x = self.patchify(x)
        freq_cis = self.fetch_pos(self.img_size // self.patch_size, self.img_size // self.patch_size, x.device)
        # outputs = [x.detach()]
        logdets = torch.zeros((), device=x.device)
        for block in self.blocks:
            x, logdet = block.classify_forward(x, logits, freq_cis)
            logdets = logdets + logdet
            # outputs.append(x.detach())
        loss = self.get_loss(x, logdets)
        return loss

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        x = self.patchify(x)
        freq_cis = self.fetch_pos(self.img_size // self.patch_size, self.img_size // self.patch_size, x.device)
        outputs = [x.detach()]
        logdets = torch.zeros((), device=x.device)
        for block in self.blocks:
            x, logdet = block(x, y, freq_cis)
            logdets = logdets + logdet
            outputs.append(x.detach())
        return x, outputs, logdets

    def repa_forward(self, y: torch.Tensor | None = None, cache: list[torch.Tensor] | None = None, align_tuple: tuple[int, ...] = tuple(range(8))):
        freq_cis = self.fetch_pos(self.img_size // self.patch_size, self.img_size // self.patch_size, y.device)
        for i, block in enumerate(self.blocks):
            if i in align_tuple:
                x, _ = block(cache[i], y, freq_cis)

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        return 0.5 * z.pow(2).mean() - logdets.mean()

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
        interval_guidance: float = 0.0,
    ) -> torch.Tensor | list[torch.Tensor]:
        freq_cis = self.fetch_pos(self.img_size // self.patch_size, self.img_size // self.patch_size, x.device)
        seq = [x]
        x = self.patchify(x)
        x = x * self.var.sqrt()
        interval = int(8 * interval_guidance)
        for i, block in enumerate(reversed(self.blocks)):
            x = block.reverse(x, y, freq_cis, 0 if i < interval else guidance, guide_what, attn_temp, annealed_guidance)
            seq.append(self.unpatchify(x))
        x = self.unpatchify(x)

        if not return_sequence:
            return x
        else:
            return seq

    def repa_reverse(
            self,
            cache: list[torch.Tensor],
            y: torch.Tensor | None = None,
            align_tuple: tuple[int, ...] = tuple(range(8)),
    ):
        cache = cache[::-1]
        x = cache[0] * self.var.sqrt()
        freq_cis = self.fetch_pos(self.img_size // self.patch_size, self.img_size // self.patch_size, x.device)
        for i, block in enumerate(reversed(self.blocks)):
            if len(self.blocks)-1-i in align_tuple:
                x = block.cache_reverse(cache[i+1], x, freq_cis, y)
            # print(x.shape, cache[i+1].shape)
            # print(torch.allclose(torch.from_numpy(x.detach().cpu().numpy()), torch.from_numpy(cache[i+1].cpu().numpy())))
        x = self.unpatchify(x)
        return x
