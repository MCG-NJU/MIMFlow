import torch
import torch.nn as nn

from src.models.layers.permutation import Permutation, PermutationIdentity, PermutationFlip
from src.models.layers.rope import precompute_freqs_cis_1d
from src.models.attention import AttentionBlockOptional


class Model1D(torch.nn.Module):
    """
    TarFlow 的 1D 变体，直接作用于 MAETok 产生的 token 序列，无需 patchify/unpatchify。
    """

    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        token_dim: int,
        seq_len: int,
        channels: int,
        layers_per_block: list[int],
        nvp: bool = True,
        num_classes: int = 0,
        head_dim: int = 64,
        use_normalize: bool = True,
        grid_size: tuple[int, int] | None = None,
        pos_encoding: str = "learnable",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.use_normalize = use_normalize
        self.pos_encoding = pos_encoding
        self.use_ape_pos, self.use_learnable_pos, self.use_rope_pos = self._parse_pos_encoding(pos_encoding)
        # print(self.use_ape_pos, self.use_learnable_pos, self.use_rope_pos)
        permutations: list[Permutation] = [PermutationIdentity(), PermutationFlip()]

        if self.use_normalize:
            self.batch_norm = torch.nn.BatchNorm1d(token_dim)
        else:
            self.batch_norm = None

        if self.use_ape_pos:
            self.register_buffer("ape_embed", self._build_1d_sincos_ape(seq_len, token_dim))
        else:
            self.ape_embed = None

        if self.use_learnable_pos:
            # 可学习的 1D 位置编码（无空间格子假设）
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, token_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        if self.use_rope_pos:
            if head_dim % 2 != 0:
                raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
            self.register_buffer("rope_freqs_cis", precompute_freqs_cis_1d(head_dim, seq_len))
        else:
            self.rope_freqs_cis = None

        blocks = []
        for i, num_layers in enumerate(layers_per_block):
            blocks.append(
                MetaBlock1D(
                    token_dim,
                    channels,
                    self.seq_len,
                    permutations[i % 2],
                    num_layers,
                    head_dim=head_dim,
                    nvp=nvp,
                    num_classes=num_classes,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)

        # prior for nvp mode should be all ones, but needs to be learned for the vp mode
        self.register_buffer("var", torch.ones(self.seq_len, token_dim))

    def classify_forward(self, x: torch.Tensor, logits: torch.Tensor, ids_restore: torch.Tensor | None = None):
        freq_cis = None
        x = self._apply_additive_positional_encoding(x)
        if self.use_rope_pos:
            freq_cis = self._get_rope_freqs(x)
        if self.use_normalize:
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)

        logdets = torch.zeros((), device=x.device)
        for block in self.blocks:
            x, logdet = block.classify_forward(x, logits, freq_cis)
            logdets = logdets + logdet
        loss = self.get_loss(x, logdets)
        return loss

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        ids_restore: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        freq_cis = None
        x = self._apply_additive_positional_encoding(x)
        if self.use_rope_pos:
            freq_cis = self._get_rope_freqs(x)
        if self.use_normalize:
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)

        outputs = [x.detach()]
        logdets = torch.zeros((), device=x.device)
        for block in self.blocks:
            x, logdet = block(x, y, freq_cis)
            logdets = logdets + logdet
            outputs.append(x.detach())
        return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        return 0.5 * z.pow(2).mean() - logdets.mean()

    def reverse(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = "ab",
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
        interval_guidance: float = 0.0,
    ) -> torch.Tensor | list[torch.Tensor]:
        seq = [x]
        # x = x * self.var.sqrt()
        freq_cis = None
        if self.use_rope_pos:
            freq_cis = self._get_rope_freqs(x)

        interval = int(8 * interval_guidance)
        for i, block in enumerate(reversed(self.blocks)):
            x = block.reverse(
                x,
                y,
                freq_cis,
                0 if i < interval else guidance,
                guide_what,
                attn_temp,
                annealed_guidance,
            )
            seq.append(x)

        if self.use_normalize:
            mean = self.batch_norm.running_mean.view(1, 1, -1)
            std = torch.sqrt(self.batch_norm.running_var.view(1, 1, -1) + self.batch_norm.eps)
            x = x * std + mean

        if not return_sequence:
            return x
        return seq

    def _get_rope_freqs(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return self.rope_freqs_cis[:seq_len].to(x.device)

    @staticmethod
    def _parse_pos_encoding(pos_encoding: str) -> tuple[bool, bool, bool]:
        value = pos_encoding.strip().lower().replace("_", "+").replace(",", "+")
        if value in {"", "none"}:
            return False, False, False

        raw_parts = [part for part in value.split("+") if part]
        aliases = {"learn": "learnable", "absolute": "ape", "abs": "ape"}
        parts: set[str] = set()
        for part in raw_parts:
            normalized = aliases.get(part, part)
            if normalized == "both":
                # Backward-compatible shortcut.
                parts.update({"learnable", "rope"})
            elif normalized == "all":
                parts.update({"ape", "learnable", "rope"})
            else:
                parts.add(normalized)

        valid_parts = {"ape", "learnable", "rope"}
        invalid_parts = parts - valid_parts
        if invalid_parts:
            invalid_text = ", ".join(sorted(invalid_parts))
            raise ValueError(f"Unknown pos_encoding option(s): {invalid_text}")

        use_ape = "ape" in parts
        use_learnable = "learnable" in parts
        use_rope = "rope" in parts
        if not (use_ape or use_learnable or use_rope):
            raise ValueError("pos_encoding parsed to empty set; use none or a valid combination")
        return use_ape, use_learnable, use_rope

    def _apply_additive_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if self.use_ape_pos:
            x = x + self.ape_embed[:, :seq_len].to(device=x.device, dtype=x.dtype)
        if self.use_learnable_pos:
            # 直接加 1D learnable pos emb
            x = x + self.pos_embed[:, :seq_len].to(dtype=x.dtype)
        return x

    @staticmethod
    def _build_1d_sincos_ape(seq_len: int, dim: int) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        ape = torch.zeros(seq_len, dim, dtype=torch.float32)
        ape[:, 0::2] = torch.sin(positions * div_term)
        cos_dim = ape[:, 1::2].shape[1]
        if cos_dim > 0:
            ape[:, 1::2] = torch.cos(positions * div_term[:cos_dim])
        return ape.unsqueeze(0)


class MetaBlock1D(torch.nn.Module):
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
            [AttentionBlockOptional(channels, head_dim, expansion, permutation) for _ in range(num_layers)]
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

        if self.class_embed is not None:
            class_weights = self.class_embed.squeeze(1)  # (num_cls, C)
            class_features = torch.matmul(logits, class_weights).unsqueeze(1)
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
        xa = a * torch.tanh(xa / a)
        scale = (-xa.float()).exp().type(xa.dtype)
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, freq_cis: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        N = x.shape[1]
        x = self.permutation(x)  # [B, T, C]
        x_in = x
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
            x = block(x, freq_cis, self.attn_mask[:N, :N])
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        a = torch.tensor(self.tanh_alpha)
        xa = a * torch.tanh(xa / a)
        scale = (-xa.float()).exp().type(xa.dtype)
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

    def reverse_step(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        i: int,
        y: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]
        x = self.proj_in(x_in)

        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, freq_cis, attn_temp=attn_temp, which_cache=which_cache)
        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, AttentionBlockOptional):
                m.attention.sample = flag
                m.attention.k_cache = {'cond': [], 'uncond': []}
                m.attention.v_cache = {'cond': [], 'uncond': []}

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
        self.set_sample_mode(True)
        T = x.size(1)
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, freq_cis, i, y, which_cache='cond')
            if guidance > 0 and guide_what:
                za_u, zb_u = self.reverse_step(x, freq_cis, i, None, attn_temp=attn_temp, which_cache='uncond')
                g = (i + 1) / (T - 1) * guidance if annealed_guidance else guidance
                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            a = torch.tensor(self.tanh_alpha)
            za = a * torch.tanh(za / a)
            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)

