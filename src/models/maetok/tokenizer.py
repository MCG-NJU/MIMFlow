# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


from src.models.maetok.modules import Encoder, Decoder, TimmViTEncoder, TimmViTDecoder
from src.models.maetok.quantizers.vq import VectorQuantizer
from src.models.maetok.quantizers.kl import DiagonalGaussianDistribution
from src.models.maetok.quantizers.softvq import SoftVectorQuantizer

from timm import create_model

class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        # if device is None:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def forward(self, x):
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        # if device is None:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def forward(self, x):
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


@dataclass
class ModelArgs:
    image_size: int = 256
    base_image_size: int = 256
    
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    vq_loss_ratio: float = 1.0 # for soft vq
    kl_loss_weight: float = 0.000001
    tau: float = 0.1
    num_codebooks: int = 1
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    enc_type: str = 'cnn'
    dec_type: str = 'cnn'
    encoder_model: str = 'llamagen_encoder'
    decoder_model: str = 'llamagen_decoder'
    num_latent_tokens: int = 256
    to_pixel: str = 'linear'
    
    # for pre-trained models
    enc_tuning_method: str = 'full'
    dec_tuning_method: str = 'full'
    enc_pretrained: bool = False
    dec_pretrained: bool = False 
    
    # for vit 
    enc_patch_size: int = 16
    dec_patch_size: int = 16
    enc_drop_path_rate: float = 0.0
    dec_drop_path_rate: float = 0.0
    
    # deocder cls token
    dec_cls_token: bool = True
    
    # rope
    use_ape: bool = True 
    use_rope: bool = False
    rope_mixed: bool = False
    rope_theta: float = 10.0
    
    # repa for vit
    repa: bool = False
    repa_patch_size: int = 16
    repa_model: str = 'vit_base_patch16_224'
    repa_proj_dim: int = 2048
    repa_loss_weight: float = 0.1
    repa_align: str = 'global'
    
    vq_mean: float = 0.0
    vq_std: float = 1.0
    
    # encoder token drop for mask modeling
    enc_token_drop: float = 0.0
    enc_token_drop_max: float = 0.6
    enc_no_mask_prob: float = 0.5

    # aux decoder model
    aux_dec_model: str = 'vit_tiny_patch14_dinov2_movq'
    aux_loss_mask: bool = False
    aux_dec_cls_token: bool = True
    aux_hog_dec: bool = True
    aux_dino_dec: bool = True
    aux_supcls_dec: bool = True
    aux_clip_dec: bool = True

    # decoder repa block index (optional, 0-based)
    dec_repa_block: Optional[int] = None
    

class VQModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align
        if config.repa and config.enc_type == 'vit':
            self.repa_model = create_model(config.repa_model, pretrained=True, img_size=config.image_size, patch_size=config.repa_patch_size)
            for param in self.repa_model.parameters():
                param.requires_grad = False
            self.repa_model.eval()
            repa_z_dim = self.repa_model.embed_dim
            self.repa_z_dim = repa_z_dim
            # self.projection = build_mlp(config.codebook_embed_dim, config.repa_proj_dim, repa_z_dim)
            self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            repa_z_dim = None
        
        
        if config.enc_type == 'cnn':
            if config.encoder_model == 'llamagen_encoder':
                self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        elif config.enc_type == 'vit':
            self.encoder = TimmViTEncoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.encoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size, 'drop_path_rate': config.enc_drop_path_rate},
                pretrained=config.enc_pretrained,
                tuning_method=config.enc_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                token_drop=config.enc_token_drop,
                token_drop_max=config.enc_token_drop_max,
                no_mask_prob=config.enc_no_mask_prob,
                base_img_size=config.base_image_size
            )
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)
            
        
        if config.dec_type == 'cnn':
            if config.decoder_model == 'llamagen_decoder':
                self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        elif config.dec_type == 'vit':
            self.decoder = TimmViTDecoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.decoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': config.dec_drop_path_rate, 'latent_dim': config.codebook_embed_dim},
                pretrained=config.dec_pretrained,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel=config.to_pixel,
                base_img_size=config.base_image_size
            )
            self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
        # check movq
        if 'movq' in config.decoder_model:
            self.use_movq = True 
        else:
            self.use_movq = False
        
        
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        
        if self.repa and self.training:
            # get z from repa_encoder
            rescale_x = self.scale(self.de_scale(x))
            z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

            # taking average over spatial dimension
            if self.repa_align == 'global':
                z = z.mean(dim=1)
                z_hat = quant.mean(dim=1)
                # calculate repa loss
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d':
                z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d_shuffle':
                # shuffle the length dimension of z and avg
                indices = torch.randperm(z.shape[1])
                z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1) , quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'repeat':
                z_hat = self.projection(quant)
                b, l, d = z_hat.shape
                z_hat = z_hat.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
            

            z = F.normalize(z, dim=-1)
            z_hat = F.normalize(z_hat, dim=-1)
            proj_loss = mean_flat(-(z * z_hat).sum(dim=-1))
            proj_loss = proj_loss.mean()
            proj_loss *= self.repa_loss_weight
            
            emb_loss += (proj_loss,)
        
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None, return_block_tokens_at=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        # only transformer decoder supports returning intermediate tokens
        if self.use_movq:
            if return_block_tokens_at is not None and self.config.dec_type == 'vit':
                dec = self.decoder(quant, tmp_quant, h, w, return_block_tokens_at=return_block_tokens_at)
            else:
                dec = self.decoder(quant, tmp_quant, h, w)
        else:
            if return_block_tokens_at is not None and self.config.dec_type == 'vit':
                dec = self.decoder(quant, None, h, w, return_block_tokens_at=return_block_tokens_at)
            else:
                dec = self.decoder(quant, None, h, w)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, x=input, h=h, w=w)
        return dec, diff, info


class SoftVQModel(VQModel, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = SoftVectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                            config.entropy_loss_ratio, 
                                            config.tau,                                   
                                            config.num_codebooks,
                                            config.codebook_l2_norm, config.codebook_show_usage)


class KLModel(VQModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.kl_loss_weight = config.kl_loss_weight
        self.quantize = None
        
        if config.enc_type == 'cnn':
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim * 2, 1)
        elif config.enc_type == 'vit':
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim * 2)
        

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h)
        h_posterior = DiagonalGaussianDistribution(h)
        return h_posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_code(self, posterior, shape=None):
        z = posterior.sample()
        dec = self.decode(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # compute kl loss
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        diff = (kl_loss * self.kl_loss_weight, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        return dec, diff, None


class AEModel(VQModel):
    def __init__(self, config: ModelArgs,
                tags=["arxiv:2502.03444", "image-generation", "1d-tokenizer", "128 tokens", "MAETok"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = None 



    def encode(self, x):
        
        h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None, return_block_tokens_at=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            if return_block_tokens_at is not None and self.config.dec_type == 'vit':
                dec = self.decoder(quant, tmp_quant, h, w, return_block_tokens_at=return_block_tokens_at)
            else:
                dec = self.decoder(quant, tmp_quant, h, w)
        else:
            if return_block_tokens_at is not None and self.config.dec_type == 'vit':
                dec = self.decoder(quant, None, h, w, return_block_tokens_at=return_block_tokens_at)
            else:
                dec = self.decoder(quant, None, h, w)
        return dec


class MaskAEModel(AEModel):
    def __init__(self, config: ModelArgs,
                tags=["arxiv:2502.03444", "image-generation", "1d-tokenizer", "128 tokens", "MAETok"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        # config.repa = True # for dinov2 decoder
        super().__init__(config)

        self.aux_loss_mask = config.aux_loss_mask
        
        self.aux_hog_decoder = config.aux_hog_dec
        if self.aux_hog_decoder:
            from src.models.maetok.utils.hog import HOGGenerator
            print('Using HOG decoder: ', config.aux_dec_model)
            self.decoder_hog = TimmViTDecoder(
                in_channels=3, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.aux_dec_model,
                model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': 0.0, 'latent_dim': config.codebook_embed_dim},
                pretrained=False,
                tuning_method='full',
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.aux_dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel='identity',
                base_img_size=config.base_image_size
            )
            self.post_quant_conv_hog = nn.Linear(config.codebook_embed_dim, self.decoder_hog.embed_dim)
            self.to_pixel_hog = nn.Linear(self.decoder_hog.embed_dim, 108)
            self.hog_generator = HOGGenerator()
            if 'movq' in config.aux_dec_model:
                self.hog_use_movq = True 
            else:
                self.hog_use_movq = False
        
        self.aux_dino_decoder = config.aux_dino_dec
        if self.aux_dino_decoder:
            print('Using DINO decoder: ', config.aux_dec_model)
            self.decoder_dino = TimmViTDecoder(
                in_channels=3, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.aux_dec_model,
                model_kwargs={'img_size': self.repa_model.img_size, 'patch_size': self.repa_model.patch_size, 'drop_path_rate': 0.0, 'latent_dim': config.codebook_embed_dim},
                pretrained=False,
                tuning_method='full',
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.aux_dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel='identity',
                base_img_size=config.base_image_size
            )
            self.post_quant_conv_dino = nn.Linear(config.codebook_embed_dim, self.decoder_dino.embed_dim)
            self.to_pixel_dino = nn.Linear(self.decoder_dino.embed_dim, self.repa_model.embed_dim)
            if 'movq' in config.aux_dec_model:
                self.dino_use_movq = True 
            else:
                self.dino_use_movq = False
        
        self.aux_clip_decoder = config.aux_clip_dec
        if self.aux_clip_decoder:
            self.clip_model = create_model('vit_so400m_patch14_siglip_gap_224', pretrained=True, img_size=config.image_size, patch_size=config.repa_patch_size)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.dynamic_img_size = True
            self.clip_model.eval()
            self.clip_de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.clip_scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            print('Using CLIP decoder: ', config.aux_dec_model)
            self.decoder_clip = TimmViTDecoder(
                in_channels=3, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.aux_dec_model,
                model_kwargs={'img_size': self.clip_model.img_size, 'patch_size': self.clip_model.patch_size, 'drop_path_rate': 0.0, 'latent_dim': config.codebook_embed_dim},
                pretrained=False,
                tuning_method='full',
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.aux_dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel='identity',
                base_img_size=config.base_image_size
            )
            self.post_quant_conv_clip = nn.Linear(config.codebook_embed_dim, self.decoder_clip.embed_dim)
            self.to_pixel_clip = nn.Linear(self.decoder_clip.embed_dim, self.clip_model.embed_dim)
            if 'movq' in config.aux_dec_model:
                self.clip_use_movq = True 
            else:
                self.clip_use_movq = False

        # decoder-side repa projection for image tokens
        if self.repa and self.config.dec_type == 'vit' and config.dec_repa_block:
            self.dec_repa_projection = build_mlp(self.decoder.embed_dim, config.repa_proj_dim, self.repa_z_dim)
        else:
            self.dec_repa_projection = None
        self.repa_block_idx = config.dec_repa_block
        self.layernorm = nn.LayerNorm(config.codebook_embed_dim, elementwise_affine=False)


    def encode(self, x):
        # h = self.encoder(x)
        if self.training:
            h, mask = self.encoder(x, return_mask=True)
        else:
            h = self.encoder(x)
        quant = self.quant_conv(h)
        quant = self.layernorm(quant)
        self.quant = quant
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        if self.training:
            return quant, emb_loss, info, mask
        else:
            return quant, emb_loss, info
    
    def decode_all(self, quant, input, h, w, mask=None):
        dec_out = self.decode(quant, x=input, h=h, w=w, return_block_tokens_at=self.repa_block_idx)
        if isinstance(dec_out, tuple):
            dec, dec_mid_tokens = dec_out
        else:
            dec, dec_mid_tokens = dec_out, None
         
        # decode hog
        if self.training:
            # decoder-side repa alignment
            if self.repa and dec_mid_tokens is not None:
                repa_rec_loss = self.decoder_repa_align(dec_mid_tokens, input)
            else:
                repa_rec_loss = torch.tensor(0.0, device=dec.device)
            # decode hog feature
            if self.aux_hog_decoder:
                dec_hog = self.decode_hog(quant, x=input, h=h, w=w)   
                dec_hog = self.to_pixel_hog(dec_hog)
                # get hog_target
                z_hog = self.hog_generator(input) 
                if self.aux_loss_mask and mask is not None and mask.sum() > 0:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog, reduction='none')
                    hog_rec_loss = (hog_rec_loss * mask).sum() / mask.sum() / z_hog.size(-1)
                else:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog)
            else:
                hog_rec_loss = torch.tensor(0.0, device=dec.device)
        
            # decode dinov2 feature
            if self.aux_dino_decoder:
                dec_dino = self.decode_dino(quant, x=input, h=h, w=w)
                dec_dino = self.to_pixel_dino(dec_dino)
                
                # get z from repa_encoder
                rescale_x = self.scale(self.de_scale(input))
                z_dino = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

                z_dino = F.normalize(z_dino, dim=-1)
                dec_dino = F.normalize(dec_dino, dim=-1)

                if self.aux_loss_mask and mask is not None and mask.sum() > 0:
                    dino_rec_loss = -(dec_dino * z_dino).sum(dim=-1, keepdim=True)
                    dino_rec_loss = (dino_rec_loss * mask).sum() / mask.sum()
                else:
                    dino_rec_loss = mean_flat(-(dec_dino * z_dino).sum(dim=-1))
                    dino_rec_loss = dino_rec_loss.mean()
            else:
                dino_rec_loss = torch.tensor(0.0, device=dec.device)
            
            # deocde clip feature
            if self.aux_clip_decoder:
                dec_clip = self.decode_clip(quant, x=input, h=h, w=w)
                dec_clip = self.to_pixel_clip(dec_clip)
                # get clip_target
                rescale_x = self.clip_scale(self.clip_de_scale(input))
                z_clip = self.clip_model.forward_features(rescale_x)[:, self.clip_model.num_prefix_tokens:]
                
                z_clip = F.normalize(z_clip, dim=-1)
                dec_clip = F.normalize(dec_clip, dim=-1)
                
                if self.aux_loss_mask and mask is not None and mask.sum() > 0:
                    clip_rec_loss = -(dec_clip * z_clip).sum(dim=-1, keepdim=True)
                    clip_rec_loss = (clip_rec_loss * mask).sum() / mask.sum()
                else:
                    clip_rec_loss = mean_flat(-(dec_clip * z_clip).sum(dim=-1))
                    clip_rec_loss = clip_rec_loss.mean()   
            else:
                clip_rec_loss = torch.tensor(0.0, device=dec.device)
            
            diff = (repa_rec_loss, dino_rec_loss, hog_rec_loss, clip_rec_loss, )

        return dec, diff

    def decode_hog(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_hog(quant)
        if self.hog_use_movq:
            dec = self.decoder_hog(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_hog(quant, None, h, w)
        return dec
    
    def decode_dino(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_dino(quant)
        if self.dino_use_movq:
            dec = self.decoder_dino(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_dino(quant, None, h, w)
        return dec

    def decode_clip(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_clip(quant)
        if self.clip_use_movq:
            dec = self.decoder_clip(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_clip(quant, None, h, w)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        if self.training:
            quant, diff, info, mask = self.encode(input)
        else:
            quant, diff, info = self.encode(input)
        self.quant = quant
        # print(quant.shape)
        dec_out = self.decode(quant, x=input, h=h, w=w, return_block_tokens_at=self.repa_block_idx)
        if isinstance(dec_out, tuple):
            dec, dec_mid_tokens = dec_out
        else:
            dec, dec_mid_tokens = dec_out, None
         
        # decode hog
        if self.training:
            # decoder-side repa alignment
            if self.repa and dec_mid_tokens is not None:
                repa_rec_loss = self.decoder_repa_align(dec_mid_tokens, input)
            else:
                repa_rec_loss = torch.tensor(0.0, device=dec.device)
            # decode hog feature
            if self.aux_hog_decoder:
                dec_hog = self.decode_hog(quant, x=input, h=h, w=w)   
                dec_hog = self.to_pixel_hog(dec_hog)
                # get hog_target
                z_hog = self.hog_generator(input) 
                if self.aux_loss_mask:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog, reduction='none')
                    hog_rec_loss = (hog_rec_loss * mask).sum() / mask.sum() / z_hog.size(-1)
                else:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog)
            else:
                hog_rec_loss = torch.tensor(0.0, device=dec.device)
        
            # decode dinov2 feature
            if self.aux_dino_decoder:
                dec_dino = self.decode_dino(quant, x=input, h=h, w=w)
                dec_dino = self.to_pixel_dino(dec_dino)
                
                # get z from repa_encoder
                rescale_x = self.scale(self.de_scale(input))
                z_dino = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

                z_dino = F.normalize(z_dino, dim=-1)
                dec_dino = F.normalize(dec_dino, dim=-1)

                if self.aux_loss_mask:
                    dino_rec_loss = -(dec_dino * z_dino).sum(dim=-1, keepdim=True)
                    dino_rec_loss = (dino_rec_loss * mask).sum() / mask.sum()
                else:
                    dino_rec_loss = mean_flat(-(dec_dino * z_dino).sum(dim=-1))
                    dino_rec_loss = dino_rec_loss.mean()
            else:
                dino_rec_loss = torch.tensor(0.0, device=dec.device)
            
            # deocde clip feature
            if self.aux_clip_decoder:
                dec_clip = self.decode_clip(quant, x=input, h=h, w=w)
                dec_clip = self.to_pixel_clip(dec_clip)
                # get clip_target
                rescale_x = self.clip_scale(self.clip_de_scale(input))
                z_clip = self.clip_model.forward_features(rescale_x)[:, self.clip_model.num_prefix_tokens:]
                
                z_clip = F.normalize(z_clip, dim=-1)
                dec_clip = F.normalize(dec_clip, dim=-1)
                
                if self.aux_loss_mask:
                    clip_rec_loss = -(dec_clip * z_clip).sum(dim=-1, keepdim=True)
                    clip_rec_loss = (clip_rec_loss * mask).sum() / mask.sum()
                else:
                    clip_rec_loss = mean_flat(-(dec_clip * z_clip).sum(dim=-1))
                    clip_rec_loss = clip_rec_loss.mean()   
            else:
                clip_rec_loss = torch.tensor(0.0, device=dec.device)
            
            diff = (repa_rec_loss, dino_rec_loss, hog_rec_loss, clip_rec_loss)

        return dec, diff

    def decoder_repa_align(self, dec_img_tokens, target):
        if self.dec_repa_projection is None:
            return torch.tensor(0.0, device=dec_img_tokens.device)
        # compute decoder-side repa loss using intermediate image tokens
        rescale_x = self.scale(self.de_scale(target))
        z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

        if self.repa_align == 'global':
            z = z.mean(dim=1)
            z_hat = dec_img_tokens.mean(dim=1)
            z_hat = self.dec_repa_projection(z_hat)
        elif self.repa_align == 'avg_1d':
            z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), dec_img_tokens.shape[1]).permute(0, 2, 1)
            z_hat = dec_img_tokens
            z_hat = self.dec_repa_projection(z_hat)
        elif self.repa_align == 'avg_1d_shuffle':
            indices = torch.randperm(z.shape[1], device=z.device)
            z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1) , dec_img_tokens.shape[1]).permute(0, 2, 1)
            z_hat = dec_img_tokens
            z_hat = self.dec_repa_projection(z_hat)
        elif self.repa_align == 'repeat':
            z_hat = self.dec_repa_projection(dec_img_tokens)
            b, l, d = z_hat.shape
            z_hat = z_hat.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
        else:
            z_hat = self.dec_repa_projection(dec_img_tokens)

        z = F.normalize(z, dim=-1)
        z_hat = F.normalize(z_hat, dim=-1)
        repa_loss = mean_flat(-(z * z_hat).sum(dim=-1))
        repa_loss = repa_loss.mean()
        repa_loss *= self.repa_loss_weight

        return repa_loss

#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
class VQ_8(VQModel):
    def __init__(self, **kwargs):
        super().__init__(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))


class VQ_16(VQModel):
    def __init__(self, **kwargs):
        super().__init__(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


class KL_8(KLModel):
    def __init__(self, **kwargs):
        super().__init__(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))


class KL_16(KLModel):
    def __init__(self, **kwargs):
        super().__init__(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


class AE_16(AEModel):
    def __init__(self, **kwargs):
        super().__init__(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


class MaskAE_16(MaskAEModel):
    def __init__(self, **kwargs):
        super().__init__(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


class SoftVQ(SoftVQModel):
    def __init__(self, **kwargs):
        super().__init__(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


VQ_models = {
    'AE-16': AE_16,
    'MaskAE-16': MaskAE_16,
    'VQ-16': VQ_16, 'VQ-8': VQ_8,
    'KL-16': KL_16, 'KL-8': KL_8,
    'SoftVQ': SoftVQ,
    }


if __name__ == '__main__':
    
    # model = VQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', repa=True, repa_model='vit_base_patch14_dinov2.lvd142m', repa_align='avg_1d_shuffle', enc_img_res=True, enc_img_align='avg_1d', dec_img_res=True)    
    # model = SoftVQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384, topk=16)
    # model = AE_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384)
    model = MaskAE_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384, aux_dec_model='vit_tinytiny_patch14_dinov2_movq2', aux_loss_mask=True, aux_dec_cls_token=True, aux_hog_dec=True, aux_dino_dec=True, aux_clip_dec=True, enc_token_drop=0.4, enc_token_drop_max=0.6)      
    model.train()
    model = model.cuda()
    # model = KL_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    # model = GMM_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    x = torch.randn(4, 3, 256, 256).cuda()
    y, _, info = model(x)
    print(_)
