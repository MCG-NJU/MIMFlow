import time

import torch
import torch.nn as nn

class BaseTrainer(nn.Module):
    def __init__(self,
                 null_condition_p=0.1,
                 noise_std=0.05,
        ):
        super(BaseTrainer, self).__init__()
        self.null_condition_p = null_condition_p
        self.noise_std = noise_std

    def preproprocess(self, raw_iamges, x, condition):
        bsz = x.shape[0]
        # eps = self.noise_std * torch.randn_like(x)
        # x = x + eps
        if self.null_condition_p > 0:
            mask = torch.rand((bsz), device=condition.device) < self.null_condition_p
            mask = mask.expand_as(condition)
            condition[mask] = -1
        return raw_iamges, x, condition

    def _impl_trainstep(self, vae, net, ema_net, raw_images, x, y, epoch, mask_ratio=None):
        raise NotImplementedError

    def __call__(self, vae, net, ema_net, raw_images, x, condition, epoch, mask_ratio=None):
        raw_images, x, condition = self.preproprocess(raw_images, x, condition)
        return self._impl_trainstep(vae, net, ema_net, raw_images, x, condition, epoch, mask_ratio=mask_ratio)

class NLLTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _impl_trainstep(self, vae, net, ema_net, raw_images, x, y, epoch, mask_ratio=None):
        with torch.no_grad():
            x = vae.encode(x)
        eps = self.noise_std * torch.randn_like(x)
        x = x + eps
        z, outputs, logdets = net(x, y)
        loss = net.get_loss(z, logdets)

        out = dict(
            logdets=logdets.mean(),
            loss=loss,
        )
        return out

class DeTokNLLTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _impl_trainstep(self, vae, net, ema_net, raw_images, x, y, epoch, mask_ratio=None):
        x = vae.tokenize(x, add_noise=False)
        # print('finish vae:', time.time())
        eps = self.noise_std * torch.randn_like(x)
        x = x + eps
        z, outputs, logdets = net(x, y)
        loss = net.get_loss(z, logdets)

        out = dict(
            logdets=logdets.mean(),
            loss=loss,
        )
        return out

class NLL_np_Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _impl_trainstep(self,vae, net, ema_net, raw_images, x, y, epoch, mask_ratio=None):
        x = vae.encode(x)
        z, outputs, logdets = net(x, y)

        nll_loss = net.get_loss(z, logdets)
        norm_penalty = 0
        for intermediate in outputs[:-1]:
            norm_penalty += intermediate.pow(2).mean()
        loss = nll_loss + 1e-4 * norm_penalty


        out = dict(
            logdets=logdets.mean(),
            loss=loss,
            nll=nll_loss,
            norm_penalty=norm_penalty,
        )
        return out
