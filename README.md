# MIMFlow: Integrating Masked Image Modeling with Normalizing Flows for End-to-End Image Generation

Yang Chen<sup>1,2</sup>, Xiaowei Xu<sup>2</sup>, Shuai Wang<sup>1</sup>, Xinwen Zhang<sup>1,2</sup>, Qiushi Guo<sup>2</sup>, Tiezheng Ge<sup>2</sup>, Limin Wang<sup>1,3</sup>

<sup>1</sup> State Key Laboratory for Novel Software Technology, Nanjing University  
<sup>2</sup> Alibaba Group  
<sup>3</sup> Shanghai AI Lab

Contact: yang-chen@smail.nju.edu.cn

[[ARXIV](https://arxiv.org/abs/2606.26016)] [[Hugging Face](https://huggingface.co/papers/2606.26016)]

This repository contains the official PyTorch implementation of **MIMFlow**.

MIMFlow unifies masked image modeling, latent reconstruction, and normalizing-flow density estimation in a single end-to-end generative framework. The main ImageNet 256 x 256 model uses a ViT-B masked autoencoder with 128 latent tokens and an Improved-STARFlow-L prior.

Generated samples are saved as NumPy archives and scored with the ADM evaluation suite.

## Highlights

- End-to-end training of a masked Transformer autoencoder and latent normalizing flow.
- Fixed 128-token semantic bottleneck with latent dimension 64.
- Random mask ratio 0.4-0.6 during joint training.
- Improved-STARFlow-L 1D latent flow with classifier-free conditioning.
- Optional decoder fine-tuning stage with reconstruction, LPIPS, and GAN losses.

## Repository Layout

```text
configs/
  mimflow_l_phase1.yaml            # 90-epoch joint VAE + NF training
  mimflow_l_phase2_decoder_ft.yaml # 2-epoch decoder fine-tuning
  mimflow_l_validate_samples.yaml  # save samples for ADM evaluation
scripts/
  train_phase1.sh
  train_phase2_decoder_ft.sh
  save_samples.sh
src/
  data/                            # ImageNet datasets and transforms
  models/                          # MIMFlow tokenizer, latent flow, layers
  trainer/                         # training objectives
  callbacks/                       # EMA, checkpointing, image saving
  utils/                           # metrics and distributed helpers
```

## Installation

The code was developed with Python 3.10 and PyTorch/Lightning.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some backbones and auxiliary feature models are loaded through `timm`/Hugging Face. Make sure the runtime can access those model weights or has them available in the local cache.

## Data

Prepare ImageNet with the standard folder layout:

```text
data/imagenet/
  train/
    n01440764/
    ...
  val/
    n01440764/
    ...
```

The training configs default to `data/imagenet`. Override it with:

```bash
--data.data_root /path/to/imagenet
```

## Training

### Phase 1: Joint MIMFlow Training

This stage jointly optimizes the masked autoencoder and the latent normalizing flow.

```bash
DATA_ROOT=/path/to/imagenet \
OUTPUT_DIR=workdirs \
NPROC_PER_NODE=8 \
bash scripts/train_phase1.sh
```

Paper setting:

- 90 epochs
- global batch size 256
- AdamW, learning rate `1e-4`, weight decay `1e-4`
- EMA decay `0.9999`
- latent noise scale `sigma=0.3`
- mask ratio `0.4-0.6`

The default per-GPU batch size is 32, so `NPROC_PER_NODE=8` gives global batch size 256.

### Phase 2: Decoder Fine-Tuning

This stage initializes from a Phase 1 checkpoint and fine-tunes the decoder with perceptual and adversarial losses.

```bash
DATA_ROOT=/path/to/imagenet \
OUTPUT_DIR=workdirs \
NPROC_PER_NODE=8 \
bash scripts/train_phase2_decoder_ft.sh /path/to/phase1.ckpt
```

Paper setting:

- 2 epochs
- same optimizer settings as Phase 1
- LPIPS weight `1.1`
- GAN weight `0.05`
- flow model frozen, decoder trainable

## Model Zoo

Checkpoint links will be added in `MODEL_ZOO.md`.

## Sampling And Evaluation

This repository does not compute FID/IS/precision/recall with `src/utils/metrics.py` during validation. Validation only generates samples and saves them as an ADM-compatible NumPy archive:

```bash
OUTPUT_DIR=workdirs \
NPROC_PER_NODE=8 \
bash scripts/save_samples.sh /path/to/mimflow_l.ckpt
```

The output is written under:

```text
workdirs/test/<experiment>/val/iter_<step>_cfg<cfg>/generated/samples.npz
```

The archive contains `arr_0` in `uint8` NHWC format.

Use the ADM evaluation suite to score generated samples:

```bash
git clone https://github.com/openai/guided-diffusion.git
cd guided-diffusion/evaluation

conda create -n adm-fid python=3.10
conda activate adm-fid
pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm

wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
```

## Citation

```bibtex
@misc{chen2026mimflow,
      title={MIMFlow: Integrating Masked Image Modeling with Normalizing Flows for End-to-End Image Generation}, 
      author={Yang Chen and Xiaowei Xu and Shuai Wang and Xinwen Zhang and Qiushi Guo and Tiezheng Ge and Limin Wang},
      year={2026},
      eprint={2606.26016},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.26016}, 
}
```

## Acknowledgements

This codebase builds on ideas and components from several excellent open-source projects. We thank the authors and maintainers of:

- [guided-diffusion](https://github.com/openai/guided-diffusion)
- [MAETok](https://github.com/Hhhhhhao/continuous_tokenizer)
- [STARFlow](https://github.com/apple/ml-starflow)
- [SimFlow](https://github.com/ByteDance-Seed/SimFlow)
- [DDT](https://github.com/MCG-NJU/DDT/tree/main)
