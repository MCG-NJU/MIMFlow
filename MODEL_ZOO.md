# Model Zoo

This file is a placeholder for public checkpoints and final evaluation commands.

Planned entries:

| Model | Resolution | Tokens | Params | Checkpoint | Notes |
| --- | --- | ---: | ---: | --- | --- |
| MIMFlow-L | 256 x 256 | 128 | 482M | TBD | Phase 1 + decoder fine-tuning |

To evaluate a checkpoint, first generate samples:

```bash
bash scripts/save_samples.sh /path/to/mimflow_l.ckpt
```

Then run the ADM evaluator on the produced `generated/samples.npz` file as described in `README.md`.
