# Open-Source Cleanup Notes

This working tree has been narrowed around the MIMFlow training path.

## Public Training Path

Use these files for the cleaned reproduction workflow:

- `configs/mimflow_l_phase1.yaml`
- `configs/mimflow_l_phase2_decoder_ft.yaml`
- `configs/mimflow_l_validate_samples.yaml`
- `scripts/train_phase1.sh`
- `scripts/train_phase2_decoder_ft.sh`
- `scripts/save_samples.sh`

## Evaluation

Validation saves generated samples to `samples.npz`. Metrics are computed outside
this repository with the ADM guided-diffusion evaluator.

## Legacy Research Artifacts

The repository may still contain analysis scripts from exploratory research. They are not part of the cleaned MIMFlow training path unless explicitly referenced by the README.

Before public release, consider moving legacy-only files into a separate branch, artifact archive, or internal backup if they are not needed for reproducibility.
