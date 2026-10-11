# Multi-user attribution experiment

This compact experiment trains one open-set model to attribute an image to its most likely registered user. If no user score passes the threshold calibrated on validation data, the image is labeled as unknown.

## Method

- Each registered user has an independent detection branch, and all branches are optimized jointly.
- Known-user samples use multiclass cross-entropy.
- Balanced binary losses teach each branch to distinguish its own user from other users, unwatermarked images, and wrong-key images.
- The rejection threshold is selected only on the validation split and frozen before final testing.
- Model selection prioritizes open-set attribution accuracy, followed by closed-set Top-1 accuracy and the unknown false-accept rate.

## Dataset format

Training, validation, and test files are PyTorch objects with this structure:

```python
{
    "images": torch.Tensor,  # [N, 1, H, W], preferably normalized to [-1, 1]
    "labels": torch.Tensor,  # [N]
}
```

Label convention:

- `0 .. U-1`: the `U` registered users;
- `U`: real, unwatermarked images;
- `U+1`: wrong-key generated images, used as a distinct class during training;
- during validation and testing, every label greater than or equal to `U` is treated as unknown.

Datasets, user keys, outputs, and checkpoints are not published. Place them under the ignored `data/`, `outputs/`, or `checkpoints/` directories.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

Copy the example configuration and set local data paths:

```bash
cp config.example.json config.json
python main.py train --config config.json
```

Training displays one overall progress bar and resumes from `outputs/last.pt` when that checkpoint exists.

```bash
python main.py evaluate --config config.json
python main.py results --config config.json
python -m unittest discover -s tests -v
```

## Threshold calibration

The presence threshold is not assumed to be `0.5`. Validation scores are scanned to find the operating point that maximizes known-user acceptance while respecting the configured unknown false-positive-rate limit. That threshold is then saved and reused unchanged for the final test set.

## Local outputs

```text
outputs/
├── best.pt
├── last.pt
├── validation_rule.json
└── summary.json
```

`outputs/` is ignored by Git. Historical results, private data, and model weights are intentionally absent from this public package.
