# Installation and local assets

## Environment

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA driver if the default package is not appropriate for your machine.

## Data

The original IAM/CVL data and pretrained weights are not redistributed here. Keep local assets in the following ignored structure:

```text
files/
├── IAM-32.pickle
├── CVL-32.pickle
└── english_words.txt

saved_models/
└── <experiment-name>/
    ├── model.pth
    ├── hashNet.pth
    └── key.pt
```

The dataset pickle format expected by the original loader is:

```python
{
    "train": [{writer_id: [{"img": image, "label": text}, ...]}, ...],
    "test": [{writer_id: [{"img": image, "label": text}, ...]}, ...],
}
```

Do not commit datasets, checkpoints, generated keys, logs, or evaluation results. The repository `.gitignore` excludes them by default.

## Optional experiment tracking

Weights & Biases runs in disabled mode unless configured otherwise:

```bash
wandb login
WANDB_MODE=online python train.py
```

Authentication must be supplied by the local environment, never hard-coded in a tracked file.
