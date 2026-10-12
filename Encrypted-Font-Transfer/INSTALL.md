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

The original IAM/CVL data and main generator weights are not redistributed here. The pretrained ten-user attribution detectors are included under `pretrained_models/multi_user_attribution/`. Keep other local assets in the following ignored structure:

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

Training locates `files/`, `saved_models/`, and `saved_images/` relative to the repository, not the shell's current directory. Evaluation paths are relative to the evaluation config file; multi-user experiment paths are relative to its config file. Absolute paths are also accepted in configs when assets are stored elsewhere.

The legacy IAM preparation helper takes explicit `--iam-root`, `--train-index`, and `--test-index` paths; optional `--word-images` and `--xmls` override locations under the IAM root. It recreates output directories, so review its output path before running it on valuable data. `data/create_data.py` only lists CVL word-image paths given `--cvl-root`; it does not create a training pickle.

Do not commit datasets, training checkpoints, generated keys, logs, or evaluation results. The repository `.gitignore` excludes them by default and allows only the ten published detector state dictionaries.

## Optional experiment tracking

Weights & Biases runs in disabled mode unless configured otherwise:

```bash
wandb login
WANDB_MODE=online python train.py
```

Authentication must be supplied by the local environment, never hard-coded in a tracked file.
