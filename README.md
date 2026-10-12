# Encrypted_Font_Transfer

Encrypted Font Transfer extends Handwriting Transformers with key-conditioned watermark embedding, extraction, robustness evaluation, and multi-user attribution experiments.

This public source package intentionally excludes datasets, model weights, user keys, experiment logs, and reported result files. Download or generate those assets locally and keep them in the ignored paths described below.

## Repository layout

```text
.
├── data/                         # Dataset loaders and preparation tools
├── models/                       # Generator, extractor, OCR, and noise modules
├── multi_user_experiments/
│   └── multi_user_attribution_experiment/
└── train.py                      # Main encrypted font-transfer training entry point
```

The included multi-user attribution experiment trains one joint open-set model over all registered users and rejects unknown inputs using a validation-calibrated presence threshold.

## Installation

Python 3.10 or newer is recommended. Create an isolated environment and run:

```bash
python -m pip install -r requirements.txt
```

See [INSTALL.md](INSTALL.md) for the expected local assets and directory structure.

## Main training

Place the IAM or CVL pickle and `english_words.txt` under `files/`, then review `params.py` and run:

```bash
python train.py
```

Weights, keys, generated images, and logs are written only to ignored local directories. Weights & Biases is disabled by default. To enable it, authenticate outside the repository and set `WANDB_MODE=online`; never store an API key in source code.

The model trained using this code is available as `model.pth` in [Releases](https://github.com/FBW-JNU/Encrypted_Font_Transfer/releases).


## Multi-user experiments


```bash
cd multi_user_experiments/multi_user_attribution_experiment
cp config.example.json config.json
python main.py train --config config.json
```

Private user keys, registered templates, base checkpoints, and datasets must remain local. 

Checkpoints for the ten independently protected model instances are available in the [multi-user release](https://github.com/FBW-JNU/Encrypted_Font_Transfer/releases#release-multi_user). Each checkpoint corresponds to a separately trained model associated with a different user identity. These checkpoints support evaluation of watermark extraction and closed-set source attribution across the ten registered identities.


## License

See [LICENSE](LICENSE).
