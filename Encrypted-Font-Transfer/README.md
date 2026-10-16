# Encrypted_Font_Transfer

Encrypted Font Transfer extends Handwriting Transformers with key-conditioned watermark embedding, extraction, robustness evaluation, and multi-user attribution experiments.

This public source package intentionally excludes datasets, model weights, user keys, experiment logs, and reported result files. Download or generate those assets locally and keep them in the ignored paths described below.

## Repository layout

```text
.
├── data/                         # Dataset loaders and preparation tools
├── demo.py                       # One-image generation and extraction example
├── evaluation/                   # Key, robustness, and new-attack evaluation
├── models/                       # Generator, extractor, OCR, and noise modules
├── multi_user_experiments/
│   └── multi_user_attribution_experiment/
├── pretrained_models/
│   └── multi_user_attribution/    # Ten ready-to-load detector weights
└── train.py                      # Main encrypted font-transfer training entry point
```


The attribution code trains one CNN detector per registered user.
The detectors have separate parameters and are optimized jointly
using pre-generated images and their source labels. The number
of registered users is specified in the configuration.

Closed-set attribution selects the user with the highest sigmoid
score without a rejection threshold. The code also supports
open-set evaluation with a validation-calibrated rejection threshold.

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


## Minimal demo

With a trained main-model checkpoint, its matching 512-bit registered key, the test-set pickle, and the word list available locally, run:

```bash
python demo.py --checkpoint saved_models/your-experiment/model.pth \
  --key saved_models/your-experiment/key.pt \
  --dataset files/IAM-32.pickle --english-words files/english_words.txt
```

The demo picks one reproducible test sample, generates `outputs/demo/generated.png`, extracts the checkpoint's 128-bit identity code from the saved image, and prints its BER. Use `--output`, `--seed`, and `--device` to change those defaults. The checkpoint, key, and dataset must belong to a compatible experiment; this source package does not include a main-model checkpoint or private registered key. Keep those assets and generated samples out of Git. This is a functional illustration, not a substitute for the full evaluation scripts. See [DEMO.md](DEMO.md) for complete usage and interpretation notes.


## License

See [LICENSE](LICENSE).
