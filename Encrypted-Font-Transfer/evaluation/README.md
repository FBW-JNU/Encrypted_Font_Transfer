# Evaluation scripts

Run these commands from the repository root with the project's dependencies installed. Make three copies of `evaluation/config.example.json`, supply your own checkpoint, IAM test pickle, English word list, and 512-bit registered key, and give each copy a different `output`. Keep the same `sample_manifest` in all three copies so references, texts, and real images are reused. All scripts default to 1,000 fixed target images. Each run writes `config.json`, `per_image.jsonl`, `summary.csv`, `summary.json`, and `summary.tex`; rerunning the same config resumes completed rows. All relative paths in a config are resolved relative to that config file, regardless of the shell's working directory. The example paths are placeholders; your files can live anywhere.

```bash
python evaluation/evaluate_key_control.py --config evaluation/config_key_control.json
python evaluation/evaluate_watermark_robustness.py --config evaluation/config_robustness.json
python evaluation/evaluate_unseen_attacks.py --config evaluation/config_unseen.json
```

For example, set the three `output` values to `../outputs/key_control`, `../outputs/robustness`, and `../outputs/unseen`, respectively. For a quick real-data check, set `sample_count` to 2 and `methods` to `["Ours"]`. To evaluate all available methods, list them in `methods` and supply `external_methods` factories. Missing external methods are reported as N/A; no baseline results are invented.

An external factory is written as `"package.module:build_method"`. It receives `config`, `key` (512 bits), and `bits` (128 projected bits) and returns an object with `embed(references, text, bits, seed)` and `extract(image)`. For HopSkipJump, it must also implement `verify(image)` returning a binary accept/reject decision from its actual calibrated verifier. The alias `InstancWmk` is accepted for `InstanceWmk`.

## Measurements

`evaluate_key_control.py` tests exact correct-key fractions of 0%, 25%, 50%, 75%, and 100% by flipping all other bits. Generation is reseeded identically per sample across key fractions. It computes FID once per full image set against the fixed real set, writer classification cross entropy from `netW`, and OCR CTC loss from `netOCR`. Generated images are saved under `images/`.

The supplied IAM pickle has disjoint train and test writer IDs. Since `netW` only has training-writer labels, a valid `L_cls` on those test writers is unavailable and is reported as N/A. A separately trained classifier with matching test-writer labels would be needed to populate that metric.

`evaluate_watermark_robustness.py` includes a clean condition and the configured AWGN, salt and pepper, Gaussian blur, motion blur, JPEG, rotation, perspective warping, and resize grid. Each condition starts from the clean watermarked image. Random parameters are derived from the sample and condition, so all methods receive matching noise/geometry. The extractor output is thresholded at 0.5 and compared to the 128-bit `keyHashNet` projection, never to the raw 512-bit key.

`evaluate_unseen_attacks.py` evaluates `x^1.4` Gamma on grayscale `[0,1]`, JPEG quality 50 followed by resize 0.75 followed by 5-degree rotation, and ART's decision-based HopSkipJump when a binary verification method is available. The 156-query cap covers initialization, search, and final decision checks; exhausted attacks are recorded as failures with their query counts. `training_attack_exposure` is recorded as not verifiable unless a training-attack list is supplied. A missing binary verifier or ART installation yields N/A for HopSkipJump.

If available, set `training_config` to a JSON training configuration containing an `attacks` or `noise_layers` list. The script records which attacks are listed; an absent entry is still not proof that the attack was unseen during training.

The shared attack defaults are documented in `config.example.json` and `attacks.py`. Attacks operate on grayscale `[0,1]`, clamp to that range, and convert back to model `[-1,1]`. Gaussian blur uses a 7×7 kernel by default, motion blur uses the project's `MotionBlur` with 25-degree direction and zero convolution padding, JPEG uses Pillow grayscale encoding, rotation uses bilinear interpolation and white fill, perspective warping moves each corner by at most `strength` times the image dimensions with bilinear interpolation and white fill, and resize uses bilinear down/up sampling with `align_corners=False`. These defaults fill details that were not specified in the requested protocol; they are not claimed to be historical paper settings.

The model expects grayscale, 32×192 images normalized to `[-1,1]`, with white right padding. Generation and extraction use the checkpoint's trained projection and extractor. For meaningful comparisons, the checkpoint, key, and IAM pickle must belong to the same experiment. Main-model checkpoints and external baselines are not included in this source package.

## Independent training runs

To evaluate multiple independently trained checkpoints, add `runs` to the config. Each entry supplies `checkpoint`, `dataset`, `key`, `sample_manifest`, `split_id`, and `independent_training_run: true`. Outputs go to `run_01`, `run_02`, etc. Mean and 95% t intervals are written only when at least five entries have distinct checkpoint hashes, split IDs, and sample-manifest hashes. Repeating one checkpoint with different seeds does not create an independent-training interval.

```json
"runs": [
  {"checkpoint": "../saved_models/run-1/model.pth", "dataset": "../files/split-1.pickle", "key": "../saved_models/run-1/key.pt", "sample_manifest": "outputs/split-1.json", "split_id": "split-1", "independent_training_run": true}
]
```

The list above illustrates the format; supply all five or more actual independent runs to obtain a confidence interval. Install `adversarial-robustness-toolbox` separately if running HopSkipJump with a real binary verifier.

Run a multi-checkpoint configuration with the same entry points, for example `python evaluation/evaluate_watermark_robustness.py --config evaluation/multi_runs.json`. The combined interval files are `across_runs.json`, `across_runs.csv`, and `across_runs.tex`.
