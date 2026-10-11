# Multi-user experiments


- `multi_user_attribution_experiment/` trains a single open-set model that attributes a watermarked image to a registered user or rejects it as unknown.

Use disjoint training, validation, and final-test samples. Thresholds and model selection must use validation data only; final-test data is for one-time reporting.
