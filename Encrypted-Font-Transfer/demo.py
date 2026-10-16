import argparse
import random
from pathlib import Path

import torch

from evaluation.common import (ber_result, generate, load_dataset, load_key,
                               load_model, projected_bits, read_png,
                               sample_tensors, save_png)


ROOT = Path(__file__).resolve().parent


def choose_sample(data, seed):
    candidates = []
    for writer, rows in sorted(data.items()):
        if len(rows) < 2:
            continue
        for index, row in enumerate(rows):
            text = row["label"]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            if 2 <= len(str(text)) <= 12 and sum(char.isalpha() for char in str(text)) >= 2:
                candidates.append((writer, index, str(text)))
    if not candidates:
        raise ValueError("test split has no usable target with another reference image")
    rng = random.Random(seed)
    writer, index, text = rng.choice(candidates)
    other_indices = [i for i in range(len(data[writer])) if i != index]
    return {"writer": writer, "target_index": index,
            "reference_indices": [rng.choice(other_indices) for _ in range(15)],
            "text": text}


def main():
    parser = argparse.ArgumentParser(description="Generate one image and calculate its 128-bit identity-code BER")
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained main-model checkpoint")
    parser.add_argument("--dataset", type=Path, required=True, help="IAM/CVL dataset pickle")
    parser.add_argument("--key", type=Path, required=True, help="matching 512-bit registered key")
    parser.add_argument("--english-words", type=Path, required=True, help="English word-list file")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    image_path = output / "generated.png"
    if image_path.exists():
        parser.error(f"output image already exists: {image_path}; choose another --output")

    config = {"checkpoint": str(args.checkpoint.expanduser().resolve()),
              "key": str(args.key.expanduser().resolve()),
              "english_words": str(args.english_words.expanduser().resolve()),
              "device": args.device}
    data = load_dataset(args.dataset.expanduser().resolve())
    sample = choose_sample(data, args.seed)
    references, _ = sample_tensors(data, sample)
    key = load_key(config["key"])
    model = load_model(config)
    target_bits = projected_bits(model, key, args.device)

    image = generate(model, references, sample["text"], key, args.device, args.seed)
    save_png(image, image_path)
    saved_image = read_png(image_path).unsqueeze(0).to(args.device)
    with torch.no_grad():
        extracted = model.netExt(saved_image)[0]
    result = ber_result(extracted, target_bits)

    print(f"Generated image: {image_path}")
    print(f"Target text: {sample['text']}")
    print(f"128-bit identity-code BER: {result['error_bits']}/{result['total_bits']} = {result['ber']:.6f}")


if __name__ == "__main__":
    main()
