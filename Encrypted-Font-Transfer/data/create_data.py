"""List CVL word images under a user-supplied dataset root."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvl-root", type=Path, required=True,
                        help="Directory containing cvl-database-1-1")
    args = parser.parse_args()
    root = args.cvl_root.expanduser().resolve()
    images = sorted((root / "cvl-database-1-1").glob("*/words/*/*tif"))
    if not images:
        parser.error(f"no CVL word images found under {root}")
    for image in images:
        print(image)


if __name__ == "__main__":
    main()
