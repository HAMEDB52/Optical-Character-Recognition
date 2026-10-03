"""Read the word in a cropped word image.

Usage: python predict.py IMAGE [IMAGE ...] [--model deep_ocr_model.pth]
"""

import argparse

import cv2
import torch

from ocr import ctc_decode, load_checkpoint, preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model", default="deep_ocr_model.pth")
    args = parser.parse_args()

    model, idx_to_char, _ = load_checkpoint(args.model)
    for path in args.images:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"{path}: cannot read image")
            continue
        with torch.no_grad():
            text = ctc_decode(model(preprocess(gray)), idx_to_char)[0]
        print(f"{path}: {text}")


if __name__ == "__main__":
    main()
