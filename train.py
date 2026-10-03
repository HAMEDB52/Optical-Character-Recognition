"""Train the CRNN on IIIT5K.

Expects the extracted dataset at DATA_ROOT/IIIT5K with traindata.mat, testdata.mat and the train/ and test/ image
folders (download IIIT5K-Word_V3.0 from CVIT, IIIT Hyderabad). The original train and test sets are pooled and
re-split with a fixed seed.

Usage: python train.py --data-root iiit5k_dataset [--train-samples 4500] [--test-samples 500] [--epochs 250]
"""

from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from ocr import CHARS, DeepCRNN, ctc_decode, preprocess

CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}


def _to_str(value) -> str:
    """Extract a string from a scipy.io.loadmat field (str, char array or nested object array)."""
    while isinstance(value, np.ndarray) and value.dtype == object and value.size:
        value = value.flat[0]
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "US":
            return str(value.flat[0]).strip() if value.size else ""
        return "".join(chr(int(c)) for c in value.flat if 32 <= int(c) <= 126).strip()
    return str(value).strip()


def load_split(mat_file: str, image_dir: str, key: str) -> list[tuple[str, str]]:
    data = scipy.io.loadmat(mat_file)[key].ravel()
    names = data.dtype.names or ()
    img_f = "ImgName" if "ImgName" in names else 0
    txt_f = "GroundTruth" if "GroundTruth" in names else 1
    samples = []
    for rec in data:
        path = os.path.join(image_dir, os.path.basename(_to_str(rec[img_f])))
        text = "".join(c for c in _to_str(rec[txt_f]) if c in CHAR_TO_IDX)
        if os.path.exists(path) and 1 <= len(text) <= 23:
            samples.append((path, text))
    return samples


class WordDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return preprocess(gray)[0], [CHAR_TO_IDX[c] for c in text], text


def collate(batch):
    images, targets, texts = zip(*batch)
    return torch.stack(images), torch.tensor([t for seq in targets for t in seq]), torch.tensor([len(t) for t in targets]), texts


def word_and_char_accuracy(pred: list[str], true: list[str]) -> tuple[float, float]:
    word = sum(p == t for p, t in zip(pred, true)) / max(1, len(true))
    total = sum(max(len(p), len(t)) for p, t in zip(pred, true))
    correct = sum(a == b for p, t in zip(pred, true) for a, b in zip(p, t))
    return word, correct / max(1, total)


def build_splits(root: str, n_train: int, n_test: int, seed: int = 42):
    base = os.path.join(root, "IIIT5K")
    pool = load_split(os.path.join(base, "traindata.mat"), os.path.join(base, "train"), "traindata")
    pool += load_split(os.path.join(base, "testdata.mat"), os.path.join(base, "test"), "testdata")
    random.Random(seed).shuffle(pool)
    n_test = min(n_test, len(pool) // 5)
    n_train = min(n_train, len(pool) - n_test)
    return pool[:n_train], pool[n_train:n_train + n_test]


def train(root, n_train=4500, n_test=500, epochs=250, batch_size=32, lr=1e-3, out="deep_ocr_model.pth", log=print):
    train_samples, test_samples = build_splits(root, n_train, n_test)
    if not train_samples or not test_samples:
        raise SystemExit(f"No usable samples found under {root}/IIIT5K")
    log(f"train {len(train_samples)}, test {len(test_samples)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(WordDataset(train_samples), batch_size, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(WordDataset(test_samples), batch_size, collate_fn=collate)

    model = DeepCRNN().to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for images, targets, lengths, _ in train_dl:
            log_probs = model(images.to(device))
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, lengths)
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for images, _, _, texts in test_dl:
                preds += ctc_decode(model(images.to(device)).cpu(), IDX_TO_CHAR)
                truths += texts
        word, char = word_and_char_accuracy(preds, truths)
        log(f"epoch {epoch:3d}  loss {total / len(train_dl):.4f}  word acc {word:.4f}  char acc {char:.4f}")
        if word > best:
            best = word
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_word_acc": word,
                        "idx_to_char": {0: "<BLANK>", **IDX_TO_CHAR}, "num_classes": len(CHARS) + 1}, out)
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the CRNN on IIIT5K")
    p.add_argument("--data-root", default="iiit5k_dataset")
    p.add_argument("--train-samples", type=int, default=4500)
    p.add_argument("--test-samples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="deep_ocr_model.pth")
    a = p.parse_args()
    print(f"best word accuracy: {train(a.data_root, a.train_samples, a.test_samples, a.epochs, a.batch_size, out=a.out):.4f}")
