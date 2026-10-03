"""Runs the full training loop for one epoch on a tiny dataset laid out like IIIT5K."""

import numpy as np
import scipy.io
from PIL import Image

from ocr import load_checkpoint
from tests.test_model import word_image
from train import train

WORDS = ["CAT", "DOG", "SUN", "PARK", "ROAD", "STOP", "EXIT", "OPEN", "BANK", "CAFE"]


def make_split(base, name, words):
    (base / name).mkdir()
    rec = np.zeros((1, len(words)), dtype=[("ImgName", "O"), ("GroundTruth", "O"), ("smallLexi", "O"), ("mediumLexi", "O")])
    for i, w in enumerate(words):
        Image.fromarray(word_image(w)).save(base / name / f"{i}.png")
        rec[0, i] = (f"{name}/{i}.png", w, np.array(["X"], dtype=object), np.array(["Y"], dtype=object))
    scipy.io.savemat(base / f"{name}data.mat", {f"{name}data": rec})


def test_one_epoch(tmp_path):
    base = tmp_path / "IIIT5K"
    base.mkdir()
    make_split(base, "train", WORDS * 2)
    make_split(base, "test", WORDS)
    out = tmp_path / "model.pth"
    logs = []
    best = train(str(tmp_path), n_train=24, n_test=6, epochs=1, batch_size=8, out=str(out), log=logs.append)
    assert logs[0] == "train 24, test 6" and 0.0 <= best <= 1.0
    model, idx_to_char, meta = load_checkpoint(str(out))
    assert meta["epoch"] == 1 and len(idx_to_char) == 94
