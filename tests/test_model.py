from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from ocr import CHARS, DeepCRNN, ctc_decode, load_checkpoint, preprocess

CKPT = Path(__file__).resolve().parents[1] / "deep_ocr_model.pth"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def word_image(word: str) -> np.ndarray:
    img = Image.new("L", (256, 64), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT), 40) if FONT.exists() else ImageFont.load_default()
    draw.text(((256 - draw.textlength(word, font=font)) / 2, 8), word, fill=0, font=font)
    return np.array(img)


def test_vocabulary():
    assert len(CHARS) == 94 and len(set(CHARS)) == 94


def test_forward_shape():
    out = DeepCRNN()(torch.zeros(2, 1, 32, 128))
    assert out.shape == (32, 2, len(CHARS) + 1)
    assert torch.allclose(out.exp().sum(-1), torch.ones(32, 2), atol=1e-4)


def test_ctc_decode_merges_repeats_and_drops_blanks():
    seq = [0, 1, 1, 0, 1, 2, 2, 0]  # blank a a blank a b b blank -> "aab"
    log_probs = torch.full((len(seq), 1, 3), -10.0)
    for t, i in enumerate(seq):
        log_probs[t, 0, i] = 0.0
    assert ctc_decode(log_probs, {1: "a", 2: "b"}) == ["aab"]


def test_preprocess_shape():
    x = preprocess(np.full((40, 200), 255, np.uint8))
    assert x.shape == (1, 1, 32, 128) and -1.0 <= x.min() and x.max() <= 1.0


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not present")
def test_checkpoint_reads_rendered_words():
    model, idx_to_char, meta = load_checkpoint(str(CKPT))
    assert meta["num_classes"] == len(CHARS) + 1
    with torch.no_grad():
        preds = [ctc_decode(model(preprocess(word_image(w))), idx_to_char)[0] for w in ("HELLO", "PARK", "2024")]
    assert preds == ["HELLO", "PARK", "2024"]
