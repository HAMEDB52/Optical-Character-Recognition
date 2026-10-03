"""CRNN word recognizer: a CNN feature extractor, a 2-layer bidirectional LSTM and a CTC output layer."""

from __future__ import annotations

import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# Index 0 is the CTC blank; characters start at 1.
CHARS = string.ascii_letters + string.digits + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
IMG_HEIGHT, IMG_WIDTH = 32, 128

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_HEIGHT, IMG_WIDTH), antialias=True),
    T.Grayscale(),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])


class DeepCRNN(nn.Module):
    def __init__(self, img_height: int = IMG_HEIGHT, num_classes: int = len(CHARS) + 1, hidden_size: int = 128):
        super().__init__()

        def block(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU()]
            return layers + ([nn.MaxPool2d(*pool)] if pool else [])

        self.cnn = nn.Sequential(
            *block(1, 64, (2, 2)), *block(64, 128, (2, 2)), *block(128, 256, None),
            *block(256, 256, ((2, 1), (2, 1))), *block(256, 512, ((2, 1), (2, 1))), *block(512, 512, ((2, 1), (2, 1))),
        )
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, 32, 128] → log-probabilities [T=32, B, num_classes]."""
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        seq, _ = self.rnn(conv.view(b, c * h, w).permute(2, 0, 1))
        return F.log_softmax(self.classifier(self.dropout(seq)), dim=2)


def preprocess(gray: np.ndarray) -> torch.Tensor:
    """uint8 grayscale image [H, W] → normalised tensor [1, 1, 32, 128]."""
    return _TRANSFORM(np.stack([gray] * 3, axis=-1)).unsqueeze(0)


def ctc_decode(log_probs: torch.Tensor, idx_to_char: dict[int, str], blank: int = 0) -> list[str]:
    """Greedy CTC decoding of [T, B, C] log-probabilities: merge repeats, drop blanks."""
    out = []
    for seq in log_probs.argmax(dim=2).T.tolist():
        chars, prev = [], blank
        for idx in seq:
            if idx != blank and idx != prev and idx in idx_to_char:
                chars.append(idx_to_char[idx])
            prev = idx
        out.append("".join(chars))
    return out


def load_checkpoint(path: str, device: str = "cpu") -> tuple[DeepCRNN, dict[int, str], dict]:
    """Load a checkpoint saved by the training notebook; returns (model in eval mode, idx_to_char, metadata)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    idx_to_char = {i: c for i, c in ck["idx_to_char"].items() if i != 0}
    model = DeepCRNN(num_classes=ck["num_classes"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    meta = {k: ck[k] for k in ("epoch", "val_word_acc", "num_classes") if k in ck}
    return model, idx_to_char, meta
