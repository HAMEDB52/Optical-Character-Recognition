# CRNN Word Recognition (IIIT5K)

A CRNN text recognizer in PyTorch that reads cropped English scene-text words: six convolutional layers extract features, a two-layer bidirectional LSTM models the character sequence, and a CTC output layer removes the need for character-level alignment.

## Result

The checkpoint in this repository, `deep_ocr_model.pth`, records **59.5% word accuracy on its validation split** (epoch 86), as saved in the checkpoint metadata. It reads clean rendered words correctly:

```
$ python predict.py samples/hello.png samples/park.png
samples/hello.png: HELLO
samples/park.png: PARK
```

IIIT5K labels are uppercase, so the model predicts uppercase text.

## Model

| | |
|---|---|
| Input | grayscale word crop resized to 32 × 128, normalised to [-1, 1] |
| CNN | 6 conv layers (64 → 512 channels) with batch norm; pooling reduces height to 1, keeping 32 time steps |
| Sequence | 2-layer bidirectional LSTM, 128 hidden units per direction |
| Output | 95 classes (94 printable ASCII characters + CTC blank), greedy CTC decoding |
| Parameters | 5,580,255 |
| Training | CTC loss, AdamW (lr 1e-3, weight decay 1e-4), cosine decay, gradient clipping at 1.0 |

## Usage

```bash
pip install -r requirements.txt

# read words
python predict.py path/to/word.png

# train (download IIIT5K-Word V3.0 from CVIT, IIIT Hyderabad, and extract it to iiit5k_dataset/IIIT5K)
python train.py --data-root iiit5k_dataset --epochs 250
```

`train.py` pools the original IIIT5K train and test sets, re-splits them with a fixed seed (4,500 / 500 by default) and saves the checkpoint with the best validation word accuracy. The notebook `OCR_Hamed_project_one.ipynb` runs the same pipeline step by step on Google Colab.

## Layout

```
ocr/model.py   model, preprocessing, CTC decoding, checkpoint loading
train.py       training on IIIT5K
predict.py     command-line inference
samples/       two rendered word images
tests/         unit tests, checkpoint test and a one-epoch training run on a tiny IIIT5K-style dataset
```

## Tests

```bash
python -m pytest
```
