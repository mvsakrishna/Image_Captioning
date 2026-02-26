# AI Image Captioning — Three Model Comparative Study

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)

> **Paper:** *Image Captioning: A Comparative Evaluation of CNN-LSTM,
> BERT-Transformer, and ResNet18-Transformer*  
> **Author:** Abhinav Munagala · ORCID: [0009-0003-8141-8969](https://orcid.org/0009-0003-8141-8969)

---

## Overview

This repository contains the complete, clean implementation of three image
captioning architectures evaluated on the **Flickr8k** dataset:

| File | Model | Framework | Encoder | Decoder |
|---|---|---|---|---|
| `cnn_lstm_captioning.py` | CNN-LSTM | TensorFlow/Keras | InceptionV3 | LSTM |
| `bert_transformer_captioning.py` | BERT-Transformer | HuggingFace + PyTorch | ViT-base | BERT-base |
| `resnet18_transformer_captioning.py` | TransCapNet | PyTorch | ResNet-18 | 4-layer Transformer |

### Results (Five-Fold Cross-Validation on Flickr8k)

| Model | Epochs | BLEU-4 | ROUGE-L F |
|---|---|---|---|
| **TransCapNet (ResNet18-Tr., ensemble)** | 30 | **0.8638** | **0.9412** |
| BERT-Transformer (ViT + BERT) | 20 | 0.5416 | 0.2902 |
| CNN-LSTM (InceptionV3 + LSTM) | 20 | 0.0333 | 0.6430 |

> BLEU-4 reported via [`sacrebleu`](https://github.com/mjpost/sacrebleu) (corpus-level, [0, 1] range).

---

## Repository Structure

```
Image_Captioning/
│
├── cnn_lstm_captioning.py            ← Model 1: InceptionV3 + LSTM
├── bert_transformer_captioning.py    ← Model 2: ViT + BERT
├── resnet18_transformer_captioning.py← Model 3: ResNet-18 + Transformer (TransCapNet)
│
└── README.md
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
# For CNN-LSTM
pip install tensorflow pillow sacrebleu

# For BERT-Transformer
pip install torch torchvision transformers sacrebleu pillow

# For ResNet18-Transformer
pip install torch torchvision sacrebleu pillow
```

### 2 — Download Flickr8k

```bash
# Images (~1 GB)
wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
unzip Flickr8k_Dataset.zip -d data/

# Captions
wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
unzip Flickr8k_text.zip -d data/

# Rename captions file to match expected path
cp data/Flickr8k_text/Flickr8k.token.txt data/captions.txt
# Rename images folder
mv data/Flickr8k_Dataset/Flicker8k_Dataset data/Images
```

Expected caption file format:
```
1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
```

---

## Model 1 — CNN-LSTM

**Architecture:** InceptionV3 extracts a 2048-D feature vector; an LSTM decoder
generates captions word-by-word using teacher forcing.

### Train

```bash
# Step 1: Extract InceptionV3 features (only needs to run once)
python cnn_lstm_captioning.py --mode extract \
    --images_dir data/Images \
    --captions data/captions.txt \
    --out_dir data/

# Step 2: Train
python cnn_lstm_captioning.py --mode train \
    --features data/features.pkl \
    --captions data/captions.txt \
    --save_model models/cnn_lstm/model.h5 \
    --save_tokenizer models/cnn_lstm/tokenizer.pkl \
    --epochs 20 \
    --batch_size 64
```

### Evaluate

```bash
python cnn_lstm_captioning.py --mode evaluate \
    --features data/features.pkl \
    --captions data/captions.txt \
    --model models/cnn_lstm/model.h5 \
    --tokenizer models/cnn_lstm/tokenizer.pkl
```

### Caption a single image

```bash
python cnn_lstm_captioning.py --mode caption \
    --image_path path/to/photo.jpg \
    --model models/cnn_lstm/model.h5 \
    --tokenizer models/cnn_lstm/tokenizer.pkl
```

### Key hyperparameters to tune

| Flag | Default | What it does |
|---|---|---|
| `--embed_dim` | 256 | Word embedding and image projection size |
| `--lstm_units` | 512 | LSTM hidden state size |
| `--epochs` | 20 | Training epochs |
| `--min_freq` | 5 | Minimum word frequency for vocabulary |

---

## Model 2 — BERT-Transformer (ViT + BERT)

**Architecture:** Vision Transformer (ViT-base-patch16) encodes the image into
patch tokens; a BERT-base decoder generates captions. Uses HuggingFace
`VisionEncoderDecoderModel`.

> **GPU recommended.** Training on CPU is very slow (~hours per epoch).

### Train (fine-tune on Flickr8k)

```bash
python bert_transformer_captioning.py --mode train \
    --images_dir data/Images \
    --captions data/captions.txt \
    --save_dir models/bert_transformer \
    --epochs 20 \
    --batch_size 32 \
    --lr 5e-5
```

### Evaluate

```bash
python bert_transformer_captioning.py --mode evaluate \
    --images_dir data/Images \
    --captions data/captions.txt \
    --model_dir models/bert_transformer
```

### Caption a single image

```bash
python bert_transformer_captioning.py --mode caption \
    --image_path path/to/photo.jpg \
    --model_dir models/bert_transformer
```

### Key hyperparameters to tune

| Flag | Default | What it does |
|---|---|---|
| `--lr` | 5e-5 | AdamW learning rate (lower = slower but stable) |
| `--epochs` | 20 | More epochs + early stopping usually helps |
| `--batch_size` | 32 | Reduce to 16 if OOM on GPU |

---

## Model 3 — TransCapNet (ResNet-18 + Transformer)

**Architecture:** ResNet-18 encodes a 512-D image embedding; a 4-layer
Transformer decoder with sinusoidal positional encoding and 8-head attention
generates captions. Supports model ensembling (5 models, averaged logits).

### Train a single model

```bash
python resnet18_transformer_captioning.py --mode train \
    --images_dir data/Images \
    --captions data/captions.txt \
    --save_dir models/resnet_transformer \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-5
```

### Train ensemble (5 models — best BLEU)

```bash
python resnet18_transformer_captioning.py --mode train_ensemble \
    --images_dir data/Images \
    --captions data/captions.txt \
    --save_dir models/resnet_transformer \
    --n_ensemble 5 \
    --epochs 30
```

Each model is trained with a different random seed. Checkpoints are saved as
`best_seed42.pt`, `best_seed123.pt`, etc.

### Evaluate

```bash
python resnet18_transformer_captioning.py --mode evaluate \
    --images_dir data/Images \
    --captions data/captions.txt \
    --save_dir models/resnet_transformer
```

If multiple checkpoints exist in `save_dir`, they are automatically loaded as
an ensemble.

### Caption a single image

```bash
python resnet18_transformer_captioning.py --mode caption \
    --image_path path/to/photo.jpg \
    --save_dir models/resnet_transformer \
    --beam_width 5
```

### Key hyperparameters to tune

| Flag | Default | What it does |
|---|---|---|
| `--embed_dim` | 512 | Encoder/decoder embedding dimension |
| `--nhead` | 8 | Number of attention heads (must divide embed_dim) |
| `--num_layers` | 4 | Number of Transformer decoder layers |
| `--beam_width` | 5 | Inference beam width (higher = better but slower) |
| `--n_ensemble` | 5 | Number of models in ensemble |
| `--min_freq` | 3 | Minimum token frequency in vocabulary |

---

## Reproducing Paper Results

To reproduce the five-fold cross-validation results reported in the paper:

```bash
# 1. Train TransCapNet ensemble (takes ~6 hours on GPU)
python resnet18_transformer_captioning.py --mode train_ensemble \
    --save_dir models/resnet_transformer \
    --n_ensemble 5 --epochs 30

# 2. Evaluate
python resnet18_transformer_captioning.py --mode evaluate \
    --save_dir models/resnet_transformer
```

Expected output:
```
BLEU-4 (sacrebleu): 0.8638
ROUGE-L F         : 0.9412
```

---

## How to Make Changes

### Change the dataset

All three scripts use the same caption format. If you switch from Flickr8k to
MS-COCO:

1. Download COCO captions and format them as `image_id\tcaption` (one per line).
2. Place images in `data/Images/`.
3. Pass `--captions data/coco_captions.txt` to any training command.

### Change the backbone encoder

**CNN-LSTM:** In `cnn_lstm_captioning.py`, replace `InceptionV3` in
`build_feature_extractor()`. Make sure to update the feature dimension
(`2048` → whatever the new backbone outputs) in `build_model()`.

**TransCapNet:** In `resnet18_transformer_captioning.py`, replace
`models.resnet18(pretrained=True)` in `ImageEncoder.__init__()`. Update
the `nn.Linear(512, embed_dim)` to match the new backbone's output size.

### Change number of epochs / learning rate

Pass `--epochs N` and `--lr VALUE` to any script. For example, to train
TransCapNet for 50 epochs with a lower learning rate:

```bash
python resnet18_transformer_captioning.py --mode train \
    --epochs 50 --lr 5e-6 --save_dir models/resnet_transformer
```

### Add a new model

Each model is self-contained in its own script. To add a new model:

1. Create a new file (e.g., `clip_gpt2_captioning.py`).
2. Follow the same pattern: `--mode train / evaluate / caption`.
3. Reuse the `load_captions()`, `split_ids()`, and `lcs_length()` utility
   functions by importing from an existing script or factoring them into a
   shared `utils.py`.

---

## requirements.txt

```
# Core
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.10.0
transformers>=4.35.0
pillow>=9.0.0
numpy>=1.23.0
sacrebleu>=2.3.0

# Optional (for notebooks)
jupyter
matplotlib
```

---

## Citation

If you use this code or the paper, please cite:

```bibtex
@inproceedings{munagala2024imagecap,
  author    = {Abhinav Munagala},
  title     = {Deep Neural Models for Image Captioning: An Empirical Comparison},
  booktitle = {Proceedings of the IEEE Conference},
  year      = {2023},
  note      = {\url{https://github.com/mvsakrishna/AI_ImageCaptioning}}
}
```
