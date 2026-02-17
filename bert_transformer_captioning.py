"""
bert_transformer_captioning.py
===============================
Image Captioning using ViT (encoder) + BERT-base (decoder)
via HuggingFace VisionEncoderDecoderModel.

Dataset  : Flickr8k  (or any folder of images + captions.txt)
Framework: HuggingFace Transformers + PyTorch

Usage
-----
  # Fine-tune on Flickr8k
  python bert_transformer_captioning.py --mode train \
         --images_dir data/Images --captions data/captions.txt \
         --save_dir models/bert_transformer --epochs 20

  # Evaluate BLEU-4 on test split
  python bert_transformer_captioning.py --mode evaluate \
         --images_dir data/Images --captions data/captions.txt \
         --model_dir models/bert_transformer

  # Caption a single image
  python bert_transformer_captioning.py --mode caption \
         --image_path photo.jpg --model_dir models/bert_transformer
"""

import os
import re
import argparse
import math
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
)

try:
    import sacrebleu
except ImportError:
    sacrebleu = None
    print("[WARN] sacrebleu not installed. Run: pip install sacrebleu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Constants
# ══════════════════════════════════════════════════════════════════════════════

PRETRAINED_MODEL = "nlpconnect/vit-gpt2-image-captioning"
MAX_LENGTH = 64
NUM_BEAMS = 4


# ══════════════════════════════════════════════════════════════════════════════
# 2. Caption utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_captions(captions_file: str) -> dict:
    """
    Parse Flickr8k captions file.
    Returns {image_filename: [caption1, caption2, ...]}
    """
    captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            img_id, caption = parts
            img_id = img_id.split('#')[0]
            caption = re.sub(r'\s+', ' ', caption.strip())
            captions.setdefault(img_id, []).append(caption)
    return captions


def split_ids(all_ids: list,
              train_frac: float = 0.85,
              val_frac: float = 0.075):
    """Return (train_ids, val_ids, test_ids)."""
    n = len(all_ids)
    t = int(n * train_frac)
    v = int(n * val_frac)
    return all_ids[:t], all_ids[t:t + v], all_ids[t + v:]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════════

class Flickr8kDataset(Dataset):
    """One sample = (image_tensor, tokenized_caption)."""

    def __init__(self,
                 image_ids: list,
                 captions: dict,
                 images_dir: str,
                 feature_extractor,
                 tokenizer,
                 max_length: int = MAX_LENGTH):
        self.samples = []
        for img_id in image_ids:
            img_path = Path(images_dir) / img_id
            if not img_path.exists():
                continue
            for cap in captions.get(img_id, []):
                self.samples.append((str(img_path), cap))

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.feature_extractor(
            images=image, return_tensors='pt'
        ).pixel_values.squeeze(0)

        labels = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).input_ids.squeeze(0)
        # Replace padding token id with -100 so loss ignores them
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {'pixel_values': pixel_values, 'labels': labels}


def collate_fn(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    return {'pixel_values': pixel_values, 'labels': labels}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Model helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_model_components(model_dir: str = None, device: str = 'cuda'):
    """
    Load model, feature extractor, and tokenizer.
    If model_dir is given, load fine-tuned weights; otherwise use pretrained.
    """
    src = model_dir if (model_dir and os.path.isdir(model_dir)) \
        else PRETRAINED_MODEL
    print(f"  Loading from: {src}")
    model = VisionEncoderDecoderModel.from_pretrained(src).to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        PRETRAINED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    return model, feature_extractor, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 5. Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    model, feat_ext, tokenizer = load_model_components(None, device)

    captions = load_captions(args.captions)
    all_ids = list(captions.keys())
    train_ids, val_ids, _ = split_ids(all_ids)

    train_ds = Flickr8kDataset(
        train_ids, captions, args.images_dir, feat_ext, tokenizer)
    val_ds = Flickr8kDataset(
        val_ids, captions, args.images_dir, feat_ext, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                pixel_values=pixel_values,
                labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels)
                val_loss += outputs.loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"  Epoch {epoch}/{args.epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(args.save_dir)
            feat_ext.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print(f"    ✓ Saved best model (val_loss={best_val_loss:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Inference
# ══════════════════════════════════════════════════════════════════════════════

def generate_caption(image_path: str,
                     model,
                     feat_ext,
                     tokenizer,
                     device: str = 'cpu',
                     max_length: int = MAX_LENGTH,
                     num_beams: int = NUM_BEAMS) -> str:
    """Generate caption for a single image."""
    img = Image.open(image_path).convert('RGB')
    pixel_values = feat_ext(
        images=img, return_tensors='pt'
    ).pixel_values.to(device)

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True)

    caption = tokenizer.decode(
        output_ids[0], skip_special_tokens=True)
    return caption.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, feat_ext, tokenizer = load_model_components(
        args.model_dir, device)

    captions = load_captions(args.captions)
    all_ids = list(captions.keys())
    _, _, test_ids = split_ids(all_ids)

    hypotheses, references = [], []
    for i, img_id in enumerate(test_ids, 1):
        img_path = Path(args.images_dir) / img_id
        if not img_path.exists():
            continue
        pred = generate_caption(
            str(img_path), model, feat_ext, tokenizer, device)
        refs = captions.get(img_id, [])
        hypotheses.append(pred)
        references.append(refs)
        if i % 50 == 0:
            print(f"  Evaluated {i}/{len(test_ids)}")

    if sacrebleu is None:
        print("[WARN] sacrebleu not available; BLEU not computed.")
        return

    refs_t = list(zip(*references))
    bleu = sacrebleu.corpus_bleu(
        hypotheses, [list(r) for r in refs_t])
    print(f"\n  BLEU-4 (sacrebleu): {bleu.score / 100:.4f}")

    # Simple ROUGE-L (token-level LCS)
    rouge_scores = []
    for hyp, refs_list in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        best_f = 0.0
        for ref in refs_list:
            ref_tokens = ref.split()
            lcs = lcs_length(hyp_tokens, ref_tokens)
            if lcs == 0:
                continue
            precision = lcs / len(hyp_tokens)
            recall = lcs / len(ref_tokens)
            f = (2 * precision * recall /
                 (precision + recall + 1e-10))
            best_f = max(best_f, f)
        rouge_scores.append(best_f)
    print(f"  ROUGE-L F         : {np.mean(rouge_scores):.4f}")


def lcs_length(a: list, b: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BERT-Transformer Image Captioning")
    p.add_argument('--mode', required=True,
                   choices=['train', 'evaluate', 'caption'])
    p.add_argument('--images_dir',  default='data/Images')
    p.add_argument('--captions',    default='data/captions.txt')
    p.add_argument('--save_dir',    default='models/bert_transformer')
    p.add_argument('--model_dir',   default='models/bert_transformer')
    p.add_argument('--image_path',  default=None)
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=5e-5)
    p.add_argument('--num_workers', type=int,   default=4)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == 'train':
        print("── Training ────────────────────────────────")
        train(args)

    elif args.mode == 'evaluate':
        print("── Evaluation ──────────────────────────────")
        evaluate(args)

    elif args.mode == 'caption':
        print("── Inference ───────────────────────────────")
        if args.image_path is None:
            raise ValueError("--image_path required for caption mode")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, feat_ext, tokenizer = load_model_components(
            args.model_dir, device)
        cap = generate_caption(
            args.image_path, model, feat_ext, tokenizer, device)
        print(f"  Caption: {cap}")


if __name__ == '__main__':
    main()
