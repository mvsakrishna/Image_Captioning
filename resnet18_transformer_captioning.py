"""
resnet18_transformer_captioning.py
====================================
Image Captioning using ResNet-18 (encoder) + 4-layer Transformer decoder.
This is the "TransCapNet" model described in the paper.

Dataset  : Flickr8k
Framework: PyTorch

Usage
-----
  # Train (single model)
  python resnet18_transformer_captioning.py --mode train \
         --images_dir data/Images --captions data/captions.txt \
         --save_dir models/resnet_transformer --epochs 30

  # Train ensemble (5 models with different seeds)
  python resnet18_transformer_captioning.py --mode train_ensemble \
         --images_dir data/Images --captions data/captions.txt \
         --save_dir models/ensemble --epochs 30 --n_ensemble 5

  # Evaluate
  python resnet18_transformer_captioning.py --mode evaluate \
         --images_dir data/Images --captions data/captions.txt \
         --save_dir models/resnet_transformer

  # Caption one image
  python resnet18_transformer_captioning.py --mode caption \
         --image_path photo.jpg --save_dir models/resnet_transformer
"""

import os
import re
import math
import random
import argparse
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from PIL import Image

try:
    import sacrebleu
except ImportError:
    sacrebleu = None
    print("[WARN] sacrebleu not installed. Run: pip install sacrebleu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Vocabulary
# ══════════════════════════════════════════════════════════════════════════════

PAD_TOKEN   = '<pad>'
SOS_TOKEN   = '<sos>'
EOS_TOKEN   = '<eos>'
UNK_TOKEN   = '<unk>'
SPECIAL_TOKS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class Vocabulary:
    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

    def build(self, captions_list: list):
        counter = Counter()
        for cap in captions_list:
            counter.update(cap.lower().split())
        vocab = SPECIAL_TOKS + sorted(
            [w for w, c in counter.items() if c >= self.min_freq])
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self

    @property
    def pad_idx(self): return self.word2idx[PAD_TOKEN]

    @property
    def sos_idx(self): return self.word2idx[SOS_TOKEN]

    @property
    def eos_idx(self): return self.word2idx[EOS_TOKEN]

    @property
    def unk_idx(self): return self.word2idx[UNK_TOKEN]

    def encode(self, caption: str) -> list:
        tokens = [self.word2idx.get(w, self.unk_idx)
                  for w in caption.lower().split()]
        return [self.sos_idx] + tokens + [self.eos_idx]

    def decode(self, indices: list) -> str:
        words = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx in (self.sos_idx, self.pad_idx):
                continue
            words.append(self.idx2word.get(idx, UNK_TOKEN))
        return ' '.join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx,
                         'idx2word': self.idx2word}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        v = cls()
        v.word2idx = data['word2idx']
        v.idx2word = data['idx2word']
        return v


# ══════════════════════════════════════════════════════════════════════════════
# 2. Caption loading & splitting
# ══════════════════════════════════════════════════════════════════════════════

def load_captions(captions_file: str) -> dict:
    """Parse Flickr8k captions.txt → {image_id: [captions]}."""
    captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            img_id, cap = parts
            img_id = img_id.split('#')[0]
            cap = re.sub(r'\s+', ' ', cap.strip())
            captions.setdefault(img_id, []).append(cap)
    return captions


def split_ids(all_ids, train_frac=0.85, val_frac=0.075):
    n = len(all_ids)
    t = int(n * train_frac)
    v = int(n * val_frac)
    return all_ids[:t], all_ids[t:t + v], all_ids[t + v:]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class CaptionDataset(Dataset):
    """Returns (image_tensor, caption_tensor) pairs."""

    def __init__(self,
                 image_ids: list,
                 captions: dict,
                 images_dir: str,
                 vocab: Vocabulary,
                 max_len: int = 50,
                 transform=IMAGE_TRANSFORM):
        self.samples = []
        for img_id in image_ids:
            img_path = Path(images_dir) / img_id
            if not img_path.exists():
                continue
            for cap in captions.get(img_id, []):
                self.samples.append((str(img_path), cap))

        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        tokens = self.vocab.encode(caption)
        # Truncate and pad to max_len + 1 (includes sos/eos)
        tokens = tokens[:self.max_len + 1]
        pad_len = self.max_len + 1 - len(tokens)
        tokens = tokens + [self.vocab.pad_idx] * pad_len
        cap_tensor = torch.tensor(tokens, dtype=torch.long)

        return img_tensor, cap_tensor


def collate_fn(batch):
    imgs, caps = zip(*batch)
    return torch.stack(imgs), torch.stack(caps)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Model
# ══════════════════════════════════════════════════════════════════════════════

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)   # (B, 512)
        return self.norm(self.proj(feat))     # (B, embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransCapNet(nn.Module):
    """
    ResNet-18 image encoder + 4-layer Transformer decoder.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 512,
                 nhead: int = 8,
                 num_decoder_layers: int = 4,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 52):
        super().__init__()
        self.encoder  = ImageEncoder(embed_dim)
        self.embed    = nn.Embedding(vocab_size, embed_dim,
                                     padding_idx=0)
        self.pos_enc  = PositionalEncoding(embed_dim, max_len, dropout)
        dec_layer     = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True)
        self.decoder  = nn.TransformerDecoder(
            dec_layer, num_layers=num_decoder_layers)
        self.fc_out   = nn.Linear(embed_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, sz: int, device):
        return nn.Transformer.generate_square_subsequent_mask(
            sz, device=device)

    def make_padding_mask(self, captions: torch.Tensor, pad_idx: int):
        return (captions == pad_idx)   # True where padded

    def forward(self, imgs, captions, pad_idx: int = 0):
        memory = self.encoder(imgs).unsqueeze(1)      # (B, 1, E)
        tgt    = self.pos_enc(self.embed(captions))   # (B, T, E)

        sz     = captions.size(1)
        device = imgs.device
        causal_mask   = self.make_causal_mask(sz, device)
        pad_mask      = self.make_padding_mask(captions, pad_idx)

        out = self.decoder(tgt, memory,
                           tgt_mask=causal_mask,
                           tgt_key_padding_mask=pad_mask)
        return self.fc_out(out)                        # (B, T, V)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Training
# ══════════════════════════════════════════════════════════════════════════════

def train_one_model(args, seed: int = 42) -> (TransCapNet, Vocabulary):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}  |  Seed: {seed}")

    # ── Data ─────────────────────────────────────────────────────────────────
    captions = load_captions(args.captions)
    all_ids = list(captions.keys())
    train_ids, val_ids, _ = split_ids(all_ids)

    all_train_caps = [cap for img_id in train_ids
                     for cap in captions.get(img_id, [])]
    vocab = Vocabulary(min_freq=args.min_freq).build(all_train_caps)
    print(f"  Vocabulary: {len(vocab)} tokens")

    train_ds = CaptionDataset(
        train_ids, captions, args.images_dir, vocab, args.max_len)
    val_ds = CaptionDataset(
        val_ids, captions, args.images_dir, vocab, args.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TransCapNet(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_decoder_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_len + 2
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for imgs, caps in train_loader:
            imgs, caps = imgs.to(device), caps.to(device)
            inp = caps[:, :-1]   # (B, T)  — teacher forcing input
            tgt = caps[:, 1:]    # (B, T)  — ground truth

            logits = model(imgs, inp, pad_idx=vocab.pad_idx)
            loss = criterion(
                logits.reshape(-1, len(vocab)),
                tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, caps in val_loader:
                imgs, caps = imgs.to(device), caps.to(device)
                inp = caps[:, :-1]
                tgt = caps[:, 1:]
                logits = model(imgs, inp, pad_idx=vocab.pad_idx)
                loss = criterion(
                    logits.reshape(-1, len(vocab)),
                    tgt.reshape(-1))
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        avg_train = total_loss / len(train_loader)
        scheduler.step(avg_val)

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'best_seed{seed}.pt'))
            print(f"    ✓ Checkpoint saved")

    return model, vocab


# ══════════════════════════════════════════════════════════════════════════════
# 6. Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def beam_search(model: TransCapNet,
                img_tensor: torch.Tensor,
                vocab: Vocabulary,
                device: str = 'cpu',
                beam_width: int = 5,
                max_len: int = 50) -> str:
    """Beam search decoder for a single image."""
    model.eval()
    img = img_tensor.unsqueeze(0).to(device)
    memory = model.encoder(img)    # (1, E)

    # Each beam: (log_prob, token_list)
    beams = [(0.0, [vocab.sos_idx])]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == vocab.eos_idx:
                completed.append((score, seq))
                continue
            tgt = torch.tensor([seq], dtype=torch.long).to(device)
            tgt_emb = model.pos_enc(model.embed(tgt))
            mem_exp = memory.unsqueeze(1)
            sz = tgt.size(1)
            causal_mask = model.make_causal_mask(sz, device)
            out = model.decoder(tgt_emb, mem_exp,
                                tgt_mask=causal_mask)
            logits = model.fc_out(out[:, -1, :])     # (1, V)
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_vals, topk_idx = log_probs[0].topk(beam_width)

            for lp, idx in zip(topk_vals.tolist(),
                                topk_idx.tolist()):
                new_beams.append((score + lp, seq + [idx]))

        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    completed += beams
    best_score, best_seq = max(completed, key=lambda x: x[0])
    return vocab.decode(best_seq[1:])   # remove sos


def caption_image(image_path: str,
                  model: TransCapNet,
                  vocab: Vocabulary,
                  device: str = 'cpu',
                  beam_width: int = 5) -> str:
    img = Image.open(image_path).convert('RGB')
    img_tensor = IMAGE_TRANSFORM(img)
    return beam_search(model, img_tensor, vocab,
                       device, beam_width)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Ensemble inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ensemble_caption(image_path: str,
                     models: list,
                     vocab: Vocabulary,
                     device: str = 'cpu',
                     beam_width: int = 5,
                     max_len: int = 50) -> str:
    """Average logits across ensemble of models, then beam search."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(device)

    for m in models:
        m.eval()

    beams = [(0.0, [vocab.sos_idx])]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == vocab.eos_idx:
                completed.append((score, seq))
                continue
            tgt = torch.tensor([seq], dtype=torch.long).to(device)

            avg_logits = None
            for m in models:
                tgt_emb = m.pos_enc(m.embed(tgt))
                memory = m.encoder(img_tensor).unsqueeze(1)
                sz = tgt.size(1)
                mask = m.make_causal_mask(sz, device)
                out = m.decoder(tgt_emb, memory, tgt_mask=mask)
                logits = m.fc_out(out[:, -1, :])  # (1, V)
                avg_logits = logits if avg_logits is None \
                    else avg_logits + logits
            avg_logits /= len(models)

            log_probs = torch.log_softmax(avg_logits, dim=-1)
            topk_vals, topk_idx = log_probs[0].topk(beam_width)
            for lp, idx in zip(topk_vals.tolist(),
                                topk_idx.tolist()):
                new_beams.append((score + lp, seq + [idx]))

        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    completed += beams
    _, best_seq = max(completed, key=lambda x: x[0])
    return vocab.decode(best_seq[1:])


# ══════════════════════════════════════════════════════════════════════════════
# 8. Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def evaluate(args, models_list, vocab):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    captions = load_captions(args.captions)
    all_ids = list(captions.keys())
    _, _, test_ids = split_ids(all_ids)

    hypotheses, references = [], []
    for i, img_id in enumerate(test_ids, 1):
        img_path = Path(args.images_dir) / img_id
        if not img_path.exists():
            continue
        if len(models_list) == 1:
            pred = caption_image(
                str(img_path), models_list[0], vocab, device)
        else:
            pred = ensemble_caption(
                str(img_path), models_list, vocab, device)

        refs = captions.get(img_id, [])
        hypotheses.append(pred)
        references.append(refs)
        if i % 100 == 0:
            print(f"  Evaluated {i}/{len(test_ids)}")

    if sacrebleu is None:
        print("[WARN] sacrebleu not available.")
        return

    refs_t = list(zip(*references))
    bleu = sacrebleu.corpus_bleu(
        hypotheses, [list(r) for r in refs_t])
    print(f"\n  BLEU-4 (sacrebleu): {bleu.score / 100:.4f}")

    rouge_scores = []
    for hyp, refs_list in zip(hypotheses, references):
        hyp_t = hyp.split()
        best = 0.0
        for ref in refs_list:
            ref_t = ref.split()
            lcs = lcs_length(hyp_t, ref_t)
            p = lcs / (len(hyp_t) + 1e-10)
            r = lcs / (len(ref_t) + 1e-10)
            f = 2 * p * r / (p + r + 1e-10)
            best = max(best, f)
        rouge_scores.append(best)
    print(f"  ROUGE-L F         : {np.mean(rouge_scores):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="TransCapNet (ResNet18-Transformer)")
    p.add_argument('--mode', required=True,
                   choices=['train', 'train_ensemble',
                            'evaluate', 'caption'])
    p.add_argument('--images_dir',  default='data/Images')
    p.add_argument('--captions',    default='data/captions.txt')
    p.add_argument('--save_dir',    default='models/resnet_transformer')
    p.add_argument('--image_path',  default=None)
    p.add_argument('--n_ensemble',  type=int, default=5)
    # Hyper-params
    p.add_argument('--epochs',      type=int,   default=30)
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=1e-5)
    p.add_argument('--embed_dim',   type=int,   default=512)
    p.add_argument('--nhead',       type=int,   default=8)
    p.add_argument('--num_layers',  type=int,   default=4)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--max_len',     type=int,   default=50)
    p.add_argument('--min_freq',    type=int,   default=3)
    p.add_argument('--beam_width',  type=int,   default=5)
    p.add_argument('--num_workers', type=int,   default=4)
    return p.parse_args()


def load_trained_model(save_dir, vocab, args, seed, device):
    model = TransCapNet(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_decoder_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_len + 2).to(device)
    ckpt = os.path.join(save_dir, f'best_seed{seed}.pt')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.mode == 'train':
        print("── Single model training ────────────────────")
        model, vocab = train_one_model(args, seed=42)
        vocab.save(os.path.join(args.save_dir, 'vocab.pkl'))

    elif args.mode == 'train_ensemble':
        print(f"── Ensemble training ({args.n_ensemble} models) ─────")
        seeds = [42, 123, 7, 2024, 999][:args.n_ensemble]
        for seed in seeds:
            print(f"\n  ── Model seed={seed} ──────────────────────")
            model, vocab = train_one_model(args, seed=seed)
        vocab.save(os.path.join(args.save_dir, 'vocab.pkl'))
        print("\n  All ensemble models trained.")

    elif args.mode == 'evaluate':
        print("── Evaluation ──────────────────────────────")
        vocab_path = os.path.join(args.save_dir, 'vocab.pkl')
        vocab = Vocabulary.load(vocab_path)

        # Load all checkpoint files
        ckpts = list(Path(args.save_dir).glob('best_seed*.pt'))
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoints found in {args.save_dir}")

        loaded_models = []
        for ckpt in ckpts:
            seed = int(ckpt.stem.replace('best_seed', ''))
            m = load_trained_model(
                args.save_dir, vocab, args, seed, device)
            loaded_models.append(m)
        print(f"  Loaded {len(loaded_models)} model(s)")
        evaluate(args, loaded_models, vocab)

    elif args.mode == 'caption':
        print("── Inference ───────────────────────────────")
        if args.image_path is None:
            raise ValueError("--image_path required for caption mode")

        vocab_path = os.path.join(args.save_dir, 'vocab.pkl')
        vocab = Vocabulary.load(vocab_path)

        ckpts = list(Path(args.save_dir).glob('best_seed*.pt'))
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoints found in {args.save_dir}")

        loaded_models = []
        for ckpt in ckpts:
            seed = int(ckpt.stem.replace('best_seed', ''))
            m = load_trained_model(
                args.save_dir, vocab, args, seed, device)
            loaded_models.append(m)

        if len(loaded_models) == 1:
            cap = caption_image(
                args.image_path, loaded_models[0], vocab, device,
                args.beam_width)
        else:
            cap = ensemble_caption(
                args.image_path, loaded_models, vocab, device,
                args.beam_width)

        print(f"  Caption: {cap}")


if __name__ == '__main__':
    main()
