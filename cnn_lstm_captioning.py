"""
cnn_lstm_captioning.py
======================
Image Captioning using InceptionV3 (encoder) + LSTM (decoder)
Dataset : Flickr8k  (https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)
Framework: TensorFlow / Keras

Usage
-----
  # 1. Extract features (only once)
  python cnn_lstm_captioning.py --mode extract --images_dir data/Images \
         --captions data/captions.txt --out_dir data/

  # 2. Train
  python cnn_lstm_captioning.py --mode train \
         --features data/features.pkl --captions data/captions.txt \
         --save_model models/cnn_lstm.h5 --save_tokenizer models/tokenizer.pkl

  # 3. Evaluate (BLEU-4 via sacrebleu)
  python cnn_lstm_captioning.py --mode evaluate \
         --features data/features.pkl --captions data/captions.txt \
         --model models/cnn_lstm.h5 --tokenizer models/tokenizer.pkl

  # 4. Caption a single image
  python cnn_lstm_captioning.py --mode caption --image_path photo.jpg \
         --model models/cnn_lstm.h5 --tokenizer models/tokenizer.pkl
"""

import os
import re
import math
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

# ── TF / Keras imports ─────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, Add)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# sacrebleu for standardised BLEU scoring
try:
    import sacrebleu
except ImportError:
    sacrebleu = None
    print("[WARN] sacrebleu not installed. Run: pip install sacrebleu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Caption preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def load_captions(captions_file: str) -> dict:
    """
    Parse Flickr8k captions file (format: image_id#N\\tcaption).
    Returns dict {image_id: [caption1, caption2, ...]}
    """
    captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle both tab-separated and first-token format
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            img_id, caption = parts
            img_id = img_id.split('#')[0]  # strip #N suffix
            caption = clean_caption(caption)
            captions.setdefault(img_id, []).append(caption)
    return captions


def clean_caption(caption: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return f"startseq {caption} endseq"


def build_tokenizer(captions: dict, min_freq: int = 5):
    """
    Build word ↔ index mappings.
    Returns (word2idx, idx2word, vocab_size, max_length).
    """
    counter = Counter()
    all_captions = [cap for caps in captions.values() for cap in caps]
    for cap in all_captions:
        counter.update(cap.split())

    vocab = sorted(
        [w for w, c in counter.items() if c >= min_freq])
    word2idx = {w: i + 1 for i, w in enumerate(vocab)}   # 0 = pad
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx) + 1
    max_length = max(len(c.split()) for c in all_captions)
    print(f"  Vocabulary size : {vocab_size}")
    print(f"  Max caption length : {max_length}")
    return word2idx, idx2word, vocab_size, max_length


# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_extractor() -> Model:
    """InceptionV3 without the classification head (global avg pool output)."""
    base = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base.input, outputs=base.output)


def extract_features(images_dir: str,
                     extractor: Model,
                     img_size: int = 299) -> dict:
    """
    Extract InceptionV3 features for all images in images_dir.
    Returns dict {filename: feature_vector(2048,)}
    """
    features = {}
    paths = list(Path(images_dir).glob("*.jpg"))
    n = len(paths)
    for i, path in enumerate(paths, 1):
        img = load_img(str(path), target_size=(img_size, img_size))
        x = img_to_array(img)
        x = np.expand_dims(x, 0)
        x = preprocess_input(x)
        feat = extractor.predict(x, verbose=0)
        features[path.name] = feat[0]
        if i % 200 == 0 or i == n:
            print(f"  Extracted {i}/{n}", end="\r")
    print()
    return features


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model definition
# ══════════════════════════════════════════════════════════════════════════════

def build_model(vocab_size: int,
                max_len: int,
                embed_dim: int = 256,
                units: int = 512,
                dropout_rate: float = 0.4) -> Model:
    """
    Encoder-decoder:
      - Image branch : Dense(embed_dim)
      - Text branch  : Embedding → Dropout → LSTM(units)
      - Merge        : element-wise Add
      - Output       : Dense(vocab_size, softmax)
    """
    # Image branch
    img_in = Input(shape=(2048,), name='image_input')
    img_fe = Dense(embed_dim, activation='relu')(img_in)

    # Text branch
    txt_in = Input(shape=(max_len,), name='text_input')
    txt_emb = Embedding(vocab_size, embed_dim, mask_zero=True)(txt_in)
    txt_drp = Dropout(dropout_rate)(txt_emb)
    txt_seq = LSTM(units)(txt_drp)

    # Merge and output
    merged = Add()([img_fe, txt_seq])
    out = Dense(vocab_size, activation='softmax')(merged)

    model = Model(inputs=[img_in, txt_in], outputs=out)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-3))
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Data generator
# ══════════════════════════════════════════════════════════════════════════════

def data_generator(captions: dict,
                   features: dict,
                   word2idx: dict,
                   max_len: int,
                   vocab_size: int,
                   batch_size: int = 64):
    """Yields (X_img, X_txt, y) batches using teacher-forcing."""
    img_batch, txt_batch, y_batch = [], [], []
    while True:
        for img_id, caps in captions.items():
            feat = features.get(img_id)
            if feat is None:
                continue
            for cap in caps:
                seq = [word2idx.get(w, 0) for w in cap.split()]
                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=max_len)[0]
                    out_word = seq[i]
                    img_batch.append(feat)
                    txt_batch.append(in_seq)
                    y_batch.append(out_word)
                    if len(img_batch) == batch_size:
                        yield (
                            [np.array(img_batch),
                             np.array(txt_batch)],
                            np.array(y_batch)
                        )
                        img_batch, txt_batch, y_batch = [], [], []


# ══════════════════════════════════════════════════════════════════════════════
# 5. Inference (greedy / beam)
# ══════════════════════════════════════════════════════════════════════════════

def predict_caption_greedy(model: Model,
                           feature: np.ndarray,
                           word2idx: dict,
                           idx2word: dict,
                           max_len: int) -> str:
    """Greedy decoding: argmax at every step."""
    in_text = "startseq"
    for _ in range(max_len):
        seq = pad_sequences(
            [[word2idx.get(w, 0) for w in in_text.split()]],
            maxlen=max_len)
        probs = model.predict(
            [feature.reshape(1, -1), seq], verbose=0)[0]
        next_idx = np.argmax(probs)
        word = idx2word.get(next_idx, '')
        if not word or word == 'endseq':
            break
        in_text += ' ' + word
    # Remove start token
    return ' '.join(in_text.split()[1:])


def predict_caption_beam(model: Model,
                         feature: np.ndarray,
                         word2idx: dict,
                         idx2word: dict,
                         max_len: int,
                         beam_width: int = 5) -> str:
    """Beam search decoding."""
    start_idx = word2idx.get('startseq', 1)
    end_idx = word2idx.get('endseq', 2)
    # Each beam: (log_prob, token_list)
    beams = [(0.0, [start_idx])]
    completed = []

    for _ in range(max_len):
        all_candidates = []
        for score, seq in beams:
            if seq[-1] == end_idx:
                completed.append((score, seq))
                continue
            padded = pad_sequences([seq], maxlen=max_len)
            probs = model.predict(
                [feature.reshape(1, -1), padded], verbose=0)[0]
            top_k = np.argsort(probs)[-beam_width:]
            for idx in top_k:
                new_score = score + math.log(probs[idx] + 1e-10)
                all_candidates.append((new_score, seq + [idx]))
        # Keep top beam_width
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]
        if not beams:
            break

    completed += beams
    best = max(completed, key=lambda x: x[0])
    words = [idx2word.get(i, '') for i in best[1][1:]]
    caption = ' '.join(w for w in words if w and w != 'endseq')
    return caption


# ══════════════════════════════════════════════════════════════════════════════
# 6. Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_bleu(model: Model,
                  captions: dict,
                  features: dict,
                  word2idx: dict,
                  idx2word: dict,
                  max_len: int,
                  test_ids: list) -> float:
    """
    Compute corpus BLEU-4 using sacrebleu (standard).
    test_ids: list of image filenames to evaluate on.
    """
    hypotheses, references = [], []
    for img_id in test_ids:
        feat = features.get(img_id)
        if feat is None:
            continue
        pred = predict_caption_greedy(
            model, feat, word2idx, idx2word, max_len)
        refs = [cap.replace('startseq ', '').replace(' endseq', '')
                for cap in captions.get(img_id, [])]
        hypotheses.append(pred)
        references.append(refs)

    if sacrebleu is None:
        print("[WARN] sacrebleu not available; skipping BLEU.")
        return -1.0

    # sacrebleu expects list-of-lists for references
    refs_transposed = list(zip(*references))  # shape: (n_refs, n_samples)
    bleu = sacrebleu.corpus_bleu(
        hypotheses, [list(r) for r in refs_transposed])
    return bleu.score / 100.0   # normalise to [0, 1]


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="CNN-LSTM Image Captioning")
    p.add_argument('--mode', required=True,
                   choices=['extract', 'train', 'evaluate', 'caption'])
    # Data
    p.add_argument('--images_dir', default='data/Images')
    p.add_argument('--captions',   default='data/captions.txt')
    p.add_argument('--out_dir',    default='data/')
    # Model / tokenizer paths
    p.add_argument('--features',   default='data/features.pkl')
    p.add_argument('--save_model', default='models/cnn_lstm.h5')
    p.add_argument('--save_tokenizer', default='models/tokenizer.pkl')
    p.add_argument('--model',      default='models/cnn_lstm.h5')
    p.add_argument('--tokenizer',  default='models/tokenizer.pkl')
    # Inference
    p.add_argument('--image_path', default=None)
    # Hyper-params
    p.add_argument('--epochs',      type=int, default=20)
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--embed_dim',   type=int, default=256)
    p.add_argument('--lstm_units',  type=int, default=512)
    p.add_argument('--min_freq',    type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'extract':
        print("── Feature extraction ──────────────────────")
        extractor = build_feature_extractor()
        feats = extract_features(args.images_dir, extractor)
        out = os.path.join(args.out_dir, 'features.pkl')
        with open(out, 'wb') as f:
            pickle.dump(feats, f)
        print(f"  Saved features to {out}")

    elif args.mode == 'train':
        print("── Training ────────────────────────────────")
        captions = load_captions(args.captions)
        with open(args.features, 'rb') as f:
            features = pickle.load(f)

        w2i, i2w, vocab_size, max_len = build_tokenizer(
            captions, args.min_freq)
        tokenizer_data = {
            'word2idx': w2i, 'idx2word': i2w,
            'vocab_size': vocab_size, 'max_len': max_len}
        os.makedirs(os.path.dirname(args.save_tokenizer), exist_ok=True)
        with open(args.save_tokenizer, 'wb') as f:
            pickle.dump(tokenizer_data, f)

        model = build_model(vocab_size, max_len,
                            args.embed_dim, args.lstm_units)
        model.summary()

        # Steps per epoch
        n_samples = sum(
            len(caps) * (max(len(c.split()) for c in caps) - 1)
            for img_id, caps in captions.items()
            if img_id in features)
        steps = n_samples // args.batch_size

        gen = data_generator(captions, features, w2i,
                             max_len, vocab_size, args.batch_size)
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        callbacks = [
            ModelCheckpoint(args.save_model, save_best_only=True,
                            monitor='loss', verbose=1),
            ReduceLROnPlateau(monitor='loss', factor=0.5,
                             patience=3, verbose=1)
        ]
        model.fit(gen, epochs=args.epochs, steps_per_epoch=steps,
                  callbacks=callbacks)

    elif args.mode == 'evaluate':
        print("── Evaluation ──────────────────────────────")
        captions = load_captions(args.captions)
        with open(args.features, 'rb') as f:
            features = pickle.load(f)
        with open(args.tokenizer, 'rb') as f:
            tok = pickle.load(f)

        model = load_model(args.model)
        all_ids = list(features.keys())
        test_ids = all_ids[-1000:]   # last 1000 as test split

        bleu4 = evaluate_bleu(model, captions, features,
                              tok['word2idx'], tok['idx2word'],
                              tok['max_len'], test_ids)
        print(f"  BLEU-4 (sacrebleu) : {bleu4:.4f}")

    elif args.mode == 'caption':
        print("── Inference ───────────────────────────────")
        if args.image_path is None:
            raise ValueError("--image_path required for caption mode")

        with open(args.tokenizer, 'rb') as f:
            tok = pickle.load(f)
        model = load_model(args.model)

        extractor = build_feature_extractor()
        img = load_img(args.image_path, target_size=(299, 299))
        x = preprocess_input(
            np.expand_dims(img_to_array(img), 0))
        feat = extractor.predict(x, verbose=0)[0]

        caption = predict_caption_beam(
            model, feat, tok['word2idx'], tok['idx2word'],
            tok['max_len'], beam_width=5)
        print(f"  Caption : {caption}")


if __name__ == '__main__':
    main()
