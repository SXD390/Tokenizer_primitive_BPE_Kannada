#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a simple BPE tokenizer on Kannada text and report:
- final vocab size
- compression ratio (chars / tokens) on test split

Usage (basic):

    python3 train_kannada_bpe.py \
        --data_path data/kannada_corpus.txt \
        --output_dir artifacts \
        --vocab_size 5200

You can change paths and vocab_size as needed.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple


class BPETokenizer:
    def __init__(self):
        self.vocab = {}          # token -> id
        self.id2token = {}       # id -> token
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks = {}      # (token1, token2) -> rank
        self._fitted = False

    # ---------- basic helpers ----------

    @staticmethod
    def _whitespace_tokenize(text: str) -> List[str]:
        # Simple: split on whitespace. This works fine for novels.
        return text.strip().split()

    @staticmethod
    def _word_to_chars(word: str) -> Tuple[str, ...]:
        # Represent each word as characters + end-of-word marker
        return tuple(list(word) + ["</w>"])

    # ---------- BPE core ----------

    def _get_stats(self, corpus_tokens: List[Tuple[str, ...]]) -> Counter:
        """
        Count frequency of adjacent token pairs across the corpus.
        """
        pairs = Counter()
        for word in corpus_tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1
        return pairs

    def _merge_corpus(
        self,
        corpus_tokens: List[Tuple[str, ...]],
        pair: Tuple[str, str],
    ) -> List[Tuple[str, ...]]:
        """
        Replace all occurrences of `pair` with a merged token.
        """
        bigram = pair
        merged_symbol = "".join(bigram)

        new_corpus = []
        for word in corpus_tokens:
            i = 0
            new_word = []
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == bigram[0]
                    and word[i + 1] == bigram[1]
                ):
                    new_word.append(merged_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus.append(tuple(new_word))
        return new_corpus

    def train(
        self,
        texts: List[str],
        vocab_size: int = 5200,
        min_pair_freq: int = 2,
        verbose: bool = True,
    ):
        """
        Train BPE tokenizer on given texts.

        vocab_size: total target vocab size (including chars + merges + special tokens)
        min_pair_freq: stop if best pair appears less than this.
        """
        # 1. Tokenize into words
        words = []
        for t in texts:
            words.extend(self._whitespace_tokenize(t))

        if verbose:
            print(f"[train] Number of words in training data: {len(words)}")

        # 2. Represent each word as tuple of chars + </w>
        corpus_tokens = [self._word_to_chars(w) for w in words]

        # 3. Build initial vocab of characters
        char_counts = Counter(ch for word in corpus_tokens for ch in word)
        vocab = set(char_counts.keys())

        if verbose:
            print(f"[train] Initial vocab size (chars + </w>): {len(vocab)}")

        merges: List[Tuple[str, str]] = []

        # 4. Iteratively merge most frequent pairs
        # We stop when vocab reaches vocab_size or no frequent pairs left.
        while len(vocab) < vocab_size:
            stats = self._get_stats(corpus_tokens)
            if not stats:
                if verbose:
                    print("[train] No more pairs to merge, stopping.")
                break

            best_pair, best_freq = stats.most_common(1)[0]

            if best_freq < min_pair_freq:
                if verbose:
                    print(
                        f"[train] Stopping: best pair freq {best_freq} < min_pair_freq {min_pair_freq}"
                    )
                break

            # Merge this pair
            corpus_tokens = self._merge_corpus(corpus_tokens, best_pair)
            merged_symbol = "".join(best_pair)
            vocab.add(merged_symbol)
            merges.append(best_pair)

            if verbose and len(merges) % 100 == 0:
                print(
                    f"[train] Merges: {len(merges)}, current vocab size (without special tokens): {len(vocab)}"
                )

            if len(vocab) >= vocab_size:
                break

        self.merges = merges
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # 5. Build final vocab mapping, including special tokens
        special_tokens = ["<pad>", "<unk>"]
        all_tokens = special_tokens + sorted(vocab)

        self.vocab = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.id2token = {idx: tok for tok, idx in self.vocab.items()}
        self._fitted = True

        if verbose:
            print(
                f"[train] Training complete. Final vocab size (including special tokens): {len(self.vocab)}"
            )

    # ---------- apply BPE to a word ----------

    def _encode_word(self, word: str) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Tokenizer not trained or loaded.")

        # Start with characters + </w>
        word_tokens = list(word) + ["</w>"]
        word_tokens = tuple(word_tokens)

        # Greedy BPE merges
        while True:
            if len(word_tokens) < 2:
                break

            pairs = [(word_tokens[i], word_tokens[i + 1]) for i in range(len(word_tokens) - 1)]

            # For each pair, find its rank (lower == earlier merge == higher priority)
            pair_ranks = [
                (self.bpe_ranks.get(p, float("inf")), p) for p in pairs
            ]
            best_rank, best_pair = min(pair_ranks, key=lambda x: x[0])

            if best_rank == float("inf"):
                # No more applicable pairs
                break

            # Merge all occurrences of best_pair
            new_word = []
            i = 0
            while i < len(word_tokens):
                if (
                    i < len(word_tokens) - 1
                    and word_tokens[i] == best_pair[0]
                    and word_tokens[i + 1] == best_pair[1]
                ):
                    new_word.append("".join(best_pair))
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            word_tokens = tuple(new_word)

        # Drop </w> marker at the end
        if word_tokens and word_tokens[-1] == "</w>":
            word_tokens = word_tokens[:-1]

        return list(word_tokens)

    # ---------- public encode/decode ----------

    def encode(self, text: str) -> List[int]:
        """
        Convert text -> list of token ids.
        """
        if not self._fitted:
            raise RuntimeError("Tokenizer not trained or loaded.")

        tokens: List[str] = []
        for word in self._whitespace_tokenize(text):
            tokens.extend(self._encode_word(word))

        unk_id = self.vocab.get("<unk>")
        token_ids = [self.vocab.get(tok, unk_id) for tok in tokens]
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Very simple decode (approximate). For compression ratio we don't really need decode,
        but this is useful for debugging / later demo.
        """
        if not self._fitted:
            raise RuntimeError("Tokenizer not trained or loaded.")

        tokens = [self.id2token.get(i, "<unk>") for i in token_ids]
        # Undo special tokens and </w> best-effort
        text = "".join(tok for tok in tokens if tok not in ["<pad>", "<unk>"])
        text = text.replace("</w>", " ")
        return text.strip()

    # ---------- save/load ----------

    def save(self, vocab_path: str, merges_path: str):
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False, indent=2)

    def load(self, vocab_path: str, merges_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(merges_path, "r", encoding="utf-8") as f:
            merges_list = json.load(f)
            # stored as lists; convert back to tuples
            self.merges = [tuple(p) for p in merges_list]

        self.id2token = {idx: tok for tok, idx in self.vocab.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._fitted = True


# ---------- training / evaluation script ----------

def compute_compression_ratio(tokenizer: BPETokenizer, texts: List[str]):
    total_chars = 0
    total_tokens = 0
    for t in texts:
        total_chars += len(t)
        ids = tokenizer.encode(t)
        total_tokens += len(ids)

    if total_tokens == 0:
        return 0.0, total_chars, total_tokens

    ratio = total_chars / total_tokens
    return ratio, total_chars, total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer on Kannada text and compute compression ratio."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to raw Kannada text file (UTF-8).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="Directory to save vocab.json and merges.json",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=5200,
        help="Target vocab size (including chars + merges + special tokens). Must be > 5000 for assignment.",
    )
    parser.add_argument(
        "--min_pair_freq",
        type=int,
        default=2,
        help="Minimum frequency of a pair to be merged. Higher value = faster but maybe lower compression.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction of lines used for test set (0-1). Default 0.1 (10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] Loading corpus from: {data_path}")
    text = data_path.read_text(encoding="utf-8")

    # Split into non-empty lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    print(f"[main] Total non-empty lines: {len(lines)}")

    if len(lines) < 10:
        print("[main] WARNING: Very few lines detected. Compression ratio might be unreliable.")

    # Train/test split
    random.seed(args.seed)
    random.shuffle(lines)
    split_idx = int((1.0 - args.test_split) * len(lines))
    train_texts = lines[:split_idx]
    test_texts = lines[split_idx:]

    print(f"[main] Train lines: {len(train_texts)}")
    print(f"[main] Test  lines: {len(test_texts)}")

    # Train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(
        texts=train_texts,
        vocab_size=args.vocab_size,
        min_pair_freq=args.min_pair_freq,
        verbose=True,
    )

    final_vocab_size = len(tokenizer.vocab)
    print(f"[main] Final vocab size (including special tokens): {final_vocab_size}")

    # Compute compression ratio on test set
    ratio, total_chars, total_tokens = compute_compression_ratio(tokenizer, test_texts)
    print("------------------------------------------------------")
    print(f"[eval] Total characters in test set: {total_chars}")
    print(f"[eval] Total tokens in test set:     {total_tokens}")
    print(f"[eval] Compression ratio (chars/tokens): {ratio:.4f}")
    print("------------------------------------------------------")

    if final_vocab_size <= 5000:
        print(
            "[eval] WARNING: Final vocab size <= 5000. "
            "Increase --vocab_size to satisfy assignment constraint."
        )

    if ratio < 3.2:
        print(
            "[eval] WARNING: Compression ratio < 3.2. "
            "Try increasing vocab_size or using more training data."
        )
    else:
        print("[eval] ✅ Compression ratio >= 3.2 requirement satisfied (for this test split).")

    # Save tokenizer
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    tokenizer.save(str(vocab_path), str(merges_path))

    print(f"[main] Saved vocab to:  {vocab_path}")
    print(f"[main] Saved merges to: {merges_path}")


if __name__ == "__main__":
    main()
