"""
Dataset preprocessing utilities for the NMT project.
"""

import re
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SEED = 4

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def normalize_text(text):
    text = text.strip().lower()

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text):
    return text.split()


def load_parallel_corpus(path):
    pairs = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            parts = line.rstrip().split("\t")

            if len(parts) < 2:
                continue

            src = normalize_text(parts[0])
            trg = normalize_text(parts[1])

            if src and trg:
                pairs.append(
                    (src, trg)
                )

    return pairs


def filter_pairs(
    pairs,
    max_seq_len
):
    filtered = []

    for src, trg in pairs:

        src_len = len(
            tokenize(src)
        ) + 2

        trg_len = len(
            tokenize(trg)
        ) + 2

        if (
            src_len <= max_seq_len
            and
            trg_len <= max_seq_len
        ):
            filtered.append(
                (src, trg)
            )

    return filtered


def deterministic_sample(
    pairs,
    max_pairs,
    seed=SEED
):
    if (
        max_pairs is None
        or
        max_pairs >= len(pairs)
    ):
        return pairs

    rng = np.random.RandomState(seed)

    indices = rng.choice(
        len(pairs),
        size=max_pairs,
        replace=False
    )

    indices.sort()

    return [
        pairs[i]
        for i in indices
    ]


def split_dataset(
    pairs,
    seed=SEED,
    train_ratio=0.8,
    val_ratio=0.1
):
    rng = np.random.RandomState(seed)

    indices = np.arange(
        len(pairs)
    )

    rng.shuffle(indices)

    n = len(indices)

    train_end = int(
        train_ratio * n
    )

    val_end = (
        train_end
        +
        int(val_ratio * n)
    )

    train = [
        pairs[i]
        for i in indices[:train_end]
    ]

    val = [
        pairs[i]
        for i in indices[
            train_end:val_end
        ]
    ]

    test = [
        pairs[i]
        for i in indices[val_end:]
    ]

    return train, val, test


class Vocabulary:

    def __init__(
        self,
        max_size=20000,
        min_freq=2
    ):
        self.max_size = max_size
        self.min_freq = min_freq

        self.token_to_idx = {}
        self.idx_to_token = {}

        for token in SPECIAL_TOKENS:
            self._add_token(token)


    def _add_token(self, token):
        if token not in self.token_to_idx:
            idx = len(
                self.token_to_idx
            )

            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token


    def build(self, sentences):
        counter = Counter()

        for sentence in sentences:
            counter.update(
                tokenize(sentence)
            )

        tokens = [
            token
            for token, count in counter.items()
            if count >= self.min_freq
        ]

        tokens.sort(
            key=lambda token: (
                -counter[token],
                token
            )
        )

        if self.max_size is not None:
            available = (
                self.max_size
                -
                len(SPECIAL_TOKENS)
            )

            tokens = tokens[
                :available
            ]

        for token in tokens:
            self._add_token(token)


    def encode(
        self,
        sentence,
        max_len
    ):
        ids = [
            self.token_to_idx[
                SOS_TOKEN
            ]
        ]

        unk = self.token_to_idx[
            UNK_TOKEN
        ]

        for token in tokenize(
            sentence
        ):
            ids.append(
                self.token_to_idx.get(
                    token,
                    unk
                )
            )

        ids.append(
            self.token_to_idx[
                EOS_TOKEN
            ]
        )

        ids = ids[:max_len]

        if ids[-1] != self.token_to_idx[
            EOS_TOKEN
        ]:
            ids[-1] = self.token_to_idx[
                EOS_TOKEN
            ]

        ids += [
            self.token_to_idx[
                PAD_TOKEN
            ]
        ] * (
            max_len - len(ids)
        )

        return ids


    def decode(
        self,
        ids
    ):
        tokens = []

        for idx in ids:

            token = self.idx_to_token[
                int(idx)
            ]

            if token == EOS_TOKEN:
                break

            if token in SPECIAL_TOKENS:
                continue

            tokens.append(token)

        return tokens


    def __len__(self):
        return len(
            self.token_to_idx
        )


class TranslationDataset(Dataset):

    def __init__(
        self,
        pairs,
        src_vocab,
        trg_vocab,
        max_seq_len
    ):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_seq_len = max_seq_len


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, index):
        src_text, trg_text = (
            self.pairs[index]
        )

        src = self.src_vocab.encode(
            src_text,
            self.max_seq_len
        )

        trg = self.trg_vocab.encode(
            trg_text,
            self.max_seq_len
        )

        return {
            "src": torch.tensor(
                src,
                dtype=torch.long
            ),
            "trg": torch.tensor(
                trg,
                dtype=torch.long
            ),
            "src_text": src_text,
            "trg_text": trg_text,
        }
