"""
Train the four NMT architectures.

Usage:

    python src/train.py --language fra

or:

    python src/train.py --language spa --epochs 20
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from preprocess import (
    SEED,
    PAD_TOKEN,
    seed_everything,
    load_parallel_corpus,
    filter_pairs,
    deterministic_sample,
    split_dataset,
    Vocabulary,
    TranslationDataset,
)

from models.rnn import (
    VanillaRNNEncoder,
    VanillaRNNDecoder,
)

from models.seq2seq import (
    GRUEncoder,
    GRUDecoder,
)

from models.additive_attn import (
    BahdanauDecoder,
)

from models.multiplicative_attn import (
    LuongDecoder,
)


# ============================================================
# Defaults
# ============================================================

EMBEDDING_DIM = 256
HIDDEN_DIM = 256
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 30
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
MAX_PAIRS = 50000

NUM_EPOCHS = 12

TEACHER_START = 1.0
TEACHER_END = 0.5

CLIP_VALUE = 1.0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Models
# ============================================================

class VanillaRNNSeq2Seq(nn.Module):

    def __init__(
        self,
        src_dim,
        trg_dim,
        src_pad_idx,
        trg_pad_idx
    ):
        super().__init__()

        self.encoder = VanillaRNNEncoder(
            src_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            src_pad_idx
        )

        self.decoder = VanillaRNNDecoder(
            trg_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            trg_pad_idx
        )


    def forward(
        self,
        src,
        trg,
        teacher_forcing_ratio
    ):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(
            batch_size,
            trg_len,
            output_dim,
            device=src.device
        )

        _, hidden = self.encoder(src)

        input_token = trg[:, 0]

        for t in range(
            1,
            trg_len
        ):
            output, hidden = (
                self.decoder(
                    input_token,
                    hidden
                )
            )

            outputs[:, t] = output

            use_teacher = (
                random.random()
                <
                teacher_forcing_ratio
            )

            top1 = output.argmax(
                dim=1
            )

            input_token = (
                trg[:, t]
                if use_teacher
                else top1
            )

        return outputs, None


class GRUSeq2Seq(nn.Module):

    def __init__(
        self,
        src_dim,
        trg_dim,
        src_pad_idx,
        trg_pad_idx
    ):
        super().__init__()

        self.encoder = GRUEncoder(
            src_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            src_pad_idx
        )

        self.decoder = GRUDecoder(
            trg_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            trg_pad_idx
        )


    def forward(
        self,
        src,
        trg,
        teacher_forcing_ratio
    ):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(
            batch_size,
            trg_len,
            output_dim,
            device=src.device
        )

        _, hidden = self.encoder(src)

        input_token = trg[:, 0]

        for t in range(
            1,
            trg_len
        ):
            output, hidden = (
                self.decoder(
                    input_token,
                    hidden
                )
            )

            outputs[:, t] = output

            use_teacher = (
                random.random()
                <
                teacher_forcing_ratio
            )

            top1 = output.argmax(
                dim=1
            )

            input_token = (
                trg[:, t]
                if use_teacher
                else top1
            )

        return outputs, None


class AttentionSeq2Seq(nn.Module):

    def __init__(
        self,
        src_dim,
        trg_dim,
        src_pad_idx,
        trg_pad_idx,
        attention_type
    ):
        super().__init__()

        self.encoder = GRUEncoder(
            src_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            src_pad_idx
        )

        if attention_type == "bahdanau":
            self.decoder = BahdanauDecoder(
                trg_dim,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                trg_pad_idx
            )

        elif attention_type == "luong":
            self.decoder = LuongDecoder(
                trg_dim,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                trg_pad_idx
            )

        else:
            raise ValueError(
                attention_type
            )

        self.src_pad_idx = src_pad_idx


    def forward(
        self,
        src,
        trg,
        teacher_forcing_ratio
    ):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(
            batch_size,
            trg_len,
            output_dim,
            device=src.device
        )

        encoder_outputs, hidden = (
            self.encoder(src)
        )

        src_mask = (
            src != self.src_pad_idx
        )

        input_token = trg[:, 0]

        for t in range(
            1,
            trg_len
        ):
            output, hidden, _ = (
                self.decoder(
                    input_token,
                    hidden,
                    encoder_outputs,
                    src_mask
                )
            )

            outputs[:, t] = output

            use_teacher = (
                random.random()
                <
                teacher_forcing_ratio
            )

            top1 = output.argmax(
                dim=1
            )

            input_token = (
                trg[:, t]
                if use_teacher
                else top1
            )

        return outputs, None


# ============================================================
# Factory
# ============================================================

def build_model(
    name,
    src_dim,
    trg_dim,
    src_pad,
    trg_pad
):

    if name == "Vanilla RNN":
        return VanillaRNNSeq2Seq(
            src_dim,
            trg_dim,
            src_pad,
            trg_pad
        )

    if name == "GRU Seq2Seq":
        return GRUSeq2Seq(
            src_dim,
            trg_dim,
            src_pad,
            trg_pad
        )

    if name == "Bahdanau Attention":
        return AttentionSeq2Seq(
            src_dim,
            trg_dim,
            src_pad,
            trg_pad,
            "bahdanau"
        )

    if name == "Luong Attention":
        return AttentionSeq2Seq(
            src_dim,
            trg_dim,
            src_pad,
            trg_pad,
            "luong"
        )

    raise ValueError(name)


# ============================================================
# Training
# ============================================================

def loss_fn(
    outputs,
    targets,
    criterion
):
    output_dim = outputs.size(-1)

    outputs = outputs[:, 1:].reshape(
        -1,
        output_dim
    )

    targets = targets[:, 1:].reshape(
        -1
    )

    return criterion(
        outputs,
        targets
    )


def teacher_ratio(
    epoch,
    epochs
):
    if epochs <= 1:
        return TEACHER_END

    progress = (
        epoch - 1
    ) / (
        epochs - 1
    )

    return (
        TEACHER_START
        +
        progress
        *
        (
            TEACHER_END
            -
            TEACHER_START
        )
    )


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    ratio
):
    model.train()

    total_loss = 0.0

    for batch in loader:

        src = batch["src"].to(
            DEVICE
        )

        trg = batch["trg"].to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs, _ = model(
            src,
            trg,
            ratio
        )

        loss = loss_fn(
            outputs,
            trg,
            criterion
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CLIP_VALUE
        )

        optimizer.step()

        total_loss += (
            loss.item()
            *
            src.size(0)
        )

    return (
        total_loss
        /
        len(loader.dataset)
    )


@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    criterion
):
    model.eval()

    total_loss = 0.0

    for batch in loader:

        src = batch["src"].to(
            DEVICE
        )

        trg = batch["trg"].to(
            DEVICE
        )

        outputs, _ = model(
            src,
            trg,
            1.0
        )

        loss = loss_fn(
            outputs,
            trg,
            criterion
        )

        total_loss += (
            loss.item()
            *
            src.size(0)
        )

    return (
        total_loss
        /
        len(loader.dataset)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        default="fra",
        choices=["fra", "spa"]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=MAX_PAIRS
    )

    args = parser.parse_args()

    seed_everything(SEED)

    dataset_path = (
        ROOT
        /
        "data"
        /
        "raw"
        /
        f"{args.language}-eng.txt"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run:\n"
            f"python data/download_data.py "
            f"--language {args.language}"
        )

    pairs = load_parallel_corpus(
        dataset_path
    )

    pairs = filter_pairs(
        pairs,
        MAX_SEQ_LEN
    )

    pairs = deterministic_sample(
        pairs,
        args.max_pairs,
        SEED
    )

    train_pairs, val_pairs, test_pairs = (
        split_dataset(
            pairs,
            SEED
        )
    )

    src_vocab = Vocabulary(
        MAX_VOCAB_SIZE,
        MIN_FREQ
    )

    trg_vocab = Vocabulary(
        MAX_VOCAB_SIZE,
        MIN_FREQ
    )

    src_vocab.build(
        src
        for src, _ in train_pairs
    )

    trg_vocab.build(
        trg
        for _, trg in train_pairs
    )

    train_dataset = TranslationDataset(
        train_pairs,
        src_vocab,
        trg_vocab,
        MAX_SEQ_LEN
    )

    val_dataset = TranslationDataset(
        val_pairs,
        src_vocab,
        trg_vocab,
        MAX_SEQ_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    src_pad = src_vocab.token_to_idx[
        PAD_TOKEN
    ]

    trg_pad = trg_vocab.token_to_idx[
        PAD_TOKEN
    ]

    criterion = nn.CrossEntropyLoss(
        ignore_index=trg_pad
    )

    models = [
        "Vanilla RNN",
        "GRU Seq2Seq",
        "Bahdanau Attention",
        "Luong Attention",
    ]

    checkpoint_dir = (
        ROOT
        /
        "results"
        /
        "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    history_dir = (
        ROOT
        /
        "results"
        /
        "history"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for model_name in models:

        print("\n")
        print("=" * 60)
        print(model_name)
        print("=" * 60)

        seed_everything(SEED)

        model = build_model(
            model_name,
            len(src_vocab),
            len(trg_vocab),
            src_pad,
            trg_pad
        ).to(DEVICE)

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        best_val = float("inf")

        history = {
            "train_loss": [],
            "val_loss": [],
            "teacher_forcing": [],
        }

        for epoch in range(
            1,
            args.epochs + 1
        ):

            ratio = teacher_ratio(
                epoch,
                args.epochs
            )

            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                ratio
            )

            val_loss = evaluate_loss(
                model,
                val_loader,
                criterion
            )

            history[
                "train_loss"
            ].append(train_loss)

            history[
                "val_loss"
            ].append(val_loss)

            history[
                "teacher_forcing"
            ].append(ratio)

            print(
                f"Epoch {epoch:02d} | "
                f"TF={ratio:.3f} | "
                f"Train={train_loss:.4f} | "
                f"Val={val_loss:.4f}"
            )

            if val_loss < best_val:

                best_val = val_loss

                filename = (
                    model_name.lower()
                    .replace(" ", "_")
                    + ".pt"
                )

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),
                        "src_vocab":
                            src_vocab.token_to_idx,
                        "trg_vocab":
                            trg_vocab.token_to_idx,
                        "src_idx_to_token":
                            src_vocab.idx_to_token,
                        "trg_idx_to_token":
                            trg_vocab.idx_to_token,
                        "model_name":
                            model_name,
                        "best_val_loss":
                            best_val,
                    },
                    checkpoint_dir / filename
                )

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".json"
        )

        with open(
            history_dir / filename,
            "w"
        ) as f:
            json.dump(
                history,
                f,
                indent=2
            )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
