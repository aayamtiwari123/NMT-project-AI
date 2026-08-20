"""
Evaluate trained NMT models.

Metrics:
    - BLEU-4
    - Perplexity
    - Mean latency
    - Median latency
    - P95 latency

Usage:
    python src/evaluate.py --language fra
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.translate.bleu_score import (
    corpus_bleu,
    sentence_bleu,
    SmoothingFunction,
)

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from preprocess import (
    SEED,
    PAD_TOKEN,
    EOS_TOKEN,
    SOS_TOKEN,
    seed_everything,
    load_parallel_corpus,
    filter_pairs,
    deterministic_sample,
    split_dataset,
    Vocabulary,
    TranslationDataset,
    tokenize,
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


EMBEDDING_DIM = 256
HIDDEN_DIM = 256
MAX_SEQ_LEN = 30
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
MAX_PAIRS = 50000
BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Model definitions
# ============================================================

class VanillaRNNSeq2Seq(nn.Module):

    def __init__(
        self,
        src_dim,
        trg_dim,
        src_pad,
        trg_pad
    ):
        super().__init__()

        self.encoder = VanillaRNNEncoder(
            src_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            src_pad
        )

        self.decoder = VanillaRNNDecoder(
            trg_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            trg_pad
        )


class GRUSeq2Seq(nn.Module):

    def __init__(
        self,
        src_dim,
        trg_dim,
        src_pad,
        trg_pad
    ):
        super().__init__()

        self.encoder = GRUEncoder(
            src_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            src_pad
        )

        self.decoder = GRUDecoder(
            trg_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            trg_pad
        )


class AttentionSeq2Seq(nn.Module):

    def __init__(
        self,
        src_dim,
        trg_dim,
        src_pad,
        trg_pad,
        attention_type
    ):
        super().__init__()

        self.encoder = GRUEncoder(
            src_dim,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            src_pad
        )

        if attention_type == "bahdanau":
            self.decoder = BahdanauDecoder(
                trg_dim,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                trg_pad
            )

        else:
            self.decoder = LuongDecoder(
                trg_dim,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                trg_pad
            )


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


@torch.no_grad()
def translate(
    model,
    sentence,
    src_vocab,
    trg_vocab
):
    model.eval()

    src_ids = src_vocab.encode(
        sentence,
        MAX_SEQ_LEN
    )

    src = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    sos = trg_vocab.token_to_idx[
        SOS_TOKEN
    ]

    eos = trg_vocab.token_to_idx[
        EOS_TOKEN
    ]

    src_pad = src_vocab.token_to_idx[
        PAD_TOKEN
    ]

    input_token = torch.tensor(
        [sos],
        device=DEVICE
    )

    attentions = []

    if isinstance(
        model,
        VanillaRNNSeq2Seq
    ):

        _, hidden = model.encoder(src)

        generated = []

        for _ in range(
            MAX_SEQ_LEN - 1
        ):
            output, hidden = (
                model.decoder(
                    input_token,
                    hidden
                )
            )

            next_token = output.argmax(
                dim=1
            ).item()

            if next_token == eos:
                break

            generated.append(
                next_token
            )

            input_token = torch.tensor(
                [next_token],
                device=DEVICE
            )

        return (
            trg_vocab.decode(generated),
            None
        )


    if isinstance(
        model,
        GRUSeq2Seq
    ):

        _, hidden = model.encoder(src)

        generated = []

        for _ in range(
            MAX_SEQ_LEN - 1
        ):
            output, hidden = (
                model.decoder(
                    input_token,
                    hidden
                )
            )

            next_token = output.argmax(
                dim=1
            ).item()

            if next_token == eos:
                break

            generated.append(
                next_token
            )

            input_token = torch.tensor(
                [next_token],
                device=DEVICE
            )

        return (
            trg_vocab.decode(generated),
            None
        )


    encoder_outputs, hidden = (
        model.encoder(src)
    )

    src_mask = (
        src != src_pad
    )

    generated = []

    for _ in range(
        MAX_SEQ_LEN - 1
    ):

        output, hidden, attention = (
            model.decoder(
                input_token,
                hidden,
                encoder_outputs,
                src_mask
            )
        )

        next_token = output.argmax(
            dim=1
        ).item()

        attentions.append(
            attention.squeeze(
                0
            ).cpu().numpy()
        )

        if next_token == eos:
            break

        generated.append(
            next_token
        )

        input_token = torch.tensor(
            [next_token],
            device=DEVICE
        )

    attention_matrix = (
        np.array(attentions)
        if attentions
        else None
    )

    return (
        trg_vocab.decode(generated),
        attention_matrix
    )


def evaluate_test_loss(
    model,
    loader,
    trg_pad
):
    model.eval()

    criterion = nn.CrossEntropyLoss(
        ignore_index=trg_pad
    )

    total_loss = 0.0

    for batch in loader:

        src = batch["src"].to(
            DEVICE
        )

        trg = batch["trg"].to(
            DEVICE
        )

        # Teacher-forced evaluation.
        if isinstance(
            model,
            VanillaRNNSeq2Seq
        ):
            _, hidden = model.encoder(
                src
            )

            outputs = torch.zeros(
                src.size(0),
                trg.size(1),
                len(trg_pad if False else []),
                device=DEVICE
            )

            # Not used for final evaluation.
            # Perplexity is evaluated through the
            # autoregressive model below.
            continue

    return None


def perplexity_from_predictions(
    model,
    pairs,
    src_vocab,
    trg_vocab
):
    """
    Calculate autoregressive token-level NLL.

    This avoids relying on the training script's model wrapper
    and makes evaluation independent.
    """

    criterion = nn.CrossEntropyLoss(
        ignore_index=trg_vocab.token_to_idx[
            PAD_TOKEN
        ],
        reduction="sum"
    )

    total_loss = 0.0
    total_tokens = 0

    for src_text, trg_text in pairs:

        src_ids = src_vocab.encode(
            src_text,
            MAX_SEQ_LEN
        )

        trg_ids = trg_vocab.encode(
            trg_text,
            MAX_SEQ_LEN
        )

        src = torch.tensor(
            src_ids,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        trg = torch.tensor(
            trg_ids,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        model.eval()

        if isinstance(
            model,
            VanillaRNNSeq2Seq
        ):

            _, hidden = model.encoder(
                src
            )

            input_token = trg[:, 0]

            for t in range(
                1,
                trg.size(1)
            ):
                output, hidden = (
                    model.decoder(
                        input_token,
                        hidden
                    )
                )

                target = trg[:, t]

                total_loss += (
                    criterion(
                        output,
                        target
                    ).item()
                )

                if target.item() == trg_vocab.token_to_idx[
                    PAD_TOKEN
                ]:
                    continue

                total_tokens += 1
                input_token = target


        elif isinstance(
            model,
            GRUSeq2Seq
        ):

            _, hidden = model.encoder(
                src
            )

            input_token = trg[:, 0]

            for t in range(
                1,
                trg.size(1)
            ):
                output, hidden = (
                    model.decoder(
                        input_token,
                        hidden
                    )
                )

                target = trg[:, t]

                total_loss += (
                    criterion(
                        output,
                        target
                    ).item()
                )

                if target.item() != trg_vocab.token_to_idx[
                    PAD_TOKEN
                ]:
                    total_tokens += 1

                input_token = target


        else:

            encoder_outputs, hidden = (
                model.encoder(src)
            )

            src_mask = (
                src
                !=
                src_vocab.token_to_idx[
                    PAD_TOKEN
                ]
            )

            input_token = trg[:, 0]

            for t in range(
                1,
                trg.size(1)
            ):
                output, hidden, _ = (
                    model.decoder(
                        input_token,
                        hidden,
                        encoder_outputs,
                        src_mask
                    )
                )

                target = trg[:, t]

                total_loss += (
                    criterion(
                        output,
                        target
                    ).item()
                )

                if target.item() != trg_vocab.token_to_idx[
                    PAD_TOKEN
                ]:
                    total_tokens += 1

                input_token = target

    if total_tokens == 0:
        return float("inf")

    return math.exp(
        total_loss / total_tokens
    )


def calculate_bleu(
    model,
    test_pairs,
    src_vocab,
    trg_vocab
):

    references = []
    hypotheses = []

    smoothing = (
        SmoothingFunction().method4
    )

    sentence_scores = []

    for src_text, trg_text in test_pairs:

        prediction, _ = translate(
            model,
            src_text,
            src_vocab,
            trg_vocab
        )

        reference = tokenize(
            trg_text
        )

        references.append(
            [reference]
        )

        hypotheses.append(
            prediction
        )

        if prediction:

            score = sentence_bleu(
                [reference],
                prediction,
                weights=(
                    0.25,
                    0.25,
                    0.25,
                    0.25
                ),
                smoothing_function=smoothing
            )

            sentence_scores.append(
                score
            )

    bleu = corpus_bleu(
        references,
        hypotheses,
        weights=(
            0.25,
            0.25,
            0.25,
            0.25
        )
    )

    return {
        "bleu": bleu * 100,
        "sentence_bleu": (
            np.mean(
                sentence_scores
            ) * 100
            if sentence_scores
            else 0
        )
    }


def measure_latency(
    model,
    test_pairs,
    src_vocab,
    trg_vocab
):

    sentences = [
        x[0]
        for x in test_pairs[:100]
    ]

    # Warm-up.
    for sentence in sentences[:10]:
        translate(
            model,
            sentence,
            src_vocab,
            trg_vocab
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []

    for sentence in sentences:

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        translate(
            model,
            sentence,
            src_vocab,
            trg_vocab
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        times.append(
            (
                time.perf_counter()
                -
                start
            ) * 1000
        )

    return {
        "mean_ms": float(
            np.mean(times)
        ),
        "median_ms": float(
            np.median(times)
        ),
        "p95_ms": float(
            np.percentile(
                times,
                95
            )
        )
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        default="fra",
        choices=["fra", "spa"]
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
            "Dataset not found. Run "
            "data/download_data.py first."
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

    train_pairs, _, test_pairs = (
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

    checkpoint_dir = (
        ROOT
        /
        "results"
        /
        "checkpoints"
    )

    models = [
        "Vanilla RNN",
        "GRU Seq2Seq",
        "Bahdanau Attention",
        "Luong Attention",
    ]

    rows = []

    for name in models:

        print(
            f"\nEvaluating {name}..."
        )

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pt"
        )

        checkpoint_path = (
            checkpoint_dir
            /
            filename
        )

        if not checkpoint_path.exists():
            print(
                f"Skipping {name}: "
                f"checkpoint not found."
            )
            continue

        src_pad = src_vocab.token_to_idx[
            PAD_TOKEN
        ]

        trg_pad = trg_vocab.token_to_idx[
            PAD_TOKEN
        ]

        model = build_model(
            name,
            len(src_vocab),
            len(trg_vocab),
            src_pad,
            trg_pad
        ).to(DEVICE)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        bleu = calculate_bleu(
            model,
            test_pairs,
            src_vocab,
            trg_vocab
        )

        latency = measure_latency(
            model,
            test_pairs,
            src_vocab,
            trg_vocab
        )

        parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        row = {
            "Model": name,
            "Parameters": parameters,
            "BLEU": bleu["bleu"],
            "Sentence_BLEU":
                bleu["sentence_bleu"],
            "Mean_Latency_ms":
                latency["mean_ms"],
            "Median_Latency_ms":
                latency["median_ms"],
            "P95_Latency_ms":
                latency["p95_ms"],
        }

        rows.append(row)

        print(
            f"BLEU: "
            f"{row['BLEU']:.3f}"
        )

        print(
            f"Mean latency: "
            f"{row['Mean_Latency_ms']:.3f} ms"
        )

    if not rows:
        raise RuntimeError(
            "No trained checkpoints found."
        )

    results = pd.DataFrame(rows)

    output_dir = (
        ROOT
        /
        "results"
    )

    results.to_csv(
        output_dir
        /
        "evaluation_results.csv",
        index=False
    )

    with open(
        output_dir
        /
        "evaluation_results.json",
        "w"
    ) as f:

        json.dump(
            rows,
            f,
            indent=2
        )

    print("\nFinal results:")
    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
