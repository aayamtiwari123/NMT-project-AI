# Comparative Analysis of Neural Machine Translation Architectures

## Abstract

This study presents a controlled comparative evaluation of four sequence-to-sequence neural machine translation architectures for English-to-French translation using the ManyThings.org Anki parallel corpus. The evaluated architectures are: (1) a Vanilla RNN encoder-decoder baseline, (2) a GRU encoder-decoder without attention, (3) a GRU encoder-decoder with Bahdanau additive attention, and (4) a GRU encoder-decoder with Luong multiplicative attention.

The experimental protocol uses a common dataset, deterministic 80/10/10 train-validation-test partitioning with seed 4, common embedding and hidden dimensions, identical batch size, optimizer, learning rate, loss function, sequence-length constraints, and a progressively reduced teacher-forcing schedule. Models are evaluated using BLEU-4, teacher-forced perplexity, and autoregressive inference latency. Attention alignment heatmaps are additionally analyzed for the two attention-based architectures.

The best BLEU score was obtained by **Bahdanau Attention**, with a BLEU-4 score of **24.564**. The lowest measured perplexity was obtained by **Bahdanau Attention**, while the lowest mean inference latency was observed for **Vanilla RNN**.

---

## Quick Overview & Key Results

| Criterion | Best Architecture | Value |
|---|---|---:|
| BLEU-4 | Bahdanau Attention | 24.564 |
| Perplexity | Bahdanau Attention | 6.712 |
| Mean Latency | Vanilla RNN | 3.385 ms |
| Fewest Parameters | Vanilla RNN | 7,486,902 |

---

## 1. Introduction

Neural Machine Translation (NMT) models learn to transform a sequence in a source language into a sequence in a target language. Early neural sequence-to-sequence systems commonly used an encoder-decoder architecture in which the encoder converted the source sequence into a hidden representation and the decoder generated the target sequence from that representation.

A major limitation of the basic encoder-decoder architecture is the fixed-length representation through which the entire source sentence must pass. This becomes increasingly problematic as sentence length increases. Attention mechanisms were introduced to address this information bottleneck by allowing the decoder to access different encoder states dynamically during generation.

This experiment investigates how model architecture affects translation quality, uncertainty, and computational cost under a controlled experimental protocol. The four architectures form a progression from a simple recurrent baseline to attention-guided encoder-decoder systems.

---

## 2. Research Objectives

The objectives of this experiment are:

1. To establish a Vanilla RNN sequence-to-sequence baseline.
2. To evaluate the effect of replacing the Vanilla RNN with a GRU encoder-decoder.
3. To determine whether Bahdanau additive attention improves translation quality.
4. To determine whether Luong multiplicative attention provides a different accuracy-efficiency trade-off.
5. To compare BLEU, perplexity, and inference latency under a common experimental setup.
6. To visually inspect learned attention alignments.
7. To identify characteristic translation errors associated with each architecture.

---

## 3. Methodology

### 3.1 Dataset

The dataset was obtained from the ManyThings.org Anki translation collection.

* **Dataset:** English-French
* **Source:** [ManyThings.org Anki Corpus](https://www.manythings.org/anki/fra-eng.zip)

The corpus contains parallel sentence pairs consisting of English sentences and their corresponding French translations.

After cleaning and sequence-length filtering, **240,486** usable sentence pairs were available. For this experiment, **50,000** sentence pairs were selected and divided into:

* Training: **40,000** pairs (80%)
* Validation: **5,000** pairs (10%)
* Test: **5,000** pairs (10%)

All partitioning used a deterministic random seed of **4**.

### 3.2 Preprocessing

The preprocessing pipeline performed the following operations:

1. Conversion to lowercase.
2. Unicode normalization of common quotation and apostrophe variants.
3. Whitespace normalization.
4. Whitespace-based tokenization.
5. Removal of sentence pairs exceeding the maximum sequence length.
6. Construction of independent source and target vocabularies.
7. Inclusion of special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`.
8. Padding sequences to a common maximum length of **30 tokens including special tokens**.

The source vocabulary contained **7,846** tokens, while the target vocabulary contained **10,166** tokens.

---

## 4. Experimental Setup

All four architectures were trained using the same major hyperparameters:

| Hyperparameter | Value |
|---|---|
| Random seed | 4 |
| Embedding dimension | 256 |
| Hidden dimension | 256 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Padding index | `<PAD>` ignored |
| Gradient clipping | 1.0 |
| Epochs | 10 |
| Initial teacher forcing | 1.00 |
| Final teacher forcing | 0.50 |
| Maximum sequence length | 30 |
| Device | cuda |

---

## 5. Model Architectures

### 5.1 Architecture 1: Vanilla RNN Encoder-Decoder
The first model uses a conventional Elman-style Vanilla RNN in both the encoder and decoder. The decoder uses the encoder's final hidden state as its initial state. This architecture provides a simple recurrent baseline and does not use an explicit attention mechanism.

### 5.2 Architecture 2: GRU Encoder-Decoder Without Attention
The second architecture replaces the Vanilla RNN with gated recurrent units (GRU), using update and reset gates to control information flow. Unlike the attention models, the decoder still receives information exclusively through a fixed encoder representation.

### 5.3 Architecture 3: Bahdanau Additive Attention
Computes a compatibility score between the decoder's current hidden state and every encoder hidden state using a nonlinear additive compatibility function, allowing the decoder to receive a dynamically computed source-context vector at every target generation step.

### 5.4 Architecture 4: Luong Multiplicative Attention
Computes compatibility through a learned multiplicative interaction (using a learned linear projection matrix followed by a dot product) rather than a nonlinear additive function.

---

## 6. Training Procedure

The decoder was trained using teacher forcing, where the ground-truth target token was provided with probability $p$, and the model's own prediction was used otherwise. The probability $p$ decreased linearly from $1.00$ to $0.50$ over 10 epochs. Cross entropy loss (excluding padding tokens) and gradient norm clipping at 1.0 were applied.

---

## 7. Evaluation Metrics

* **BLEU-4:** Measures n-gram overlap between generated translations and reference translations (reported as a percentage).
* **Perplexity:** Calculated from teacher-forced test cross entropy ($PPL = \exp(\mathcal{L})$). Lower perplexity indicates higher probability assigned to observed target tokens.
* **Inference Latency:** Measured using autoregressive greedy decoding (average time required to translate one sentence, with explicit CUDA synchronization).

---

## 8. Quantitative Results

| Model | Parameters | BLEU | Perplexity | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Training Time (min) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vanilla RNN** | 7,486,902 | 1.03 | 41.18 | 3.39 | 3.34 | 3.55 | 9.25 |
| **GRU Seq2Seq** | 8,013,238 | 13.33 | 12.12 | 5.47 | 5.10 | 8.28 | 9.51 |
| **Bahdanau Attention** | 13,546,166 | **24.56** | **6.71** | 11.14 | 10.21 | 19.52 | 15.21 |
| **Luong Attention** | 13,480,374 | 20.37 | 8.22 | 9.62 | 8.92 | 15.99 | 14.31 |

### 📊 Artifact Verification & Results Access
All underlying configuration settings, computed metrics, and generated evaluation artifacts are systematically stored within the repository structure:
* **Structured Results:** Detailed data logs can be inspected inside the `results/` directory:
  * Raw metrics table: `results/quantitative_results.csv`
  * Qualitative text inspections: `results/qualitative_translations.csv`
  * Saved configuration state: `results/experiment_config.json`
* **Performance Plots:** Visual evaluation artifacts are archived under the `plots/` directory for direct review.

---

## 9. Qualitative Translation Assessment

| Complexity | English | Reference | Vanilla RNN | GRU Seq2Seq | Bahdanau Attention | Luong Attention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Simple** | you won't need that. | vous n'en aurez pas besoin. | je ne suis pas | tu ne dois pas faire ça. | vous ne pas pas besoin de cela. | vous ne pas besoin de ça. |
| **Complex** | tom has a warped sense of humor. | tom a un sens de l'humour tordu. | je ne suis pas | tom a une personnalité de | tom a une brève de de | tom a une barbe de l'humour. |
| **Long** | she made me so angry on the telephone that i hung up on her. | elle m'a mise tellement en colère au téléphone que je lui ai raccroché au nez. | je ne suis pas | elle m'a dit que je devais acheter la veille de noël. | elle me fait tellement énervée dans ce que j'ai | elle m'a fait si près ce que j'ai entendu sur le |

---

## 10. Visual Demonstrations & Generated Figures

To provide rigorous empirical demonstration suitable for a formal research demonstration, the model tracking logs, loss convergence, and cross-attention mechanics have been visualized and saved as assets. You can inspect or reference the following figures directly from the `plots/` directory:

### 10.1 Training Dynamics & Loss Convergence
* **Loss Curves (`plots/loss_curves_all_models.png`):** Illustrates the multi-epoch training and validation loss progression across all four models, demonstrating convergence rates and stability under the shared schedule.
* **BLEU Comparison (`plots/test_blues_comparison.png`):** Visualizes the performance gap on test BLEU scores across the recurrent baseline and attentional variants.

### 10.2 Efficiency and Trade-off Analysis
* **Latency Benchmarks (`plots/inference_latency_comparison.png`):** Compares mean, median, and tail latency profiles across architectures.
* **Trade-off Frontier (`plots/bleu_vs_latency_trade_off.png`):** Mappings showing the precise cost-accuracy curve between inference latency and translation quality (BLEU).

### 10.3 Attention Alignment Analysis
To evaluate internal reasoning, alignment heatmaps track how source tokens correlate with target generation steps:
* **Bahdanau Heatmap (`plots/bahdanau_attention_attention_heatmap.png` & `plots/bahdanau_vs_luong_attention.png`):** Displays the additive alignment distribution showing focused diagonal patterns for matching source-target sequences.
* **Luong Heatmap (`plots/luong_attention_attention_heatmap.png`):** Displays the multiplicative dot-product attention alignment weights.

---

## 11. Discussion and Conclusion

This study provides a controlled comparison of four recurrent NMT architectures. The measured results show that **Bahdanau Attention** achieved the highest BLEU score of **24.564** and the lowest perplexity of **6.712**. 

The findings illustrate the central motivation for attention: a fixed encoder representation creates an information bottleneck for longer sequences, whereas attention permits the decoder to dynamically access source-side representations. Because attention introduces additional computational overhead, architecture selection ultimately depends on the desired balance between translation accuracy and inference speed.

---

## 12. Reproducibility

All experiments enforced strict determinism using fixed random seeds ($4$) across Python, NumPy, PyTorch CPU, and PyTorch CUDA alongside deterministic CUDA configurations. Results may vary slightly depending on specific hardware and library versions.
