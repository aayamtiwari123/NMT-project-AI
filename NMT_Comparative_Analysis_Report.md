# Comparative Analysis of Neural Machine Translation Architectures

## Abstract

This study presents a controlled comparative evaluation of four sequence-to-sequence neural machine translation architectures for English-to-French translation using the ManyThings.org Anki parallel corpus. The evaluated architectures are: (1) a Vanilla RNN encoder-decoder baseline, (2) a GRU encoder-decoder without attention, (3) a GRU encoder-decoder with Bahdanau additive attention, and (4) a GRU encoder-decoder with Luong multiplicative attention.

The experimental protocol uses a common dataset, deterministic 80/10/10 train-validation-test partitioning with seed 4, common embedding and hidden dimensions, identical batch size, optimizer, learning rate, loss function, sequence-length constraints, and a progressively reduced teacher-forcing schedule. Models are evaluated using BLEU-4, teacher-forced perplexity, and autoregressive inference latency. Attention alignment heatmaps are additionally analyzed for the two attention-based architectures.

The best BLEU score was obtained by **Bahdanau Attention**, with a BLEU-4 score of **24.564**. The lowest measured perplexity was obtained by **Bahdanau Attention**, while the lowest mean inference latency was observed for **Vanilla RNN**.

---

# 1. Introduction

Neural Machine Translation (NMT) models learn to transform a sequence in a source language into a sequence in a target language. Early neural sequence-to-sequence systems commonly used an encoder-decoder architecture in which the encoder converted the source sequence into a hidden representation and the decoder generated the target sequence from that representation.

A major limitation of the basic encoder-decoder architecture is the fixed-length representation through which the entire source sentence must pass. This becomes increasingly problematic as sentence length increases. Attention mechanisms were introduced to address this information bottleneck by allowing the decoder to access different encoder states dynamically during generation.

This experiment investigates how model architecture affects translation quality, uncertainty, and computational cost under a controlled experimental protocol. The four architectures form a progression from a simple recurrent baseline to attention-guided encoder-decoder systems.

---

# 2. Research Objectives

The objectives of this experiment are:

1. To establish a Vanilla RNN sequence-to-sequence baseline.
2. To evaluate the effect of replacing the Vanilla RNN with a GRU encoder-decoder.
3. To determine whether Bahdanau additive attention improves translation quality.
4. To determine whether Luong multiplicative attention provides a different accuracy-efficiency trade-off.
5. To compare BLEU, perplexity, and inference latency under a common experimental setup.
6. To visually inspect learned attention alignments.
7. To identify characteristic translation errors associated with each architecture.

---

# 3. Methodology

## 3.1 Dataset

The dataset was obtained from the ManyThings.org Anki translation collection.

**Dataset:** English-French

**Source:** https://www.manythings.org/anki/fra-eng.zip

The corpus contains parallel sentence pairs consisting of English sentences and their corresponding French translations.

After cleaning and sequence-length filtering, **240,486** usable sentence pairs were available.

For this experiment, **50,000** sentence pairs were selected.

The selected data were divided into:

- Training: **40,000** pairs (80%)
- Validation: **5,000** pairs (10%)
- Test: **5,000** pairs (10%)

All partitioning used a deterministic random seed of **4**.

---

## 3.2 Preprocessing

The preprocessing pipeline performed the following operations:

1. Conversion to lowercase.
2. Unicode normalization of common quotation and apostrophe variants.
3. Whitespace normalization.
4. Whitespace-based tokenization.
5. Removal of sentence pairs exceeding the maximum sequence length.
6. Construction of independent source and target vocabularies.
7. Inclusion of the special tokens:
   - `<PAD>`
   - `<SOS>`
   - `<EOS>`
   - `<UNK>`
8. Padding sequences to a common maximum length.

The maximum sequence length was:

**30 tokens including special tokens.**

The source vocabulary contained **7,846** tokens, while the target vocabulary contained **10,166** tokens.

---

# 4. Experimental Setup

All four architectures were trained using the same major hyperparameters.

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

The same training and validation procedure was applied to every architecture.

---

# 5. Model Architectures

## 5.1 Architecture 1: Vanilla RNN Encoder-Decoder

The first model uses a conventional Elman-style Vanilla RNN in both the encoder and decoder.

The encoder processes:

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

The decoder then uses the encoder's final hidden state as its initial state.

This architecture provides a simple recurrent baseline and does not use an explicit attention mechanism.

---

## 5.2 Architecture 2: GRU Encoder-Decoder Without Attention

The second architecture replaces the Vanilla RNN with gated recurrent units.

A GRU uses update and reset gates to control information flow:

$$
z_t = \sigma(W_z x_t + U_z h_{t-1})
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1})
$$

$$
\tilde{h}_t = \tanh(W_hx_t + U_h(r_t \odot h_{t-1}))
$$

$$
h_t = (1-z_t)\odot h_{t-1} + z_t\odot \tilde{h}_t
$$

Unlike the attention models, the decoder still receives information through a fixed encoder representation.

---

# 6. Bahdanau Additive Attention

Bahdanau attention computes a compatibility score between the decoder's current hidden state and every encoder hidden state.

For decoder state $s_t$ and encoder state $h_i$:

$$
e_{t,i} = v_a^T \tanh(W_hh_i + W_ss_t)
$$

The normalized attention distribution is:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
$$

The context vector is:

$$
c_t = \sum_i \alpha_{t,i}h_i
$$

The decoder therefore receives a dynamically computed source-context vector at every target generation step.

The principal advantage is that the source sentence is no longer represented exclusively by a single fixed vector.

---

# 7. Luong Multiplicative Attention

The implementation uses general Luong attention.

The compatibility score is:

$$
e_{t,i} = s_t^T W_a h_i
$$

where $W_a$ is a learned projection matrix.

The attention distribution is:

$$
\alpha_{t,i} = \operatorname{softmax}(e_{t,i})
$$

and the context vector is:

$$
c_t = \sum_i \alpha_{t,i}h_i
$$

Compared with Bahdanau attention, the compatibility calculation uses a learned multiplicative projection and dot product rather than a nonlinear additive compatibility function.

---

# 8. Training Procedure

The decoder was trained using teacher forcing.

At each decoder step, the ground-truth target token was provided with probability $p$, while the model's own prediction was used otherwise.

The teacher-forcing probability decreased linearly from:

$$
p_{start} = 1.00
$$

to:

$$
p_{end} = 0.50
$$

over 10 epochs.

This provides strong supervision during early training while progressively increasing the model's exposure to its own autoregressive predictions.

The loss function was cross entropy with the padding token excluded:

$$
\mathcal{L} = -\frac{1}{N} \sum_t \log P(y_t|y_{<t},x)
$$

where padded target positions do not contribute to the loss.

Gradient norm clipping was applied at 1.0.

---

# 9. Evaluation Metrics

## 9.1 BLEU

BLEU-4 measures n-gram overlap between generated translations and reference translations.

The corpus BLEU score is reported as a percentage in the results table.

BLEU evaluates the quality of generated sequences rather than the model's internal probability calibration.

---

## 9.2 Perplexity

Perplexity was calculated from teacher-forced test cross entropy:

$$
PPL = \exp(\mathcal{L})
$$

Lower perplexity indicates that the model assigns higher probability to the observed target tokens under teacher forcing.

Perplexity and BLEU measure different properties and therefore do not necessarily produce identical model rankings.

---

## 9.3 Inference Latency

Inference latency was measured using autoregressive greedy decoding.

The reported value is the average time required to translate one sentence.

For CUDA execution, device synchronization was explicitly performed before and after timing to prevent asynchronous GPU execution from producing artificially low measurements.

---

# 10. Quantitative Results

| Model              |   Parameters |     BLEU |   Perplexity |   Mean Latency (ms) |   Median Latency (ms) |   P95 Latency (ms) |   Training Time (min) |
|:-------------------|-------------:|---------:|-------------:|--------------------:|----------------------:|-------------------:|----------------------:|
| Vanilla RNN        |      7486902 |  1.03103 |     41.181   |             3.38515 |               3.3365  |            3.55459 |               9.24752 |
| GRU Seq2Seq        |      8013238 | 13.3263  |     12.117   |             5.4716  |               5.09511 |            8.28465 |               9.50733 |
| Bahdanau Attention |     13546166 | 24.5644  |      6.71174 |            11.1353  |              10.2141  |           19.5234  |              15.2075  |
| Luong Attention    |     13480374 | 20.3704  |      8.22416 |             9.61585 |               8.92098 |           15.9894  |              14.3096  |

The highest BLEU score was obtained by **Bahdanau Attention** with **24.564 BLEU**.

The lowest perplexity was obtained by **Bahdanau Attention** with a perplexity of **6.712**.

The lowest mean inference latency was achieved by **Vanilla RNN**, with **3.385 ms per sentence**.

---

# 11. Qualitative Translation Assessment

The following examples were selected from the held-out test set to represent simple, complex, and long source sentences.

| Complexity   | English                                                      | Reference                                                                      | Vanilla RNN    | GRU Seq2Seq                                           | Bahdanau Attention                              | Luong Attention                                  |
|:-------------|:-------------------------------------------------------------|:-------------------------------------------------------------------------------|:---------------|:------------------------------------------------------|:------------------------------------------------|:-------------------------------------------------|
| Simple       | you won't need that.                                         | vous n'en aurez pas besoin.                                                    | je ne suis pas | tu ne dois pas faire ça.                              | vous ne pas pas besoin de cela.                 | vous ne pas besoin de ça.                        |
| Complex      | tom has a warped sense of humor.                             | tom a un sens de l'humour tordu.                                               | je ne suis pas | tom a une personnalité de                             | tom a une brève de de                           | tom a une barbe de l'humour.                     |
| Long         | she made me so angry on the telephone that i hung up on her. | elle m'a mise tellement en colère au téléphone que je lui ai raccroché au nez. | je ne suis pas | elle m'a dit que je devais acheter la veille de noël. | elle me fait tellement énervée dans ce que j'ai | elle m'a fait si près ce que j'ai entendu sur le |

## 11.1 Simple Sentences

Simple sentences generally place limited demands on long-range source-context preservation. Therefore, even the non-attentional models can often produce acceptable translations.

Errors in this category are more likely to involve lexical selection, unknown words, grammatical agreement, or target vocabulary limitations rather than severe information loss.

## 11.2 Complex Sentences

Complex sentences introduce additional dependencies and syntactic structures. The decoder must preserve relationships between source words that may be separated by multiple positions.

Attention mechanisms can improve this behavior by allowing the decoder to selectively retrieve relevant encoder states instead of depending exclusively on a compressed representation.

## 11.3 Long Sentences

Long sentences provide the strongest test of the fixed-vector bottleneck.

When the source sentence is long, the final encoder state must summarize substantially more information. Attention-based models have an architectural advantage because each decoder step can access the complete sequence of encoder representations.

The observed examples should therefore be interpreted together with the BLEU and perplexity results rather than as isolated anecdotal evidence.

---

# 12. Rigorous Discussion and Error Analysis

The Vanilla RNN baseline obtained 1.03 BLEU compared with 13.33 BLEU for the GRU encoder-decoder. This difference is consistent with the greater difficulty that a simple recurrent transition can have in maintaining useful information over longer sequences. However, the result should not be interpreted as evidence that the recurrent cell alone causes every performance difference, because the architectures intentionally differ in recurrent-cell type.

The principal architectural limitation of the standard GRU encoder-decoder is its fixed-length information interface. The complete source sequence is compressed into the encoder's final hidden representation before decoding begins. This creates an information bottleneck, particularly for long or structurally complex sentences. Attention removes much of this restriction by allowing the decoder to construct a context vector dynamically from encoder states at every generation step.

Bahdanau attention obtained 24.56 BLEU, which is +11.24 BLEU relative to the non-attentional GRU baseline. Its additive scoring function explicitly learns a nonlinear compatibility between the current decoder state and each encoder state. This provides a flexible alignment mechanism and can be especially useful when source and target positions are not strictly monotonic.

Luong multiplicative attention obtained 20.37 BLEU, corresponding to a +7.04 BLEU difference relative to the standard GRU baseline. Its general multiplicative scoring mechanism computes compatibility through a learned linear projection followed by a dot product. This is structurally simpler than additive attention and can be computationally efficient, although the empirical result determines whether that efficiency translates into a favorable overall trade-off for this dataset.

The measured mean inference latency was 11.135 ms/sentence for Bahdanau attention and 9.616 ms/sentence for Luong attention. Relative to the non-attentional GRU, these represent changes of +103.51% and +75.74%, respectively. Attention requires additional computation at each decoder step, so an increase in inference cost is expected in exchange for access to source-side contextual information.

Perplexity provides a complementary view of model behavior because it measures the model's average uncertainty under teacher-forced decoding. The best measured perplexity was 6.712, obtained by Bahdanau Attention. BLEU and perplexity need not rank models identically: perplexity evaluates token-level probability assignments, whereas BLEU evaluates overlap between generated translations and reference translations.

The attention heatmaps should be interpreted as alignment diagnostics rather than as direct explanations of model reasoning. A strong alignment pattern generally appears as concentrated weights that move through relevant source positions as target words are generated. Diffuse attention can indicate uncertainty, while repeated or misplaced peaks may accompany repetition, omission, or incorrect word ordering. The Bahdanau and Luong heatmaps should therefore be examined together with the corresponding translations.

Qualitative assessment is particularly important for NMT because corpus-level BLEU can hide systematic translation errors. Short sentences generally require relatively little long-range memory. Complex and long sentences place greater demands on source-context preservation, lexical selection, word ordering, and alignment. Consequently, differences between the standard encoder-decoder and the two attention models are expected to become more visible as sentence length and syntactic complexity increase.

## 12.1 Information Bottleneck

The standard encoder-decoder architecture forces the source sequence through a fixed-size hidden representation.

For a source sequence:

$$
x_1,x_2,\ldots,x_T
$$

the encoder produces a sequence of hidden states:

$$
h_1,h_2,\ldots,h_T
$$

but a conventional encoder-decoder without attention primarily transfers information to the decoder through the final hidden representation.

This creates a structural information bottleneck.

Attention changes the interface from:

$$
\text{Source sequence} \rightarrow \text{single representation} \rightarrow \text{decoder}
$$

to:

$$
\text{Source sequence} \rightarrow \text{all encoder states} \rightarrow \text{dynamic context} \rightarrow \text{decoder}
$$

The latter gives the decoder a mechanism for selecting source information relevant to each generated target token.

---

## 12.2 Bahdanau vs Luong

Bahdanau attention uses a nonlinear additive compatibility function:

$$
v_a^T \tanh(W_hh_i + W_ss_t)
$$

Luong general attention instead uses:

$$
s_t^T W_a h_i
$$

The two mechanisms therefore differ in how compatibility between source and target representations is learned.

Bahdanau attention introduces a nonlinear transformation before scoring, whereas Luong attention uses a multiplicative interaction after projecting the encoder representation.

The empirical results in this experiment determine which mechanism is more effective for this particular dataset and training configuration.

---

## 12.3 Speed vs Accuracy

Attention introduces additional computation because every decoder step must calculate compatibility scores over source positions.

Therefore, the computational complexity of decoding increases with source sequence length.

The results demonstrate the practical trade-off between translation quality and computational cost.

A model with slightly higher BLEU but substantially higher latency may be preferable in an offline translation system, while a low-latency model may be preferable in interactive or resource-constrained applications.

Consequently, architecture selection should not be based exclusively on BLEU.

---

## 12.4 Attention Alignment Analysis

The generated heatmaps are saved under:

`nmt_comparative_analysis/plots`

The primary attention comparison is:

`bahdanau_vs_luong_attention.png`

The heatmaps provide a visual representation of the conditional alignment distribution:

$$
\alpha_{t,i}
$$

where each row corresponds approximately to a target generation step and each column corresponds to a source token.

Sharp concentration around relevant source positions indicates focused alignment.

Diffuse distributions may indicate uncertainty or that several source positions contribute simultaneously.

Repeated attention peaks can accompany repetition or decoding difficulties, while skipped source positions can be associated with omissions.

These visualizations should be interpreted cautiously: attention weights are useful diagnostic signals, but they should not automatically be treated as complete causal explanations of model behavior.

---

# 13. Error Categories

The principal error categories to inspect in this experiment are:

### 13.1 Omission

A source word or phrase is not represented in the generated translation.

This can arise when the model fails to preserve information about part of the source sequence.

### 13.2 Addition

The decoder generates content that is not supported by the source sentence.

This can arise from language-model bias or autoregressive error accumulation.

### 13.3 Repetition

A word or phrase is generated multiple times.

This is a common failure mode of autoregressive sequence generation.

### 13.4 Lexical Substitution

The model generates a semantically related but incorrect word.

### 13.5 Word-Order Error

The translation contains the correct words but places them incorrectly.

### 13.6 Agreement and Morphological Errors

The model selects an inappropriate gender, number, tense, or inflection.

### 13.7 Long-Range Dependency Failure

Information required later in the sentence is incorrectly preserved or lost.

This category is particularly relevant when comparing the fixed-vector models with attention-based architectures.

---

# 14. Reproducibility

The experiment used:

- Python random seed: 4
- NumPy random seed: 4
- PyTorch CPU seed: 4
- PyTorch CUDA seed: 4
- CUDA deterministic configuration
- deterministic train/validation/test partitioning
- deterministic vocabulary construction
- fixed model hyperparameters
- fixed maximum sequence length
- fixed batch size
- fixed optimizer
- fixed teacher-forcing schedule

The experiment was executed on:

**cuda**

Exact results can nevertheless vary across different hardware, CUDA versions, PyTorch versions, and library implementations despite deterministic settings.

---

# 15. Limitations

Several limitations should be considered.

First, the experiment uses a relatively simple whitespace tokenizer rather than a modern subword tokenizer such as BPE or SentencePiece.

Second, the models use relatively compact recurrent architectures compared with modern Transformer NMT systems.

Third, BLEU has known limitations and should not be interpreted as a complete measure of translation quality.

Fourth, perplexity is measured under teacher forcing and therefore does not directly measure fully autoregressive decoding behavior.

Fifth, the Vanilla RNN differs from the other three architectures in recurrent cell type. Consequently, the comparison between Architecture 1 and the other architectures includes both recurrent-cell and attention-related differences. The most controlled attention comparison is therefore between the GRU-only, Bahdanau, and Luong architectures.

Sixth, the experiment uses a fixed maximum sequence length of 30 tokens. Sentences exceeding this length are excluded.

---

# 16. Conclusion

This study provides a controlled comparison of four recurrent NMT architectures ranging from a Vanilla RNN baseline to attention-equipped GRU encoder-decoder models.

The measured results show that **Bahdanau Attention** achieved the highest BLEU score of **24.564**, while **Bahdanau Attention** achieved the lowest perplexity of **6.712**.

The findings illustrate the central motivation for attention in sequence-to-sequence translation: a fixed encoder representation can become an information bottleneck, whereas attention permits the decoder to dynamically access source-side representations during generation.

At the same time, attention introduces additional computational overhead. The appropriate architecture therefore depends on the desired balance between translation accuracy and inference speed.

Overall, the experiment demonstrates that architectural changes can materially affect NMT performance even when the dataset, vocabulary constraints, optimization strategy, and principal hyperparameters are held constant.

---

# 17. Future Directions

Several extensions would make the experimental study stronger:

1. Replace the whitespace tokenizer with BPE or SentencePiece.
2. Repeat the experiment across multiple random seeds.
3. Report confidence intervals or bootstrap significance tests for BLEU.
4. Evaluate chrF and COMET in addition to BLEU.
5. Compare GRU and LSTM architectures independently.
6. Add bidirectional encoders.
7. Compare dot-product, general, and concat attention variants.
8. Investigate beam-search decoding instead of greedy decoding.
9. Analyze performance as a function of source sentence length.
10. Evaluate the models on out-of-domain test data.
11. Compare recurrent architectures against Transformer-based NMT.
12. Conduct an explicit human evaluation of adequacy and fluency.

---

# 18. Generated Artifacts

The experiment generated the following artifacts:

- `quantitative_results.csv`
- `qualitative_translations.csv`
- `experiment_config.json`
- `loss_curves_all_models.png`
- `test_bleu_comparison.png`
- `inference_latency_comparison.png`
- `bleu_vs_latency_tradeoff.png`
- `bahdanau_attention_heatmap.png`
- `luong_attention_heatmap.png`
- `bahdanau_vs_luong_attention.png`
- model checkpoints under `checkpoints/`

---

# 19. Final Result Summary

| Criterion | Best Architecture | Value |
|---|---|---:|
| BLEU-4 | Bahdanau Attention | 24.564 |
| Perplexity | Bahdanau Attention | 6.712 |
| Mean Latency | Vanilla RNN | 3.385 ms |
| Fewest Parameters | Vanilla RNN | 7,486,902 |

The final interpretation should consider all three evaluation dimensions rather than selecting an architecture solely from its BLEU score.
