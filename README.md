<!-- ===== SECTION 1: BANNER (Capsule Render) ===== -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1b27,100:4a6cf7&height=220&section=header&text=LLM%20Engineering%20Handbook&fontSize=42&fontColor=ffffff&fontAlignY=35&desc=From%20Transformer%20Internals%20to%20Agentic%20Workflows&descSize=18&descAlignY=55&animation=fadeIn" alt="Banner" width="100%" />
</p>

<!-- ===== SECTION 2: BADGES (Shields.io) ===== -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/github/license/hgabrali/llm-engineering-handbook?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/github/last-commit/hgabrali/llm-engineering-handbook?style=for-the-badge" alt="Last Commit" />
  <img src="https://img.shields.io/github/stars/hgabrali/llm-engineering-handbook?style=for-the-badge&color=yellow" alt="Stars" />
  <img src="https://img.shields.io/github/forks/hgabrali/llm-engineering-handbook?style=for-the-badge&color=blue" alt="Forks" />
  <img src="https://img.shields.io/github/issues/hgabrali/llm-engineering-handbook?style=for-the-badge" alt="Issues" />
  <img src="https://img.shields.io/badge/LLM-Engineering-blueviolet?style=for-the-badge&logo=openai&logoColor=white" alt="LLM Engineering" />
</p>

<!-- ===== SECTION 3: CENTERED DESCRIPTION ===== -->
<p align="center">
  <em>A comprehensive, practitioner-oriented reference on the core building blocks of modern Large Language Models — from low-level transformer mechanics to production-grade agentic systems.</em>
</p>

---

<!-- ===== DARK/LIGHT THEME RESPONSIVE IMAGE ===== -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hgabrali/llm-engineering-handbook/main/Large%20Language%20Models.png" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/hgabrali/llm-engineering-handbook/main/Large%20Language%20Models.png" />
    <img alt="LLM Architecture Overview" src="https://raw.githubusercontent.com/hgabrali/llm-engineering-handbook/main/Large%20Language%20Models.png" width="700" />
  </picture>
</p>

---

<!-- ===== COLLAPSIBLE TABLE OF CONTENTS ===== -->
<details>
<summary><strong>📑 Table of Contents (click to expand)</strong></summary>

1. [How Transformers Actually Work](#1-how-transformers-actually-work)
2. [Decoding Strategies & Mixture-of-Experts](#2-decoding-strategies--mixture-of-experts)
3. [LLM Finetuning](#3-llm-finetuning)
4. [Evaluation Techniques](#4-evaluation-techniques)
5. [Optimization Tricks](#5-optimization-tricks)
6. [Reasoning & Scaling](#6-reasoning--scaling)
7. [Agentic Workflows](#7-agentic-workflows)
8. [References & Further Reading](#8-references--further-reading)
9. [LLM Deep Dive: Expert Technical Reference](#llm-deep-dive-expert-technical-reference)

</details>

---

## 1. How Transformers Actually Work

### 1.1 Tokenization

Raw text must be converted into discrete integer IDs before a neural network can process it. Modern LLMs rely on **subword tokenization** algorithms that strike a balance between vocabulary size and sequence length.

| Algorithm | Core Idea | Used By |
|-----------|-----------|---------|
| Byte-Pair Encoding (BPE) | Iteratively merge the most frequent adjacent byte/character pairs | GPT-2/3/4, LLaMA |
| WordPiece | Likelihood-based merging similar to BPE | BERT, DistilBERT |
| Unigram / SentencePiece | Probabilistic subword model; trims from a large initial vocab | T5, mT5, PaLM |
| Byte-Level BPE | Operates directly on raw UTF-8 bytes — no unknown tokens | GPT-2, LLaMA-3 |

**Key concepts:**
- **Vocabulary size trade-off:** A larger vocabulary shortens sequences (faster inference) but inflates the embedding matrix and may underfit rare tokens.
- **Special tokens:** `[BOS]`, `[EOS]`, `[PAD]`, `[UNK]`, `[MASK]` — each carries model-specific semantics.
- **Pre-tokenization:** Whitespace splitting, regex patterns applied before BPE.

```python
# Example: tiktoken (OpenAI)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Transformers are powerful.")
print(tokens)
```

### 1.2 Embeddings

Tokens are projected into a continuous vector space through a learned embedding matrix **E ∈ R^{V x d}**, where V is the vocabulary size and d is the model dimension.

**Types of embeddings in a transformer:**
- **Token embeddings:** Lookup table mapping token IDs to dense vectors.
- **Positional embeddings:** Encode the order of tokens in a sequence. Originally sinusoidal (Vaswani et al., 2017); modern models prefer **Rotary Position Embeddings (RoPE)** or **ALiBi**.
- **Segment embeddings (encoder models):** Distinguish sentence A from sentence B in tasks like NLI.

### 1.3 Self-Attention

Self-attention is the mechanism that allows every token to attend to every other token in the sequence.

```
Q = X * W_Q   (queries)
K = X * W_K   (keys)
V = X * W_V   (values)

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

**Multi-Head Attention (MHA):** Runs h parallel heads with smaller dimensionality, then concatenates and projects.

**Variants in modern LLMs:**

| Variant | Description | Models |
|---------|-------------|--------|
| Multi-Head Attention (MHA) | Standard h independent heads | GPT-2, BERT |
| Multi-Query Attention (MQA) | Single shared K, V across all heads | PaLM, Falcon |
| Grouped-Query Attention (GQA) | K, V shared within groups of heads | LLaMA-2 70B, Mistral |

### 1.4 Feed-Forward Network (FFN)

Modern models use **SwiGLU** (LLaMA, PaLM) or **GeGLU** instead of ReLU, introducing a gating mechanism.

### 1.5 Layer Normalization

- **Post-LayerNorm:** Original transformer. Norm after residual addition.
- **Pre-LayerNorm (Pre-LN):** Norm before attention/FFN. Used by GPT-2+, LLaMA.
- **RMSNorm:** Removes mean-centering from LayerNorm. Used by LLaMA, Mistral.

---

## 2. Decoding Strategies & Mixture-of-Experts

### 2.1 Decoding Strategies

| Strategy | Mechanism | Trade-off |
|----------|-----------|-----------|
| Greedy | Pick argmax at each step | Fast but repetitive |
| Beam Search | Track top-k partial sequences | Better quality, more compute |
| Temperature Sampling | Divide logits by T before softmax | T<1 sharper, T>1 creative |
| Top-k Sampling | Zero out all but top k tokens | Limits low-prob noise |
| Top-p (Nucleus) | Keep smallest set with cumulative prob >= p | Adaptive threshold |
| Min-p Sampling | Keep tokens with prob >= p * max_prob | Scales with confidence |
| Repetition Penalty | Penalize previously generated tokens | Reduces loops |
| Contrastive Search | Balances confidence with diversity | High quality, slower |

**Speculative decoding:** Use a small draft model to generate k candidate tokens, verify in parallel with the large model.

### 2.2 Mixture-of-Experts (MoE)

MoE replaces the dense FFN block with multiple expert sub-networks, only activating a subset per token.

| Model | Experts | Active | Total Params | Active Params |
|-------|---------|--------|-------------|---------------|
| Switch Transformer | 128 | 1 | 1.6T | ~12B |
| Mixtral 8x7B | 8 | 2 | 46.7B | ~12.9B |
| DeepSeek-V2 | 160 | 6 | 236B | ~21B |
| DBRX | 16 | 4 | 132B | ~36B |

---

## 3. LLM Finetuning

### 3.1 Full Finetuning

All model parameters are updated. Simple but expensive in memory and risks catastrophic forgetting.

### 3.2 Parameter-Efficient Finetuning (PEFT) — LoRA

LoRA freezes W and adds a low-rank decomposition: `W' = W + (alpha/r) * B @ A`

| Variant | Key Difference |
|---------|---------------|
| QLoRA | 4-bit NormalFloat base model, LoRA adapters in fp16 |
| DoRA | Decomposes weight into magnitude + direction |
| LoRA+ | Different learning rates for A and B |
| rsLoRA | Scales by 1/sqrt(r) for rank-stability |
| VeRA | Shared frozen random A, B; trains scaling vectors |

### 3.3 Supervised Finetuning (SFT)

Trains on (instruction, response) pairs. Loss computed only on response tokens.

### 3.4 RLHF

**Stage 1 — Reward Model:** Bradley-Terry loss on human preference pairs.

**Stage 2 — PPO:** Maximize reward with KL penalty to stay near SFT distribution.

| Method | Key Idea |
|--------|----------|
| DPO | Direct preference optimization without reward model |
| KTO | Binary feedback instead of pairwise |
| ORPO | Single-stage SFT + alignment |
| SimPO | Length-normalized, reference-free objective |

### 3.5 Constitutional AI / RLAIF

AI-generated preference labels from a set of principles, enabling scalable alignment.

---

## 4. Evaluation Techniques

### 4.1 Benchmarks

| Benchmark | Measures | Format |
|-----------|----------|--------|
| MMLU | Multitask knowledge (57 subjects) | MC |
| HellaSwag | Commonsense reasoning | MC |
| HumanEval / MBPP | Code generation | pass@k |
| GSM8K | Math reasoning | CoT |
| TruthfulQA | Factual accuracy | Open/MC |
| ARC | Science questions | MC |
| MATH | Competition math | Exact match |
| MT-Bench | Multi-turn conversation | LLM-judged |

### 4.2 LLM-as-a-Judge

Using a strong LLM as automated evaluator for open-ended tasks.

- **Pointwise scoring:** Rate on a rubric (1-10).
- **Pairwise comparison:** Pick the better response.
- **Bias mitigation:** Position swapping, verbosity control, separate judge model.

### 4.3 Task-Specific Metrics

| Metric | Domain | Description |
|--------|--------|-------------|
| BLEU / ROUGE | Translation / Summarization | N-gram overlap |
| BERTScore | General text | Embedding similarity |
| pass@k | Code | Problems solved in k samples |
| Exact Match | QA / Math | Binary correctness |

---

## 5. Optimization Tricks

### 5.1 Rotary Position Embeddings (RoPE)

Encodes position by rotating query/key vectors in 2D subspaces. Relative position captured via dot product depending only on (m-n). Extended via YaRN, NTK-aware scaling.

### 5.2 Quantization

| Method | Precision | Key Idea |
|--------|-----------|----------|
| INT8 (LLM.int8()) | W8A8/W8A16 | Mixed-precision for outlier features |
| GPTQ | W4A16 | Post-training quantization via second-order info |
| AWQ | W4A16 | Protects salient channels by activation distribution |
| GGUF (llama.cpp) | W2-W8 | CPU-friendly mixed precision |
| bitsandbytes NF4 | W4A16 | Optimal for normal distributions |
| SmoothQuant | W8A8 | Migrates difficulty from activations to weights |
| FP8 | W8A8 | Native H100+ support, near-lossless |

### 5.3 Attention & Memory Optimizations

| Technique | How It Works |
|-----------|-------------|
| FlashAttention | IO-aware tiled attention, avoids full N×N in HBM |
| PagedAttention (vLLM) | Non-contiguous KV cache pages |
| GQA / MQA | Shared key-value heads reduce KV cache |
| Sliding Window | Local attention window (Mistral: 4096) |
| Ring Attention | Sequence distributed across devices |

### 5.4 Training Optimizations

- **Mixed-precision (bf16/fp16):** Halves memory, enables tensor cores.
- **Gradient checkpointing:** Trades compute for memory.
- **ZeRO / FSDP:** Shards optimizer states, gradients, parameters.
- **Tensor / Pipeline / Sequence Parallelism:** Distribute across GPUs.

### 5.5 Inference Optimizations

- **Continuous batching:** Dynamic request insertion.
- **Speculative decoding:** Draft + verify in parallel.
- **Prefix caching:** Reuse KV for shared system prompts.
- **Structured generation:** Constrained decoding via FSMs.

---

## 6. Reasoning & Scaling

### 6.1 Scaling Laws

Chinchilla: model size and data should scale equally with compute budget.

### 6.2 Prompting Strategies

| Strategy | Description |
|----------|-------------|
| Chain-of-Thought | Step-by-step intermediate reasoning |
| Self-Consistency | Multiple CoT paths, majority vote |
| Tree-of-Thought | Branch, evaluate, prune reasoning paths |
| ReAct | Interleave reasoning and tool actions |
| Reflexion | Learn from past mistakes |

### 6.3 Process vs Outcome Reward Models

- **ORM:** Scores final answer only.
- **PRM:** Scores each reasoning step — finer-grained but harder to label.

### 6.4 Test-Time Compute Scaling

Best-of-N sampling, iterative refinement, verifier-guided search, MCTS.

---

## 7. Agentic Workflows

### 7.1 RAG (Retrieval-Augmented Generation)

```
Query -> Embed -> Retrieve (Vector DB) -> Augment Prompt -> Generate
```

**Advanced:** Hybrid search (dense + BM25), re-ranking, HyDE, CRAG, Self-RAG.

### 7.2 Tool Calling

Define tools as JSON Schema -> Model outputs structured calls -> Orchestrator executes -> Model continues.

### 7.3 Agent Architectures

| Pattern | Description |
|---------|-------------|
| ReAct | Thought -> Action -> Observation |
| Plan-and-Execute | Plan then execute steps |
| Multi-Agent | Specialized agents collaborate |
| Hierarchical | Manager delegates to workers |

### 7.4 Frameworks

LangChain/LangGraph, LlamaIndex, CrewAI, AutoGen, Semantic Kernel, DSPy.

### 7.5 Memory

Short-term (context window), long-term (vector DB), episodic (summaries), working (scratchpad).

### 7.6 Guardrails

Input (injection detection), output (hallucination/toxicity), tool-use (permission boundaries), monitoring (audit logs).

---

## 8. References

### Foundational
- Vaswani et al. (2017) — Attention Is All You Need
- Brown et al. (2020) — GPT-3
- Touvron et al. (2023) — LLaMA
- Jiang et al. (2024) — Mixtral of Experts

### Finetuning & Alignment
- Hu et al. (2021) — LoRA
- Dettmers et al. (2023) — QLoRA
- Ouyang et al. (2022) — InstructGPT / RLHF
- Rafailov et al. (2023) — DPO

### Optimization
- Su et al. (2021) — RoPE
- Dao et al. (2022) — FlashAttention
- Frantar et al. (2022) — GPTQ
- Lin et al. (2023) — AWQ
- Kwon et al. (2023) — PagedAttention

### Reasoning
- Wei et al. (2022) — Chain-of-Thought
- Hoffmann et al. (2022) — Chinchilla
- Yao et al. (2023) — Tree of Thoughts
- Lightman et al. (2023) — Process Reward Models

### RAG & Agents
- Lewis et al. (2020) — RAG
- Yao et al. (2023) — ReAct
- Shinn et al. (2023) — Reflexion

---

## Stanford CME295

- Lecture 1: https://zurl.co/F0QR5
- Lecture 2: https://zurl.co/hG5lp
- Lecture 3: https://zurl.co/PnKrW
- Lecture 4: https://zurl.co/XCZoE
- Lecture 5: https://zurl.co/GWlYI
- Lecture 6: https://zurl.co/zGqqQ
- Lecture 7: https://zurl.co/T06NM
- Lecture 8: https://zurl.co/Un42q
- Lecture 9: https://zurl.co/rR3YL

---

<!-- ===== SECTION SEPARATOR ===== -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=0:1a1b27,100:4a6cf7&height=2&section=header" alt="separator" width="100%" />
</p>

# LLM Deep Dive: Expert Technical Reference

> A rigorous, sequential treatment of modern large language model architecture, training, and deployment.

---

<details>
<summary><strong>MODULE 1 — How Transformers Actually Work (click to expand)</strong></summary>

### 1.1 Tokenization

Tokenization is not a trivial preprocessing step — it fundamentally shapes what the model can and cannot represent.

#### Byte-Pair Encoding (BPE)

BPE starts with a character-level vocabulary and iteratively merges the most frequent adjacent pair of symbols into a single token. This is a greedy, corpus-frequency-driven compression.

```
Corpus: "low lower lowest"
Init vocab: {l, o, w, e, r, s, t, ' '}
Iteration 1: most frequent pair = (l, o) → merge to "lo"
Iteration 2: (lo, w) → "low"
...
```

The final vocabulary size (e.g., 50,257 for GPT-2; 100,256 for GPT-4o) is a hyperparameter. Larger vocab = fewer tokens per sequence = longer effective context. Smaller vocab = more tokens, better character-level generalization.

#### SentencePiece / Unigram LM

Used by LLaMA, Gemma, T5. Operates directly on raw Unicode (no pre-tokenization whitespace rules). The Unigram variant trains a probabilistic model over subwords and prunes tokens by their marginal log-likelihood contribution — producing a vocabulary that maximizes corpus likelihood under the model.

#### Critical Design Decisions

| Decision | Impact |
|----------|--------|
| Vocabulary size | Embedding matrix size (V × d), coverage of rare words |
| Pre-tokenization rules | Language-specific whitespace, punctuation handling |
| Special tokens | `<BOS>`, `<EOS>`, `<PAD>` — Sequence boundary signaling |
| Byte-level fallback | 100% coverage of any Unicode input |

#### Tokenization Pathologies

- **Numbers split arbitrarily:** `"1234"` → `["12", "34"]` — arithmetic becomes non-local reasoning
- **Non-English scripts over-tokenize** → effectively shorter context budget
- **Prompt injection** via unusual tokenization boundaries

### 1.2 Embeddings

After tokenization, each token index t ∈ {0, ..., V-1} is mapped to a dense vector via an embedding lookup matrix E ∈ ℝ^(V×d_model).

#### Token Embeddings

A learned matrix. Each row is a d-dimensional vector. These are trained end-to-end; geometrically, semantically similar tokens cluster.

#### Positional Encodings

Transformers have no inherent notion of sequence order — attention is permutation-equivariant. Positional encodings inject order information.

**Sinusoidal (Vaswani et al. 2017):**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Learned absolute positional embeddings:** GPT-2, early BERT. Simply a second embedding matrix of size (max_seq_len × d_model).

**RoPE (Rotary Position Embedding):** See Module 5.1. Currently the dominant approach.

**ALiBi:** Adds a linear position bias directly to attention logits before softmax.

### 1.3 The Attention Mechanism

#### Scaled Dot-Product Attention

```
Q = X W_Q,  W_Q ∈ ℝ^(d_model × d_k)
K = X W_K,  W_K ∈ ℝ^(d_model × d_k)
V = X W_V,  W_V ∈ ℝ^(d_model × d_v)

Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

#### Multi-Head Attention (MHA)

```
head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
MHA(X) = Concat(head_1, ..., head_h) W_O
```

#### Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Attention (naive) | O(n² d) | O(n²) |
| FFN | O(n d²) | O(d²) |

#### Causal (Autoregressive) Masking

For decoder-only models (GPT family), position i cannot attend to positions j > i. Enforced by adding −∞ to the upper triangle of the QKᵀ matrix before softmax.

#### KV Cache

During inference, previously computed K and V matrices are cached and reused.

#### Grouped Query Attention (GQA)

Multiple Q heads share a single KV pair. LLaMA-2 70B uses GQA. Reduces KV cache size by a factor of n_groups.

### 1.4 The Transformer Block

```
x = x + Attention(RMSNorm(x))   # residual + self-attention
x = x + FFN(RMSNorm(x))         # residual + feed-forward
```

#### RMSNorm vs LayerNorm

`RMSNorm(x) = x / RMS(x) · γ` where `RMS(x) = √(mean(x²))`. Used by LLaMA, Mistral, Gemma.

#### FFN Block — SwiGLU

```
FFN(x) = (SiLU(x W_1) ⊙ x W_2) W_3
```

where `SiLU(x) = x · σ(x)`. Used by LLaMA, PaLM.

</details>

---

<details>
<summary><strong>MODULE 2 — Decoding Strategies & Mixture of Experts (click to expand)</strong></summary>

### 2.1 Decoding Strategies

#### Greedy Decoding
Always pick argmax P(token | context). Deterministic, fast, but produces repetitive and often degenerate output.

#### Beam Search
Maintain k candidate sequences (beams) at each step, expand each by all vocabulary tokens, keep top-k by cumulative log-probability.

```
score(sequence) = Σ log P(tᵢ | t₁...tᵢ₋₁) / length_penalty
```

#### Temperature Scaling

```
P_T(tᵢ) = softmax(logits / T)
```

- **T < 1:** sharper distribution, more deterministic
- **T > 1:** flatter distribution, more creative/chaotic
- **T → 0:** degenerates to greedy

#### Top-k Sampling
Restrict sampling to the k highest-probability tokens.

#### Top-p (Nucleus) Sampling
Sample from the smallest set of tokens whose cumulative probability ≥ p.

#### Min-p Sampling
Prunes tokens with `P(t) < min_p × max(P(·))`.

#### Speculative Decoding

1. Run a small draft model M_q to generate k tokens speculatively
2. Run the large target model M_p in a single forward pass to score all k tokens
3. Accept tokens where M_q's distribution is "close enough" to M_p's

Achieves ~2-3× speedup. Theoretically lossless.

### 2.2 Mixture of Experts (MoE)

MoE is a conditional computation architecture: only a subset of model parameters are active per input token.

```
MoE(x) = Σᵢ G(x)ᵢ · Eᵢ(x)
```

#### Top-k Gating

```
G(x) = Softmax(TopK(x · W_gate, k))
```

Only k experts (typically k=2) are activated per token.

#### Expert Load Balancing

Auxiliary load balancing loss:

```
L_aux = α × N × Σᵢ fᵢ · pᵢ
```

#### DeepSeek-V3's MoE Innovations

Uses "fine-grained MoE" with 256 experts (top-8 routing), auxiliary-loss-free load balancing via expert bias terms, and a multi-token prediction auxiliary head.

</details>

---

<details>
<summary><strong>MODULE 3 — LLM Finetuning (click to expand)</strong></summary>

### 3.1 Supervised Finetuning (SFT)

```
L_SFT = -Σₜ log P_θ(yₜ | x, y₁...yₜ₋₁)
```

#### Data Quality >> Quantity

The LIMA paper (2023) demonstrated that 1,000 carefully curated examples can match the instruction-following quality of models trained on 50k+ examples.

### 3.2 LoRA Deep Dive

#### Low-Rank Adaptation

```
ΔW = B · A, where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
h = W₀x + (B·A)x · (α/r)
```

**Parameter reduction:** For d=4096, r=16: LoRA = 131K params vs Full = 16.7M → **128× reduction**

#### QLoRA

Enables finetuning a 65B model on a single 48GB GPU:
- **NF4 (NormalFloat4):** quantization scheme optimal for normally distributed weights
- **Double quantization:** quantize the quantization constants themselves
- **Paged optimizers:** use CPU RAM to handle GPU memory spikes

#### LoRA Variants

| Variant | Key Idea |
|---------|----------|
| DoRA | Decomposes weight into magnitude + direction; trains direction via LoRA |
| LoRA+ | Different learning rates for A and B matrices |
| AdaLoRA | Adaptively allocates rank budget across weight matrices via SVD |
| LoRA-FA | Freezes A, only trains B → reduces activation memory |

### 3.3 RLHF

#### Stage 1: Reward Model Training

```
L_RM = -E[(x,yw,yl)] [log σ(R_φ(x, yw) - R_φ(x, yl))]
```

#### Stage 2: PPO

```
L_PPO(θ) = E[R_φ(x, y)] - β · KL[π_θ(y|x) || π_ref(y|x)]
```

#### DPO (Direct Preference Optimization)

```
L_DPO(θ) = -E [log σ(β log(π_θ(yw|x)/π_ref(yw|x)) - β log(π_θ(yl|x)/π_ref(yl|x)))]
```

#### DPO Variants

| Variant | Improvement |
|---------|-------------|
| IPO | Avoids reward overfitting with identity loss |
| KTO | Uses per-sample binary signals instead of pairs |
| SimPO | Removes reference model dependency entirely |
| ORPO | Combines SFT and preference loss into single stage |

</details>

---

<details>
<summary><strong>MODULE 4 — Evaluation Techniques (click to expand)</strong></summary>

### 4.1 Classical Benchmarks

- **MMLU:** 57 academic subjects, 4-choice MCQ. Measures breadth of world knowledge.
- **HellaSwag:** Commonsense NLI — choose most plausible sentence continuation.
- **HumanEval / MBPP:** Code generation evaluated by unit test execution.
- **MATH:** Competition mathematics requiring multi-step symbolic reasoning.
- **GSM8K:** Grade school math word problems.

#### Benchmark Contamination

Training data often contains benchmark test sets. Models can "memorize" answers without reasoning capability. Mitigation: n-gram deduplication, held-out evaluation sets, dynamic benchmarks.

### 4.2 LLM-as-a-Judge

#### Pairwise Comparison

```
System: You are an expert evaluator...
User:
[PROMPT]:    {question}
[RESPONSE A]: {response_a}
[RESPONSE B]: {response_b}
Which response is better? Explain your reasoning, then output "A" or "B".
```

#### Failure Modes of LLM-as-a-Judge

| Bias | Description |
|------|-------------|
| Positional bias | Prefers whichever response appears first |
| Verbosity bias | Longer responses rated higher regardless of quality |
| Self-enhancement bias | GPT-4 prefers GPT-4 outputs |
| Sycophancy | Agrees with human's stated preference even when wrong |
| Calibration failure | Confidently wrong on factual accuracy assessment |

</details>

---

<details>
<summary><strong>MODULE 5 — Optimization Tricks (click to expand)</strong></summary>

### 5.1 RoPE (Rotary Position Embedding)

```
R(m, θ) = [[cos(mθ), -sin(mθ)],
            [sin(mθ),  cos(mθ)]]

q_m = R(m) q,   k_n = R(n) k
(R(m)q)ᵀ (R(n)k) = qᵀ R(m-n)ᵀ k
```

#### Context Length Extension

- **Positional Interpolation (PI):** Scale positions down
- **YaRN:** Different scaling for different frequency bands
- **LongRoPE:** Non-uniform scaling factors per frequency component

### 5.2 Quantization

#### GPTQ

```
min_Q ||WX - QX||²_F
```

Column-wise quantization with lazy batch updates. Near-fp16 quality at 4-bit for large models.

#### AWQ (Activation-Aware Weight Quantization)

Protects ~1% of weight channels corresponding to activation outliers.

#### Data Types Summary

| Dtype | Bits | Range | Use Case |
|-------|------|-------|----------|
| FP32 | 32 | ±3.4e38 | Reference, gradient accumulation |
| BF16 | 16 | ±3.4e38 | Training (same range as FP32) |
| FP16 | 16 | ±65504 | Training, some inference |
| INT8 | 8 | -128 to 127 | Inference, activations |
| INT4 | 4 | -8 to 7 | Weight-only quantization |
| NF4 | 4 | (non-linear) | QLoRA base model storage |

### 5.3 FlashAttention

Standard attention materializes the full n×n attention matrix in HBM — O(n²) memory.

FlashAttention never materializes the full attention matrix:
1. Tile Q, K, V into blocks that fit in SRAM
2. Compute attention block-by-block using online softmax normalization
3. Accumulate output without writing intermediate matrices to HBM

Result: O(n²/B) HBM accesses vs O(n²) for naive — 2-4× speedup, O(n) memory.

### 5.4 Paged Attention (vLLM)

KV cache is fragmented across GPU memory similarly to virtual memory paging in OS. Eliminates KV cache memory waste. Enables 2-4× higher throughput.

</details>

---

<details>
<summary><strong>MODULE 6 — Reasoning & Scaling (click to expand)</strong></summary>

### 6.1 Chinchilla Scaling Laws

```
N_opt ∝ C^0.5
D_opt ∝ C^0.5
N_opt ≈ D_opt / 20 (tokens per parameter)
```

### 6.2 Chain-of-Thought (CoT)

**Zero-shot CoT:** Append "Let's think step by step."

**Self-Consistency:** Sample k reasoning paths, take majority vote on the final answer. Improves accuracy by ~10% on MATH and GSM8K.

### 6.3 RL-Based Reasoning

#### GRPO (Group Relative Policy Optimization)

```
Aᵢ = (rᵢ - mean(r)) / std(r)
```

No critic/value model needed — variance reduction via group normalization.

#### Process Reward Models (PRM)

PRMs substantially outperform outcome reward models on challenging math problems — the reasoning process matters, not just the answer.

</details>

---

<details>
<summary><strong>MODULE 7 — Agentic Workflows (click to expand)</strong></summary>

### 7.1 RAG Deep Dive

```
Query → Embed → ANN Search → Retrieve top-k chunks → Augment prompt → Generate
```

#### Hybrid Search

```
RRF_score(d) = Σ_r 1 / (k + rank_r(d))
```

where k=60 is a smoothing constant. Consistently outperforms either retrieval method alone.

#### Advanced RAG Patterns

| Pattern | Description |
|---------|-------------|
| HyDE | Generate a hypothetical answer, embed it, retrieve similar documents |
| Query rewriting | LLM reformulates ambiguous queries before retrieval |
| Multi-hop retrieval | Iteratively retrieve and reason |
| Self-RAG | Model decides when to retrieve and reflects on retrieved results |
| RAPTOR | Hierarchical document clustering + summarization |

### 7.2 Tool Calling

```json
{
  "name": "search_web",
  "arguments": {
    "query": "current GDP of Germany 2024",
    "num_results": 5
  }
}
```

### 7.3 ReAct Agent Pattern

```
Thought: I need to find the population of Tokyo.
Action: search_web(query="Tokyo population 2024")
Observation: 13.96 million (city proper), 37.4 million (greater area)
Thought: Now I can answer.
Answer: Tokyo's city proper population is approximately 14 million.
```

### 7.4 Agent Memory Systems

| Memory Type | Implementation | Use Case |
|-------------|---------------|----------|
| In-context | Messages in context window | Short-term working memory |
| External (episodic) | Vector store with retrieved memories | Long-term user/session state |
| External (semantic) | Knowledge graph or structured DB | Factual knowledge retrieval |
| Procedural | Finetuning / system prompt | Behavioral tendencies |

### 7.5 Key Challenges

| Challenge | Description |
|-----------|-------------|
| Error propagation | Early mistakes cascade through multi-step plans |
| Context management | Long agent traces exhaust context windows |
| Tool reliability | External APIs fail; model must handle gracefully |
| Safety | Agents with real-world effects require careful permission scoping |

</details>

---

## Quick Reference: Architecture Choices by Model Family

| Model | Arch | Norm | PE | Attention | FFN | MoE |
|-------|------|------|----|-----------|-----|-----|
| GPT-2 | Decoder | LayerNorm | Learned abs | MHA | GeLU | No |
| LLaMA-3 | Decoder | RMSNorm | RoPE | GQA | SwiGLU | No |
| Mistral-7B | Decoder | RMSNorm | RoPE | GQA + sliding window | SwiGLU | No |
| Mixtral 8×7B | Decoder | RMSNorm | RoPE | GQA | SwiGLU | Yes (8 experts, top-2) |
| DeepSeek-V3 | Decoder | RMSNorm | RoPE | MLA | SwiGLU | Yes (256 experts, top-8) |
| Gemma-2 | Decoder | RMSNorm | RoPE | GQA + logit softcap | GeGLU | No |

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>If you find this resource helpful, please consider giving it a ⭐!</strong>
</p>

---

> This strategic analysis was curated and prompt-engineered by  
> Hande Gabrali-Knobloch  
> Powered by NotebookLM — based on the provided texts.

---

<!-- ===== FOOTER BANNER (Capsule Render) ===== -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1b27,100:4a6cf7&height=120&section=footer" alt="Footer" width="100%" />
</p>
