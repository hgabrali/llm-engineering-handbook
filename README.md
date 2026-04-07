# LLM Engineering Handbook

> A comprehensive, practitioner-oriented reference on the core building blocks of modern Large Language Models — from low-level transformer mechanics to production-grade agentic systems.

---

## Table of Contents

1. [How Transformers Actually Work](#1-how-transformers-actually-work)
2. [Decoding Strategies & Mixture-of-Experts](#2-decoding-strategies--mixture-of-experts)
3. [LLM Finetuning](#3-llm-finetuning)
4. [Evaluation Techniques](#4-evaluation-techniques)
5. [Optimization Tricks](#5-optimization-tricks)
6. [Reasoning & Scaling](#6-reasoning--scaling)
7. [Agentic Workflows](#7-agentic-workflows)
8. [References & Further Reading](#8-references--further-reading)

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
- **Special tokens:** \`[BOS]\`, \`[EOS]\`, \`[PAD]\`, \`[UNK]\`, \`[MASK]\` — each carries model-specific semantics.
- **Pre-tokenization:** Whitespace splitting, regex patterns applied before BPE.



# Example: tiktoken (OpenAI)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Transformers are powerful.")
print(tokens)
\`\`\`

### 1.2 Embeddings

Tokens are projected into a continuous vector space through a learned embedding matrix **E ∈ R^{V x d}**, where V is the vocabulary size and d is the model dimension.

**Types of embeddings in a transformer:**

- **Token embeddings:** Lookup table mapping token IDs to dense vectors.
- **Positional embeddings:** Encode the order of tokens in a sequence. Originally sinusoidal (Vaswani et al., 2017); modern models prefer **Rotary Position Embeddings (RoPE)** or **ALiBi**.
- **Segment embeddings (encoder models):** Distinguish sentence A from sentence B in tasks like NLI.

### 1.3 Self-Attention

Self-attention is the mechanism that allows every token to attend to every other token in the sequence.

\`\`\`
Q = X * W_Q    (queries)
K = X * W_K    (keys)
V = X * W_V    (values)

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
\`\`\`

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

LoRA freezes W and adds a low-rank decomposition: W' = W + (alpha/r) * B @ A

| Variant | Key Difference |
|---------|----------------|
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

Query -> Embed -> Retrieve (Vector DB) -> Augment Prompt -> Generate

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

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**If you find this resource helpful, please consider giving it a ⭐!**


---

# LLM Deep Dive: Expert Technical Reference

> A rigorous, sequential treatment of modern large language model architecture, training, and deployment.

---

## MODULE 1 — How Transformers Actually Work

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
|---|---|
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
A learned matrix. Each row is a d-dimensional vector. These are trained end-to-end; geometrically, semantically similar tokens cluster (though this is less interpretable than word2vec because context is handled by attention, not co-occurrence statistics alone).

#### Positional Encodings
Transformers have no inherent notion of sequence order — attention is permutation-equivariant. Positional encodings inject order information.

**Sinusoidal (Vaswani et al. 2017):**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

These are fixed, not learned. The key property: relative position differences are representable as a linear transformation, which is theoretically elegant but practically outperformed by learned alternatives.

**Learned absolute positional embeddings:**
GPT-2, early BERT. Simply a second embedding matrix of size (max_seq_len × d_model). Problem: hard cap on sequence length; cannot generalize beyond training length.

**RoPE (Rotary Position Embedding):**
See Module 5.1. Currently the dominant approach.

**ALiBi:**
Adds a linear position bias directly to attention logits before softmax: `bias(i,j) = -m · |i - j|` where m is a per-head slope. Extremely simple. Allows length extrapolation beyond training context but degrades smoothly rather than catastrophically.

### 1.3 The Attention Mechanism

#### Scaled Dot-Product Attention

Given input matrix X ∈ ℝ^(n×d_model), three projections produce Q, K, V:

```
Q = X W_Q,   W_Q ∈ ℝ^(d_model × d_k)
K = X W_K,   W_K ∈ ℝ^(d_model × d_k)
V = X W_V,   W_V ∈ ℝ^(d_model × d_v)
```

Attention output:

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

The √d_k scaling prevents softmax saturation in high dimensions (without it, dot products grow in magnitude with d_k, pushing softmax into near-one-hot distributions with vanishing gradients).

#### Multi-Head Attention (MHA)

Run h parallel attention heads with separate projections, concatenate, then project:

```
head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
MHA(X) = Concat(head_1, ..., head_h) W_O
```

Each head can attend to different positional or semantic relationships. Empirically, heads specialize: some track syntactic dependencies, others coreference, etc.

#### Complexity Analysis

| Operation | Time | Space |
|---|---|---|
| Attention (naive) | O(n² d) | O(n²) |
| FFN | O(n d²) | O(d²) |

The O(n²) attention cost is the core scalability bottleneck — the subject of most efficient-attention research (FlashAttention, Sparse Attention, Linear Attention).

#### Causal (Autoregressive) Masking
For decoder-only models (GPT family), position i cannot attend to positions j > i. This is enforced by adding −∞ to the upper triangle of the QKᵀ matrix before softmax, producing zero attention weight after exponentiation.

#### KV Cache
During inference, previously computed K and V matrices are cached and reused. Only the new token's Q is computed per step. This reduces per-step cost from O(n²) to O(n·d) but introduces memory pressure proportional to `batch_size × n_layers × 2 × n_heads × head_dim × seq_len`.

#### Grouped Query Attention (GQA)
A memory-efficient variant: instead of one KV pair per head, multiple Q heads share a single KV pair. LLaMA-2 70B uses GQA. Reduces KV cache size by a factor of n_groups while maintaining most of MHA's expressive capacity.

### 1.4 The Transformer Block

A full transformer block (decoder-only, pre-norm variant used by modern LLMs):

```
x = x + Attention(RMSNorm(x))     # residual + self-attention
x = x + FFN(RMSNorm(x))           # residual + feed-forward
```

#### RMSNorm vs LayerNorm
`RMSNorm(x) = x / RMS(x) · γ`
where `RMS(x) = √(mean(x²))`. Faster than LayerNorm (no mean subtraction), numerically stable. Used by LLaMA, Mistral, Gemma.

#### FFN Block
**Standard:**
`FFN(x) = max(0, x W_1 + b_1) W_2 + b_2`

**SwiGLU (used by LLaMA, PaLM):**
`FFN(x) = (SiLU(x W_1) ⊙ x W_2) W_3`
where `SiLU(x) = x · σ(x)`. This gated architecture provides richer expressivity at similar parameter count and is empirically superior.

The FFN typically has a 4× expansion: `d_ff = 4 × d_model` (or ~8/3 × d_model for SwiGLU to maintain parameter parity).

---

## MODULE 2 — Decoding Strategies & Mixture of Experts

### 2.1 Decoding Strategies

At inference, the model produces a probability distribution over the vocabulary at each step. The decoding strategy determines how to sample from or optimize over this distribution.

#### Greedy Decoding
Always pick argmax P(token | context). Deterministic, fast, but produces repetitive and often degenerate output. Vulnerable to "probability mass traps."

#### Beam Search
Maintain k candidate sequences (beams) at each step, expand each by all vocabulary tokens, keep top-k by cumulative log-probability.

```
score(sequence) = Σ log P(tᵢ | t₁...tᵢ₋₁) / length_penalty
```

Length penalty (α typically 0.6–1.0) prevents beam search from favoring short sequences. Beam search is the standard for translation/summarization but produces overly conservative/generic text for open-ended generation.

#### Sampling
Draw from P(·) directly. Introduces stochasticity, diversity, but risks low-probability "hallucination" tokens.

#### Temperature Scaling

```
P_T(tᵢ) = softmax(logits / T)
```

- **T < 1:** sharper distribution, more deterministic
- **T > 1:** flatter distribution, more creative/chaotic
- **T → 0:** degenerates to greedy

#### Top-k Sampling
Restrict sampling to the k highest-probability tokens. Problem: k is context-insensitive — sometimes the top-k should be 1 (unambiguous next token), sometimes 1000 (creative continuation).

#### Top-p (Nucleus) Sampling
Sample from the smallest set of tokens whose cumulative probability ≥ p:
`V_p = min S such that Σ_{t∈S} P(t) ≥ p`
Then renormalize and sample. Adapts dynamically to the distribution's shape. p=0.9 or 0.95 is typical.

#### Min-p Sampling
Prunes tokens with `P(t) < min_p × max(P(·))`. Relative threshold — preserves the distribution's dynamic range better than top-p in high-entropy situations.

#### Repetition Penalty

```
logit'(t) = logit(t) / penalty   if t appeared in context
           = logit(t)             otherwise
```

Prevents degenerate repetition loops.

#### Speculative Decoding
A critical inference acceleration technique:

1. Run a small draft model M_q to generate k tokens speculatively
2. Run the large target model M_p in a single forward pass to score all k tokens
3. Accept tokens where M_q's distribution is "close enough" to M_p's; reject and resample otherwise

Achieves ~2-3× speedup for large models. Theoretically lossless (provably samples from M_p's distribution). Requires draft and target models to share vocabulary.

### 2.2 Mixture of Experts (MoE)

MoE is a conditional computation architecture: only a subset of model parameters are active per input token.

#### Architecture

Replace the dense FFN in each transformer block with a mixture:

```
MoE(x) = Σᵢ G(x)ᵢ · Eᵢ(x)
```

where:
- **Eᵢ** = expert FFN i (there are N total experts, typically 8–128)
- **G(x)** = gating network output (sparse vector)

#### Top-k Gating

```
G(x) = Softmax(TopK(x · W_gate, k))
```

Only k experts (typically k=2) are activated per token. The other N-k experts receive zero weight and are not computed.

#### Why This Matters

- **Total parameters:** N × (FFN params) — can be huge
- **Active parameters per token:** k × (FFN params) — constant compute
- **Mixtral 8×7B:** 46.7B total parameters, ~12.9B active per token → inference cost ≈ 7B dense model

#### Expert Load Balancing
A critical training problem: without regularization, the router collapses (always sends tokens to the same 1–2 experts, "expert collapse").

Auxiliary load balancing loss:

```
L_aux = α × N × Σᵢ fᵢ · pᵢ
```

where fᵢ = fraction of tokens routed to expert i, pᵢ = mean routing probability to expert i. This encourages uniform routing.

#### Expert Specialization
Empirically, experts develop loose specialization: some handle syntactic patterns, some domain-specific content, some particular languages. This is emergent — not explicitly trained.

#### Distributed Inference Challenge
In distributed serving, each expert typically lives on a different device. Token routing requires all-to-all communication across devices — a significant networking overhead. Expert parallelism + expert affinity routing are active research areas.

#### DeepSeek-V3's MoE Innovations
DeepSeek-V3 uses a "fine-grained MoE" with 256 experts (top-8 routing), auxiliary-loss-free load balancing via expert bias terms, and a multi-token prediction auxiliary head. This achieves state-of-the-art at dramatically reduced training cost.

---

## MODULE 3 — LLM Finetuning

### 3.1 Supervised Finetuning (SFT)

Standard finetuning: continue training on labeled (prompt, completion) pairs with cross-entropy loss on the completion tokens only (prompt tokens are masked).

```
L_SFT = -Σₜ log P_θ(yₜ | x, y₁...yₜ₋₁)
```

#### Data Quality >> Quantity
The LIMA paper (2023) demonstrated that 1,000 carefully curated examples can match the instruction-following quality of models trained on 50k+ examples. Distribution matters enormously — diversity of task types, correct formatting, consistent persona.

#### Full Finetuning Challenges

- **Gradient updates to all parameters:** memory = 16–20 bytes/param (weights + gradients + optimizer states for AdamW)
- **Catastrophic forgetting:** finetuning on narrow domain degrades general capabilities
- **Overfitting:** small datasets → multiple epochs risk memorization

### 3.2 Parameter-Efficient Finetuning (PEFT) — LoRA

#### Low-Rank Adaptation (LoRA)
Key insight: the update to pretrained weights during finetuning has low intrinsic rank.

Instead of learning ΔW ∈ ℝ^(d×k) directly, decompose:

```
ΔW = B · A,   where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
```

During finetuning, W₀ is frozen. Only A and B are trained.

**Initialization:**
A ~ N(0, σ²), B = 0 → ΔW = 0 at training start (training stability)

**Forward pass:**

```
h = W₀x + (B·A)x · (α/r)
```

α is a scaling hyperparameter (often set equal to r, making α/r = 1).

**Parameter reduction:**
Standard: updating attention matrices (Q, K, V, O). For d=4096, r=16:

- ΔW full: 4096 × 4096 = 16.7M params
- LoRA: (4096×16) + (16×4096) = 131K params → **128× reduction**

#### Where to Apply LoRA
Originally: Q and V projections only. Subsequent work (DoRA, LLaMA-Adapter) shows applying to all linear layers (Q, K, V, O, up_proj, down_proj, gate_proj) improves performance.

#### QLoRA
Combine 4-bit NF4 quantization of the frozen base model with LoRA adapters in bf16. Enables finetuning a 65B model on a single 48GB GPU.

- **NF4 (NormalFloat4):** a quantization scheme optimal for normally distributed weights
- **Double quantization:** quantize the quantization constants themselves
- **Paged optimizers:** use CPU RAM to handle GPU memory spikes during backward pass

#### LoRA Variants

| Variant | Key Idea |
|---|---|
| DoRA | Decomposes weight into magnitude + direction; trains direction via LoRA |
| LoRA+ | Different learning rates for A and B matrices (B benefits from higher LR) |
| AdaLoRA | Adaptively allocates rank budget across weight matrices via SVD |
| LoRA-FA | Freezes A, only trains B → reduces activation memory |

### 3.3 RLHF (Reinforcement Learning from Human Feedback)

RLHF is the pipeline that transforms a pretrained/SFT model into a helpful, harmless assistant. Three stages:

#### Stage 1: Supervised Finetuning
Already covered. Produces an SFT model capable of following instructions.

#### Stage 2: Reward Model Training

Collect human preference data: for each prompt x, show humans two completions (yᵢ, yⱼ), collect preference label yᵢ ≻ yⱼ.

Train a reward model R_φ (typically same architecture as LLM but with a scalar head) to predict human preferences via Bradley-Terry model:

```
L_RM = -E[(x,yw,yl)] [log σ(R_φ(x, yw) - R_φ(x, yl))]
```

where yw = preferred completion, yl = dispreferred.

#### Stage 3: RL Policy Optimization (PPO)

Maximize reward while maintaining proximity to the reference SFT policy:

```
L_PPO(θ) = E[R_φ(x, y)] - β · KL[π_θ(y|x) || π_ref(y|x)]
```

The KL penalty β prevents reward hacking (the policy finding adversarial completions that fool the reward model while degrading quality).

PPO (Proximal Policy Optimization) clips the probability ratio to prevent large policy updates:

```
L_CLIP = E[min(rₜ Aₜ, clip(rₜ, 1-ε, 1+ε) Aₜ)]
```

where `rₜ = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)`, `Aₜ = advantage estimate`.

#### Computational Cost of PPO
4 models must reside in memory simultaneously:

1. **Policy model π_θ** (trained)
2. **Reference model π_ref** (frozen SFT)
3. **Reward model R_φ**
4. **Value model V_ψ** (critic)

This is extremely memory-intensive. The DeepSpeed-Chat / TRL frameworks implement various parallelism strategies to make this feasible.

#### DPO (Direct Preference Optimization)
DPO bypasses the RL loop entirely by deriving an analytical relationship between the reward function and the optimal policy:

```
L_DPO(θ) = -E[(x,yw,yl)] [log σ(β log(π_θ(yw|x)/π_ref(yw|x)) - β log(π_θ(yl|x)/π_ref(yl|x)))]
```

This is equivalent to RLHF under the Bradley-Terry preference model but requires only a frozen reference model and the LLM — no reward model, no PPO. Dramatically simpler, now dominant in production.

#### DPO Variants

| Variant | Improvement |
|---|---|
| IPO | Avoids reward overfitting with identity loss |
| KTO | Uses per-sample binary signals (good/bad) instead of pairs |
| SimPO | Removes reference model dependency entirely |
| ORPO | Combines SFT and preference loss into single stage |

---

## MODULE 4 — Evaluation Techniques

### 4.1 Classical Benchmarks

- **MMLU (Massive Multitask Language Understanding):** 57 academic subjects, 4-choice MCQ. Measures breadth of world knowledge.
- **HellaSwag:** Commonsense NLI — choose most plausible sentence continuation. Tests grounded understanding.
- **HumanEval / MBPP:** Code generation evaluated by unit test execution. Ground truth via functional correctness.
- **MATH:** Competition mathematics requiring multi-step symbolic reasoning.
- **GSM8K:** Grade school math word problems. Tests arithmetic reasoning and instruction following.

#### Benchmark Contamination
A critical issue: training data often contains benchmark test sets. Models can "memorize" answers without reasoning capability. Mitigation: n-gram deduplication against benchmarks, held-out evaluation sets, dynamic benchmarks.

### 4.2 LLM-as-a-Judge

Human evaluation is expensive, slow, and inconsistent. LLM-as-a-judge uses a powerful model (typically GPT-4 or Claude) to evaluate outputs along specified dimensions.

#### Single-Answer Grading
Prompt the judge to rate a response on a 1–10 scale for helpfulness, accuracy, coherence, etc. Provide a detailed rubric.

#### Pairwise Comparison
Present two responses A and B to the judge, ask which is better. More reliable than absolute scoring; humans and LLM judges both show higher inter-rater agreement on pairwise tasks.

Prompt template:
```
System: You are an expert evaluator...
User:
[PROMPT]: {question}
[RESPONSE A]: {response_a}
[RESPONSE B]: {response_b}
Which response is better? Explain your reasoning, then output "A" or "B".
```

#### MT-Bench
Designed by Zheng et al. (2023). 80 multi-turn questions across 8 categories (reasoning, math, coding, roleplay, extraction, STEM, humanities, writing). Uses GPT-4 as judge. Scores correlate well with human preferences (Pearson r ≈ 0.93).

#### Chatbot Arena (LMSYS)
Crowdsourced ELO rating system. Users chat with two anonymous models, select the winner. >1M human votes. The most reliable ranking system for overall user preference — though it biases toward engaging/verbose responses.

#### Failure Modes of LLM-as-a-Judge

| Bias | Description |
|---|---|
| Positional bias | Prefers whichever response appears first |
| Verbosity bias | Longer responses rated higher regardless of quality |
| Self-enhancement bias | GPT-4 prefers GPT-4 outputs |
| Sycophancy | Agrees with human's stated preference even when wrong |
| Calibration failure | Confidently wrong on factual accuracy assessment |

**Mitigations:**

- Swap A/B positions and average results
- Use chain-of-thought reasoning in judge prompts
- Provide explicit, rubric-grounded scoring criteria
- Use ensemble of multiple judge models

#### G-Eval Framework
Probability-weighted scoring: instead of greedy decoding, compute the expected score as a weighted sum over the model's token probabilities for score digits:

```
Score = Σᵢ P("i" | prompt) × i
```

More calibrated than argmax decoding for numeric judgments.

---

## MODULE 5 — Optimization Tricks

### 5.1 RoPE (Rotary Position Embedding)

RoPE encodes position by rotating query and key vectors before the attention dot product. The rotation matrix for position m in 2D:

```
R(m, θ) = [[cos(mθ), -sin(mθ)],
            [sin(mθ),  cos(mθ)]]
```

Applied to each pair of dimensions with frequency θᵢ = 10000^(-2i/d):

```
q_m = R(m) q,   k_n = R(n) k
```

#### Why This Works
The dot product qₘᵀ kₙ depends only on the relative position (m - n):

```
(R(m)q)ᵀ (R(n)k) = qᵀ R(m-n)ᵀ k
```

This means attention patterns naturally become a function of relative distance — a theoretically motivated inductive bias.

#### Context Length Extension with RoPE
The base θ = 10000 was designed for shorter contexts. To extend:

**Positional Interpolation (PI):**
Scale positions down: `m' = m × (original_length / target_length)`. Allows 32k+ context with minimal finetuning.

**YaRN (Yet Another RoPE extensioN):**
Applies different scaling to different frequency bands of the RoPE dimensions — high-frequency components (local dependencies) are less scaled than low-frequency (long-range). Better perplexity on long sequences.

**LongRoPE:**
Searches for non-uniform scaling factors per frequency component using evolutionary optimization. Used by Phi-3-mini to extend to 128k context.

### 5.2 Quantization

Quantization reduces numerical precision to decrease model size and accelerate inference.

#### Post-Training Quantization (PTQ)
No retraining required. Calibrate scale factors using a small dataset.

**Round-to-nearest (RTN):** Simply round weights to nearest quantized value. Fast, ~1% accuracy drop at 8-bit, significant degradation at 4-bit.

**GPTQ:** Layer-wise quantization using second-order information (approximate Hessian). Solves the quantization error minimization problem:

```
min_Q ||WX - QX||²_F
```

Column-wise quantization with lazy batch updates. Achieves near-fp16 quality at 4-bit for large models.

**AWQ (Activation-Aware Weight Quantization):**
Observes that ~1% of weight channels correspond to activation outliers and disproportionately impact quantization error. Protects these channels by scaling before quantization. Outperforms GPTQ on most benchmarks.

#### Quantization-Aware Training (QAT)
Simulate quantization during forward pass, use straight-through estimator (STE) for gradients through the non-differentiable rounding operation. More accurate but requires full training run.

#### KV Cache Quantization
KV cache is a major memory bottleneck at long contexts. Quantizing KV cache to INT8 or INT4 (with outlier handling) can halve memory without significant quality loss. Used in production by vLLM, TensorRT-LLM.

#### Data Types Summary

| Dtype | Bits | Range | Use Case |
|---|---|---|---|
| FP32 | 32 | ±3.4e38 | Reference, gradient accumulation |
| BF16 | 16 | ±3.4e38 | Training (same range as FP32) |
| FP16 | 16 | ±65504 | Training, some inference |
| INT8 | 8 | -128 to 127 | Inference, activations |
| INT4 | 4 | -8 to 7 | Weight-only quantization |
| NF4 | 4 | (non-linear) | QLoRA base model storage |

### 5.3 FlashAttention

Standard attention materializes the full n×n attention matrix in HBM (GPU high-bandwidth memory) — O(n²) memory, slow due to memory bandwidth bottleneck.

FlashAttention (Dao et al., 2022) never materializes the full attention matrix:

1. Tile Q, K, V into blocks that fit in SRAM (on-chip, fast)
2. Compute attention block-by-block using online softmax normalization
3. Accumulate output without writing intermediate matrices to HBM

Result: O(n²/B) HBM accesses vs O(n²) for naive — 2-4× speedup, O(n) memory (vs O(n²)).

FlashAttention-2 adds better parallelism over sequence dimension and reduces non-matmul FLOPs. FlashAttention-3 (for H100) leverages asynchronous execution and FP8 precision.

### 5.4 Other Approximations

#### Sparse Attention
Instead of full n×n attention, restrict each token to a structured subset of positions:

- **Local windowed attention:** each token attends only to w neighboring tokens → O(nw) complexity
- **Global tokens (Longformer):** some tokens attend globally, others locally
- **Strided/dilated patterns (BigBird):** combines local, global, and random attention

#### Linear Attention
Approximate softmax attention via kernel trick:

```
Attention(Q,K,V) ≈ φ(Q) (φ(K)ᵀ V)
```

The key insight: compute `(φ(K)ᵀ V)` first — O(nd²) instead of O(n²d). But the feature map φ must approximate exp(·) — getting this right is the research challenge.

#### Paged Attention (vLLM)
KV cache is fragmented across GPU memory similarly to virtual memory paging in OS. Eliminates KV cache memory waste from padding and internal fragmentation. Enables dynamic batching and sharing prefixes across requests. The core innovation of vLLM, enabling 2-4× higher throughput vs naive serving.

---

## MODULE 6 — Reasoning & Scaling

### 6.1 Scaling Laws

#### Chinchilla Scaling Laws (Hoffmann et al., 2022)
For a given compute budget C (FLOPs), the optimal model size N and training tokens D follow:

```
N_opt ∝ C^0.5
D_opt ∝ C^0.5
```

Empirically: `N_opt ≈ D_opt / 20` (tokens per parameter).

This showed that GPT-3 (175B params, ~300B tokens) was significantly undertrained relative to compute-optimal. Chinchilla (70B, 1.4T tokens) matched GPT-3 performance at 4× fewer parameters.

#### Inference-Time Scaling (Test-Time Compute)
A newer paradigm: allocating more compute at inference rather than training improves quality on hard tasks.

**Methods:**

- **Best-of-N:** sample N completions, select highest reward → reward model required
- **Self-consistency:** sample multiple chain-of-thought solutions, take majority vote
- **Process reward models (PRM):** reward each reasoning step, not just final answer → enables beam search over reasoning traces
- **Monte Carlo Tree Search (MCTS):** systematic exploration of reasoning branches

OpenAI o1 / o3 / DeepSeek-R1 use inference-time scaling via extended chain-of-thought generation with RL training to make the reasoning process itself learnable.

### 6.2 Chain-of-Thought (CoT)

CoT prompting (Wei et al., 2022) elicits step-by-step reasoning by including worked examples in the prompt. Dramatically improves performance on multi-step reasoning tasks.

**Zero-shot CoT:** Append "Let's think step by step." to any prompt. Surprisingly effective.

**Emergent property:** CoT only helps models above ~100B parameters (with few-shot prompting). Smaller models produce fluent but incorrect reasoning chains.

#### Self-Consistency (Wang et al., 2023):
Sample k reasoning paths, take the majority vote on the final answer. Improves accuracy by ~10% on MATH and GSM8K over greedy CoT.

### 6.3 RL-Based Reasoning

#### RLHF for Reasoning (GRPO / PPO on outcome reward)
For math/code tasks, outcome verification is automatic (answer correct or test passes). Use binary reward without a reward model:

**GRPO (Group Relative Policy Optimization, DeepSeek-R1):**
For each prompt, sample G outputs. Normalize rewards within the group:

```
Aᵢ = (rᵢ - mean(r)) / std(r)
```

No critic/value model needed — variance reduction via group normalization. Simpler and more stable than PPO for reasoning tasks.

#### Process Reward Models (PRM)
Label each intermediate reasoning step as correct/incorrect. Train a model to predict step-level correctness. Use PRM to:

- Rank full solutions via product of step probabilities
- Guide beam search over reasoning traces at inference time

**Key finding (Lightman et al., 2023):** PRMs substantially outperform outcome reward models on challenging math problems — the reasoning process matters, not just the answer.

---

## MODULE 7 — Agentic Workflows

### 7.1 Retrieval-Augmented Generation (RAG)

RAG grounds LLM outputs in external knowledge, addressing hallucination and knowledge cutoff limitations.

#### Naive RAG Pipeline

```
Query → Embed → ANN Search → Retrieve top-k chunks → Augment prompt → Generate
```

#### Embedding Models
Documents and queries are embedded into a shared dense vector space. Typical dimensions: 768 (BERT-class) to 4096 (E5-mistral). Training objective: contrastive learning — bring query-document pairs close, push negatives apart.

**ColBERT:** late interaction — embed query and document tokens separately, compute MaxSim scores at retrieval time. More expressive than single-vector embedding, more expensive.

#### Chunking Strategy
Critical and often underappreciated. Options:

- Fixed-size (e.g., 512 tokens with overlap)
- Semantic/sentence-level splitting
- Document-structure-aware (headers, sections)
- Small-to-big: retrieve small chunks, return parent document sections to the LLM

#### Hybrid Search
Combine dense (embedding) and sparse (BM25) retrieval, merge results via Reciprocal Rank Fusion (RRF):

```
RRF_score(d) = Σ_r 1 / (k + rank_r(d))
```

where k=60 is a smoothing constant. Consistently outperforms either retrieval method alone.

#### Reranking
Apply a cross-encoder (query + document → relevance score) to the top-N dense/BM25 results to select top-k for the LLM. Cross-encoders are slower (can't precompute document embeddings) but far more accurate.

#### Advanced RAG Patterns

| Pattern | Description |
|---|---|
| HyDE | Generate a hypothetical answer, embed it, retrieve similar documents |
| Query rewriting | LLM reformulates ambiguous queries before retrieval |
| Multi-hop retrieval | Iteratively retrieve and reason; use intermediate answers to form new queries |
| Self-RAG | Model decides when to retrieve (via special tokens) and reflects on retrieved results |
| RAPTOR | Hierarchical document clustering + summarization; retrieval at multiple abstraction levels |

#### RAG Failure Modes

- **Retrieval failure:** correct documents not retrieved (recall problem)
- **Context utilization failure:** documents retrieved but model ignores them (attention dilution)
- **Conflicting contexts:** retrieved chunks contradict each other or the model's parametric knowledge
- **Lost in the middle:** LLMs attend poorly to information in the middle of long contexts

### 7.2 Tool Calling (Function Calling)

Tool calling allows LLMs to invoke external APIs, execute code, query databases, and interact with environments — extending capabilities beyond what language modeling can represent.

#### Mechanism
The model is trained/prompted to output structured JSON describing a function call instead of a natural language answer:

```json
{
  "name": "search_web",
  "arguments": {
    "query": "current GDP of Germany 2024",
    "num_results": 5
  }
}
```

The host application executes the function, returns results, and continues the conversation with the tool result as context.

#### Training for Tool Use
Models are finetuned on datasets of tool-augmented conversations. The chat template includes special tokens delineating tool definitions, tool calls, and tool results. OpenAI function calling, Anthropic tool use, and LLaMA-3's tool calling format all follow variations of this pattern.

#### Tool Schemas
Tools are described in JSON Schema format in the system prompt. The model must learn to respect type constraints, required fields, and valid enum values — essentially learning to produce schema-valid JSON on demand.

### 7.3 Agentic Architectures

An agent is a system where an LLM acts iteratively, with access to tools and memory, to complete a goal.

#### ReAct (Reason + Act)
Alternates between:

- **Thought:** explicit reasoning about current state and next action
- **Action:** tool call with parameters
- **Observation:** tool result

```
Thought: I need to find the population of Tokyo.
Action: search_web(query="Tokyo population 2024")
Observation: 13.96 million (city proper), 37.4 million (greater area)
Thought: Now I can answer.
Answer: Tokyo's city proper population is approximately 14 million.
```

#### Agent Memory Systems

| Memory Type | Implementation | Use Case |
|---|---|---|
| In-context | Messages in context window | Short-term working memory |
| External (episodic) | Vector store with retrieved memories | Long-term user/session state |
| External (semantic) | Knowledge graph or structured DB | Factual knowledge retrieval |
| Procedural | Finetuning / system prompt | Behavioral tendencies |

#### Multi-Agent Systems
Multiple LLM agents interact to solve complex tasks:

- **Orchestrator-subagent:** one agent plans and delegates subtasks to specialized agents
- **Debate:** multiple agents argue positions, resolve disagreements
- **Society of Mind:** large pool of agents, each with narrow capability

Frameworks: LangGraph, AutoGen, CrewAI.

#### Key Challenges in Agentic Systems

| Challenge | Description |
|---|---|
| Error propagation | Early mistakes cascade through multi-step plans |
| Context management | Long agent traces exhaust context windows |
| Tool reliability | External APIs fail; model must handle gracefully |
| Reward hacking | Agent finds shortcuts that satisfy literal goal but not intent |
| Safety | Agents with real-world effects require careful permission scoping |

#### Structured Output Enforcement
Constrained decoding (via grammar-guided sampling) ensures tool calls are always syntactically valid JSON. Outlines and llama.cpp implement this via pushdown automata over the token generation process — only tokens that extend a valid partial parse are allowed.

---

## Quick Reference: Architecture Choices by Model Family

| Model | Arch | Norm | PE | Attention | FFN | MoE |
|---|---|---|---|---|---|---|
| GPT-2 | Decoder | LayerNorm | Learned abs | MHA | GeLU | No |
| LLaMA-3 | Decoder | RMSNorm | RoPE | GQA | SwiGLU | No |
| Mistral-7B | Decoder | RMSNorm | RoPE | GQA + sliding window | SwiGLU | No |
| Mixtral 8×7B | Decoder | RMSNorm | RoPE | GQA | SwiGLU | Yes (8 experts, top-2) |
| DeepSeek-V3 | Decoder | RMSNorm | RoPE | MLA | SwiGLU | Yes (256 experts, top-8) |
| Gemma-2 | Decoder | RMSNorm | RoPE | GQA + logit softcap | GeGLU | No |

---

*Document generated for advanced ML engineering study. All formulations reflect current literature as of 2024–2025.*
