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

\`\`\`python

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

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**If you find this resource helpful, please consider giving it a ⭐!**
