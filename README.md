# Empirical Benchmark: TypeSafe AI Jev System One vs. Specialized Sovereign RAG Architecture

> **Author:** Abdiel Memon ([Memon Systems Architecture](https://memonsystems.com))  
> **Evaluation Standard:** EU AI Act Article 15 (Accuracy & Deterministic Robustness)  
> **Target Statute:** UK Primary Legislation (*Companies Act 2006*, *Employment Rights Act 1996*, *Insolvency Act 1986*)  
> **Hardware Target:** Apple Silicon (MPS Unified Memory) / Linux NVIDIA CUDA  
> **Provider:** OpenRouter (`POST /api/alpha/decisions` — `typesafe/jev-1.13`)

---

## 1. Executive Summary & Thesis

In this empirical benchmark suite, I evaluate TypeSafe AI's **Jev System One** (`typesafe/jev-1.13` via OpenRouter's Decisions API) against my **Specialized Sovereign RAG Architecture** across three pipeline nodes:
1. **Node 1: Intent Routing & Triage** (Jev `Choice` vs. My DistilBERT classifier + statutory regex/NER coordinate extraction)
2. **Node 2: Candidate Passage Reranking** (Jev Pairwise `Noul` vs. My ColBERT-v2 late-interaction $MaxSim$ + Sub-chunk span localization)
3. **Node 3: Factual Verification & Legal NLI Sentinel** (Jev `Choice` vs. My Co-hosted `DeBERTa-v3` with epistemic abstention gates)

TypeSafe AI’s premise correctly recognizes that open-ended LLM agent `while` loops thrash and collapse in production—a core design principle I have demonstrated in my legal RAG research. However, TypeSafe's proposed remedy—substituting agent loops with an external, cloud-hosted generalist decision API—introduces fatal operational and regulatory liabilities when deployed inside high-liability enterprise pipelines.

```
[ User Query ]
       │
       ▼
┌──────────────────────────────────────┐
│  NODE 1: Intent Routing & Triage     │ <── Jev ('Choice', ~700ms HTTP) vs. Sovereign (<1ms, +NER Coordinates)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Fast Retrieval Shortlist (BM25)     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  NODE 2: Passage Reranking           │ <── Jev Pairwise ($48k/mo token trap) vs. ColBERT-v2 (<5ms, 50-Word Span)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Generation Tier (Llama 3 Stream)    │
└──────────────────┬───────────────────┘
                   │ (Sentence stream)
                   ▼
┌──────────────────────────────────────┐
│  NODE 3: Factual Verification / NLI  │ <── Jev ('Confidently Wrong' Trap) vs. DeBERTa-v3 (100% Epistemic Gate)
└──────────────────────────────────────┘
```

---

## 2. Core Empirical Findings

### Node 1: Intent Classification & Statutory Entity Extraction
* **Execution Latency:** Live HTTPS roundtrips to Jev average **712ms P50 (up to 1,348ms P90)**. In contrast, my sovereign unit runs in **0.14ms P50 (0.33ms P90)**—running over **5,000x faster**. A front-door router running at hundreds of milliseconds introduces an unviable latency penalty before retrieval even begins.
* **Token Coordinate Extraction:** Jev is strictly a categorical classifier. It outputs an enum, but cannot extract statutory coordinates (e.g., `Companies Act 2006 s.382`). My sovereign unit extracts canonical URI coordinates directly, enabling hard database pre-filtering.

### Node 2: Candidate Passage Reranking & The "Cheap Token" Volume Trap
* **The Margin Inversion:** TypeSafe markets $0.042/1M input tokens as negligible. However, pairwise reranking scales linearly ($N \times M$). In enterprise legal search (e.g. 40 queries evaluated against 30 candidate passages), Jev consumes **~38,400 tokens per search query**.
* **Enterprise OPEX at 1M Queries/Day:**
  $$\text{1,000,000 queries/day} \times \text{38,400 tokens} = \text{38.4 Billion tokens/day}$$
  $$\text{38,400 Mtok} \times \$0.042 = \mathbf{\$1{,}612.80/\text{day}} \implies \mathbf{\sim \$48{,}384/\text{month}}$$
  Deploying Jev recreates the exact **$50,000/month pass-through SaaS tax** that deterministic sovereign caching eliminates.
* **Sub-Chunk Span Attribution:** Jev treats candidate text as an opaque string and outputs a single scalar number. My ColBERT late-interaction engine localizes the exact **50-word relevant span inside a 500-word chunk**, cutting downstream NLI auditor compute by 90%.

### Node 3: Factual Verification & The "Confidently Wrong" Entropy Trap
* **Streaming Physics:** An in-flight sentinel auditing sentences during token streaming (e.g., Llama emitting a 15-token sentence in ~375ms) cannot wait 80–700ms for an external HTTP call without causing severe typewriter jitter, while immediately hitting Jev's 1,200 RPM rate limit ceiling.
* **Statistical Entropy vs. Epistemic Grounding:** Jev’s confidence metric is derived from softmax distribution shape (entropy). When fed an adversarial probe citing an invented, fictional statute (*Marchwood Commercial Arbitration Order 2022*), Jev outputs `choice: supports` with `confidence: 0.96`. It is statistically confident, but epistemically ungrounded. My sovereign sentinel intercepts the ungrounded instrument at the gate, enforcing a **100% clean epistemic abstention by construction**.

---

## 3. Scale OPEX Margin Inversion Table

Empirical token costs comparing Jev's cloud billing against my sovereign architecture ($0 marginal token cost on dedicated hardware):

| Daily Query Volume | Jev Daily Tokens | Jev Cost / Day | Jev Cost / Month | Sovereign Marginal Cost | Monthly Enterprise Savings |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10,000 queries/day** | 20,040,000 | $0.84 | $25.25 | **$0.00** | **$25.25** |
| **100,000 queries/day** | 200,400,000 | $8.42 | $252.50 | **$0.00** | **$252.50** |
| **1,000,000 queries/day** | 2,004,000,000 | $84.17 | **$2,525.04** | **$0.00** | **$2,525.04** |
| **1,000,000 q/day (Rerank 30)** | 38,400,000,000 | $1,612.80 | **$48,384.00** | **$0.00** | **$48,384.00** |

---

## 4. Regulatory Compliance & TPRM Under EU AI Act Article 15

Under **EU AI Act Article 15 (Accuracy, Robustness, and Cybersecurity)** and Third-Party Risk Management (TPRM) standards:
* **The Cloud Alias Vulnerability:** TypeSafe's documentation explicitly notes:
  > *"An alias moves when a new release ships, so the answers behind it can change without a change on your side."*
  A closed SaaS endpoint with shifting aliases (`jev-latest`, `typesafe/jev-1.13-20260917`) breaks auditability. An enterprise cannot guarantee bit-level reproducibility 18 months later for an auditor.
* **The Sovereign Guarantee:** My sovereign models (`cross-encoder/nli-deberta-v3-base`) operate with permanently frozen weight hashes deployed inside private customer VPCs, guaranteeing bit-level deterministic reproducibility.

---

## 5. Where Jev System One Actually Wins

While Jev does not replace the online production pipeline, there is one critical area in developer tooling where it excels:
* **Offline Diagnostic Grading (`legal-rag-audit`):** In my offline test harness, evaluating whether a model preserved complex statutory carve-outs (Q4) previously required an expensive GPT-4o judge. Jev's `Choice`/`Noul` primitive is an ideal, low-cost evaluator for offline grading manifests where 100ms network latency is completely harmless.
* **Standard SaaS Software:** Replacing brittle OpenAI JSON prompt engineering in non-critical webhooks with calibrated probabilities.

---

## 6. Frictionless One-Command Reproduction

### Prerequisites
* Python 3.10+ (tested on Python 3.14 on macOS MPS and Linux CUDA)
* PyTorch 2.1+, Transformers 4.38+, Rich, Httpx

### Installation
```bash
git clone https://github.com/azterizm/jev-vs-sovereign-benchmark.git
cd jev-vs-sovereign-benchmark

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tests
```bash
pytest tests/ -v
```

### 1. Offline Dry-Run Benchmark (Zero Cloud Cost)
Execute the complete suite offline using realistic recorded telemetry traces:
```bash
python3 scripts/run_benchmark.py --dry-run --node all --warmup 2 --iterations 5
```

### 2. Live Empirical Benchmark Against OpenRouter
Export your OpenRouter API key and execute the full test battery:
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python3 scripts/run_benchmark.py --mode live --node all --warmup 5 --iterations 20 --export-markdown results/benchmark_report.md
```

The runner automatically generates:
* Rich terminal tables with P50, P90, P99 latency percentiles and token volume economics.
* A standalone publication whitepaper (`results/benchmark_report.md`).
* A cryptographically sealed JSON telemetry log (`results/benchmark_telemetry.json`) with an immutable SHA-256 master certificate for EU AI Act Article 15 compliance.
