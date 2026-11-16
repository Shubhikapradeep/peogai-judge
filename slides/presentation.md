---
marp: true
theme: default
paginate: true
header: "PEOGAI Mini Project | Problem Statement 3"
footer: "UE24AM343AB5 — Evaluative Benchmarking of LLM-as-a-Judge for Code Auditing"
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 24px;
    padding: 40px;
  }
  h1 { color: #1e3a8a; }
  h2 { color: #2563eb; }
  .highlight { color: #dc2626; font-weight: bold; }
  .success { color: #16a34a; font-weight: bold; }
  table { font-size: 19px; width: 100%; border-collapse: collapse; }
  th, td { border: 1px solid #cbd5e1; padding: 6px 10px; }
  th { background-color: #f1f5f9; }
---

# Evaluative Benchmarking of LLM-as-a-Judge for Automated Code Auditing
### Assessing Position Bias, Leniency Bias, and Mitigation Bounds in Edge-Scale LLMs

**Course:** UE24AM343AB5 | Problem Statement 3  
**Team PEOGAI:**
- **P1:** Model Setup, Inference Pipelines & Colab Batch Runs
- **P2:** Dataset Stratification, Prompts & Metrics Engine
- **P3:** Interactive Demo System, Slide Deck & Open-Source Repository
- **P4:** IEEE Conference Report & Research Synthesis

---

## 2. The Problem: Can We Trust Small LLM Judges?

- **Context:** Automated code review is a vital gatekeeper in continuous integration / continuous deployment (CI/CD) pipelines.
- **The LLM-as-a-Judge Paradigm:** Evaluator LLMs are increasingly deployed to grade code correctness, review pull requests, and filter test candidates.
- **The Core Dilemma:**
  - Modern production uses massive closed models (GPT-4, Claude 3.5), which are costly, non-private, and slow.
  - Can small, edge-deployable open-source models ($\le 7\text{B}$) running locally on developer laptops reliably judge code?
  - **The Risk:** If small models suffer from superficial biases (e.g. presentation order), broken code could pass silently into production.

---

## 3. Literature Summary & Foundations

- **LLM-as-a-Judge Foundations:**
  - *Zheng et al. (2023) [MT-Bench]:* First documented that conversational LLMs exhibit position bias, verbosity bias, and self-enhancement bias when judging text.
- **Code Evaluation Benchmarks:**
  - *CodeJudgeBench (Chen et al., 2024):* Established pairwise code generation testbeds with compiler-verified ground truth.
  - *CodeXGLUE / HumanEval / MBPP:* Standard single-generation code generation and testing datasets.
- **Prior Bias Research:**
  - *"Bias in the Loop" (Stureborg et al., 2024):* Focused on length and style biases in proprietary 70B+ models, leaving smaller edge-scale coder models uncharacterized in code evaluation.

---

## 4. Research Gaps & Our Contributions

| Prior Literature | Our Contribution |
|---|---|
| Focused primarily on massive closed models (GPT-4, Claude 3.5, 70B+) | **Empirical evaluative audit of edge-scale coder LLMs (3B & 7B)** on local consumer hardware. |
| Relied on subjective human grading or loose NLP metrics | **Ground truth rooted entirely in objective unit-test execution** (no human label noise). |
| Examined general NLP / chatbot dialogues | **Focused strictly on functional code auditing** under competitive programming constraints. |
| Assumed prompt fixes work across domains | **Empirically demonstrated that debias prompts fail**; dual-order averaging acts as an inconsistency filter (retaining 83.3% accuracy on the 8% decided pairs). |

---

## 5. Research Questions (RQs)

- **RQ1 (Accuracy):** How accurately can small open-source models (3B & 7B) judge functional code correctness in pairwise and pointwise settings?
- **RQ2 (Position Bias):** To what extent does answer presentation order dictate the model's verdict, and what is the resulting consistency gap?
- **RQ3 (False Acceptance):** When evaluating single submissions (pointwise auditing), what is the False Acceptance Rate (FAR) of broken code?
- **RQ4 (Mitigation Bounds):** Can prompt-level instructions or stochastic majority voting eliminate position bias, or is architectural order-averaging required?
- **RQ5 (Verbosity Bias):** Does superficial comment padding that claims correctness sway the judge without changing underlying code logic?

---

## 6. Project Scope: What We Did vs. Did Not Do

- **What We Did:**
  - Evaluated **150 balanced code pairs (300 solutions)** from CodeJudgeBench.
  - Stratified evenly: 50 Easy, 50 Medium, 50 Hard across AtCoder, LeetCode, and Codeforces.
  - Tested **both permutations** (Order 1 vs Order 2) across all pairs (2,400 total calls).
  - Evaluated 3 distinct mitigation techniques: Order-Averaging, Debias Prompts, Majority Vote@3.
- **Deliberate Scope Boundaries (Why):**
  - *No closed proprietary APIs:* Evaluated only open-weight models ($\le 7\text{B}$) for privacy and reproducibility.
  - *No self-enhancement bias:* Models did not evaluate their own generated code.
  - *No subjective style/security metrics:* Confined to binary functional correctness (hidden test passes).

---

## 7. System Architecture & Pipeline

```
CodeJudgeBench (2,103 raw pairs)
          │
          ▼
Stratified Sampler (seed=42, 50 Easy / 50 Med / 50 Hard) ──> data/pairs_150.jsonl
          │
          ▼
Prompt Formatter (Pairwise, Debias, Pointwise) ──────────> prompts/
          │
          ▼
Inference Core (src/judge.py: vLLM / Transformers / Ollama)
          │
   ┌──────┴───────────────────────┐
   ▼                               ▼
Batch Runner                    Live Demo App
(Google Colab T4 GPU)           (MacBook M3 Pro Ollama)
   │                               │
   ▼                               ▼
Raw JSONL Logs                   Dual-Order Runner
(results/raw/*.jsonl)           (app/app.py & web_demo.py)
   │                               │
   ▼                               ▼
Metrics & Plots Engine           Bias Detector &
(src/metrics.py)                 Order-Averaged Filter
   │
   ▼
IEEE Report & Slide Deck
```

---

## 8. Dataset Profile (CodeJudgeBench Sample)

- **Sample Size:** 150 pairs (300 candidate solutions), seeded at 42.
- **Difficulty Stratification:** 50 Easy (33.3%), 50 Medium (33.3%), 50 Hard (33.3%).
- **Platforms:** LeetCode (78), AtCoder (71), Codeforces (1).
- **Source Generator Splits:** Claude 3.7 Sonnet (30), Gemini 2.5 Flash (36), Gemini 2.5 Flash-Lite (28), Claude 4 Sonnet (18), Claude 4 Opus (14), Qwen 235B (14), Gemini 2.5 Pro (10).
- **Code Length Balance:**
  - Correct solutions: **64.8 average lines**
  - Incorrect solutions: **65.3 average lines**
  - Ensures no inherent length confounder in the baseline data!

---

## 9. Models & Experimental Setup

- **Evaluator Models:**
  - **Qwen2.5-Coder-3B-Instruct:** Evaluated in fp16 (vLLM on Colab T4; GGUF Q4_K_M on M3 Pro).
  - **Qwen2.5-Coder-7B-Instruct:** Evaluated in 4-bit AWQ via vLLM on Colab T4.
- **Decoding Configuration:**
  - Greedy decoding ($T=0$) for Exp 1, 2, 3, and 4b; max 16 new tokens.
  - Temperature $T=0.7$, top-$p=0.95$, $N=3$ samples for Majority Voting (Exp 4c).
- **Scale and Compute:**
  - Total calls: **2,400 judge evaluations** across 8 full experiment runs.
  - Total compute: **1.27 GPU-hours** on a single NVIDIA Tesla T4.
  - Zero parse failures across all 2,400 calls.

---

## 10. Experiments Overview

- **Exp 1: Pairwise Judging & Position Bias**
  - Present Candidate 1 and Candidate 2 in both orders: $(1,2)$ and $(2,1)$.
  - Measure accuracy, consistency across orders, and share of `A` verdicts.
- **Exp 2: Verbosity Bias Perturbation**
  - Pad the incorrect answer with comments claiming correctness (AST logic unchanged).
  - Measure shift in win-rate of flawed code.
- **Exp 3: Pointwise Single-Code Auditing**
  - Evaluate each of the 300 answers individually (`correct` vs `incorrect`).
  - Measure False Acceptance Rate (FAR) and False Rejection Rate (FRR).
- **Exp 4: Mitigation Strategies**
  - **4a:** Dual-Order Averaging (accept only if both orders agree).
  - **4b:** Explicit debias prompt instruction.
  - **4c:** Stochastic Majority Voting (Vote@3 at $T=0.7$).

---

## 11. Results: Pairwise & Pointwise Auditing

### Exp 1: Pairwise Position Bias (150 pairs, 300 calls/model)
| Model | Acc (Correct=A) | Acc (Correct=B) | Accuracy Gap | Consistency | % A Verdicts | Flips (AA) |
|---|---|---|---|---|---|---|
| **Qwen2.5-Coder-3B** | **98.7%** | **6.7%** | <span class="highlight">+92.0 pp</span> | <span class="highlight">8.0%</span> | **96.0%** | **138 / 150** |
| **Qwen2.5-Coder-7B** | **96.0%** | **12.7%** | <span class="highlight">+83.3 pp</span> | <span class="highlight">15.3%</span> | **91.7%** | **126 / 150** |

### Exp 3: Pointwise Auditing (300 answers evaluated)
| Model | Overall Accuracy | False Acceptance (FAR) | False Rejection (FRR) | % Labeled Correct |
|---|---|---|---|---|
| **Qwen2.5-Coder-3B** | 53.0% | <span class="highlight">76.0%</span> (7.6 in 10 pass) | 18.0% | **79.0%** (Leniency bias) |
| **Qwen2.5-Coder-7B** | 56.3% | <span class="highlight">72.0%</span> (7.2 in 10 pass) | 15.3% | **78.3%** (Leniency bias) |

---

## 12. Results: Before and After Fixes

### Exp 4: Comparison of Position Bias Mitigations
| Model | Method | Accuracy | Consistency | % A Verdicts | Coverage | $p$ (accuracy) |
|---|---|---|---|---|---|---|
| **3B** | Baseline (Exp 1) | 52.7% | 8.0% | 96.0% | 100.0% | — |
| | **Order Averaging (4a)** | <span class="success">83.3%</span> | <span class="success">100.0%</span> | — | 8.0% | — |
| | Debias Prompt (4b) | 51.7% | 3.3% | 98.3% | 100.0% | $p = 0.453$ |
| | Majority of 3 (4c) | 50.3% | 8.7% | 65.7% | 100.0% | $p = 0.555$ |
| **7B** | Baseline (Exp 1) | 54.3% | 15.3% | 91.7% | 100.0% | — |
| | **Order Averaging (4a)** | <span class="success">78.3%</span> | <span class="success">100.0%</span> | — | 15.3% | — |
| | Debias Prompt (4b) | 53.0% | 11.3% | 93.7% | 100.0% | $p = 0.289$ |

- **Key Takeaway:** Debias prompt actually increased Position A choices (96.0% $\rightarrow$ 98.3%, $p=0.016$) with no accuracy change ($p=0.453$). Majority voting dropped A choices to 65.7% ($p<0.001$) but did not improve accuracy ($p=0.555$) or consistency (8.7%, $p=1.000$). Dual-order averaging serves as an inconsistency filter, raising accuracy on decided pairs to 83.3% while abstaining on 92.0% of pairs.

---

## 13. Key Findings & Limitations

- **Finding 1 (Position Bias):** 3B voted A in 96.0% and 7B in 91.7% of calls; overall accuracy (52.7% / 54.3%) is not better than chance. Rare B votes carry true signal (83.3% right on 3B, $p=0.039$).
- **Finding 2 (Pointwise Leniency):** 3B accepts 76.0% and 7B accepts 72.0% of buggy code (FAR), approving ~79% of all submissions.
- **Finding 3 (Scale Comparison):** 7B is less position-biased (91.7% vs 96.0% A, $p=0.011$), but no more accurate (54.3% vs 52.7%, $p=0.405$).
- **Finding 4 (Fixes Fail):** Debias prompt worsened 3B bias (98.3% A, $p=0.016$); vote@3 spread votes without accuracy gain (50.3%, $p=0.555$). Order-averaging acts as an inconsistency filter (83.3% acc, 8.0% coverage).
- **Finding 5 (Verbosity Padding):** Comment padding on flawed code only boosted wins in position A (93.3% $\rightarrow$ 99.3%, $p=0.004$) while position B dropped (1.3% $\rightarrow$ 0%, $p=0.500$; overall shift 47.3% $\rightarrow$ 49.7%, $p=0.065$ not significant), suggesting comments amplify presentation order rather than indicating semantic preference.
- **Limitations:**
  - **Exp 2 Length Confound:** Comment padding made the incorrect candidate longer in 127 of 150 pairs.
  - **Statistical Power on Small Shifts:** With $N=150$ pairs, 95% CIs are approximately $\pm 8$ percentage points per pair-level rate; sufficient for the massive +92 pp position gap, but underpowered for subtle differences.
  - **Single Prompt Template:** Evaluated one prompt wording per task format; sensitivity to prompt variations remains unmeasured.
  - **Difficulty Noise & Scope:** Difficulty labels in CodeJudgeBench disagree across generator splits for 82% of shared problems; evaluated on Python competitive programming capped at $\le 7$B parameters.

---

## 14. Live Demonstration & Repository

- **Interactive Demo System (`app/app.py` & `app/web_demo.py`):**
  - Real-time dual-order execution on Apple Silicon (M3 Pro).
  - Instant detection and flagging of presentation-order flips.
  - Automated order-averaged mitigation filter for CI/CD safety.
  - Interactive exploration of benchmark figures and tables.
- **Ready-Made Benchmark Cases:**
  - *Example 1 (p015):* Severe Position Bias Flip (AA Flip on DP Subsequence Sum).
  - *Example 2 (p062):* Consistent Correct Decision (Greedy $O(N)$ vs DP $O(N^2)$).
  - *Example 3 (p024):* Consistent Correct Decision (Server Request Window Counter).
  - *Example 4 (p028):* Severe Position Bias Flip (Maximum Strong Pair XOR).
- **GitHub Repository Deliverables:** Fully reproducible code, prompts, raw logs, figures, and IEEE report draft.

---

# Thank You!
### Questions & Discussion

**Repository:** `peogai-judge`  
**Demo Launch:** `python -m app.app` (or `python -m app.web_demo`)  
**IEEE Report:** `report/results.tex` & `report/experimental_setup.tex`

