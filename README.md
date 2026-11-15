# Evaluative Benchmarking of LLM-as-a-Judge for Automated Code Auditing

**Course:** UE24AM343AB5, Semester 5 | **Problem Statement 3**  
**Team Deliverables:** High-Level Design, Curated Data, Working Demo, Benchmark Results, IEEE Conference Report (10 pages), Presentation Slides, and Viva Preparation.

---

## 🎯 Executive Summary (Project in One Line)

We test whether small, open-source language models ($\le 7\text{B}$) running on local developer hardware can reliably judge code correctness, quantify how severely their verdicts change when only candidate presentation order is reversed (position bias), and evaluate whether prompt-level instructions or architectural order-averaging eliminate the vulnerability.

---

## ⚡ Headline Benchmark Results (2,400 Model Evaluations)

| Metric | Qwen2.5-Coder-3B | Qwen2.5-Coder-7B (4-bit) | Key Takeaway |
|---|---|---|---|
| **Pairwise Accuracy (Correct = A)** | **98.7%** | **96.0%** | Near-perfect score when correct code is presented first. |
| **Pairwise Accuracy (Correct = B)** | **6.7%** | **12.7%** | Catastrophic drop when correct code is placed second. |
| **Position Accuracy Gap** | **+92.0 pp** | **+83.3 pp** | Extreme presentation-order sensitivity. |
| **Order Consistency** | **8.0%** (138 AA flips) | **15.3%** (126 AA flips) | In **92% of pairs**, the verdict followed position, not code! |
| **Pointwise False Acceptance (FAR)** | **76.0%** | **72.0%** | ~3 out of 4 buggy programs pass review (leniency bias). |
| **Fix: Order-Averaging (Exp 4a)** | **83.3% Acc (100% Cons)** | **78.3% Acc (100% Cons)** | **Only robust fix:** filters out inconsistent pairs. |
| **Fix: Debias Prompt (Exp 4b)** | 51.7% Acc ($p = 0.453$) | 53.0% Acc ($p = 0.289$) | Prompts **fail** to reduce position bias. |
| **Fix: Majority Vote@3 (Exp 4c)** | 50.3% Acc ($p = 0.555$) | — | Sampling does not overcome systematic token bias. |

---

## 📂 Repository Structure

```
peogai-judge/
├── app/
│   ├── app.py                   # Full interactive Gradio web application
│   └── web_demo.py              # Zero-dependency standalone demo (built-in http.server)
├── data/
│   ├── pairs_150.jsonl          # 150 stratified code pairs (50 easy / 50 med / 50 hard)
│   ├── pairs_150_stats.json     # Dataset balance & line count metadata
│   ├── pairs_150_verbose.jsonl  # Exp 2 verbosity perturbation pairs (AST logic preserved)
│   └── demo_examples.json       # 3 curated ready-made demo examples for the live demo
├── demo/
│   └── DEMO_SCRIPT.md           # 3-to-4 minute video script, cue cards, & two-device checklist
├── notebooks/
│   └── p1_judge_runs.ipynb      # End-to-end Colab/Kaggle notebook for all 2,400 calls
├── prompts/
│   ├── pairwise_v1.txt          # Standard pairwise prompt (verdict: A / B)
│   ├── pairwise_debias.txt      # Debias prompt instructing judge to ignore order & length
│   └── pointwise_v1.txt         # Pointwise prompt (verdict: correct / incorrect)
├── report/
│   ├── experimental_setup.tex   # Section IV: Hardware, models, latency, and decoding
│   └── results.tex              # Section V: Detailed empirical results, tables, and figures
├── results/
│   ├── figures/                 # Publication-ready plots (PDF for LaTeX, PNG for slides)
│   │   ├── f1_position_bias.png # Fig 1: Position bias gap (A vs B)
│   │   ├── f2_pointwise.png     # Fig 2: Pointwise FAR and FRR rates
│   │   ├── f3_fixes.png         # Fig 3: Baseline vs Order-Avg vs Debias vs Vote@3
│   │   ├── f4_verbosity.png     # Fig 4: Effect of comment padding on wrong answer wins
│   │   └── f5_difficulty.png    # Fig 5: Results across difficulty tiers
│   ├── tables/                  # Formatted tables (.csv, .md for slides, .tex for report)
│   ├── raw/                     # 2,400 raw execution JSONL logs from all 8 experiment runs
│   ├── key_numbers.md           # Master list of all numerical results with 95% Wilson CIs
│   └── setup_summary.md         # Hardware, latency, and session cost summary
├── scripts/
│   └── setup_ollama.sh          # One-click Ollama verification & model pull for macOS M3 Pro
├── slides/
│   ├── index.html               # Interactive 14-slide HTML5 presentation (open in any browser)
│   ├── presentation.md          # Marp-compatible Markdown presentation deck
│   └── presentation.tex         # LaTeX Beamer slide deck for Overleaf / pdflatex
├── src/
│   ├── judge.py                 # Core unified judge() function with multiple backends
│   ├── run_experiment.py        # Resumable batch runner with Google Drive checkpointing
│   ├── sample_data.py           # Stratified sampling from CodeJudgeBench (seed 42)
│   ├── perturb.py               # AST-validated comment padding for Exp 2
│   ├── metrics.py               # Statistical metrics, Wilson CIs, and McNemar paired tests
│   ├── plots.py                 # Plot generation script with consistent model palette
│   └── setup_summary.py         # Summary generator for experimental setup
├── requirements.txt             # Python dependencies
└── README.md                    # Master documentation
```

---

## 🚀 Quickstart & Setup

### 1. Environment Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/vaibhav/peogai-judge.git
cd peogai-judge

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Local Model Setup (macOS Apple Silicon / M3 Pro)
For the live interactive demo, install [Ollama](https://ollama.com) and pull the 3B coder model:
```bash
# Using the automated setup script:
bash scripts/setup_ollama.sh

# Or manually:
brew install --cask ollama
ollama serve &
ollama pull qwen2.5-coder:3b
```

---

## 🧪 Running the Interactive Demo (P3 Deliverable)

### Option A: Full Gradio Application
```bash
python -m app.app
```
Then open your browser at **`http://localhost:7860`**.

### Option B: Zero-Dependency Standalone Web Demo
If running in an environment without Gradio installed, launch the lightweight built-in server:
```bash
python -m app.web_demo
```
Then open **`http://localhost:7860`**.

### Demo Features:
1. **Dual-Order Execution:** Automatically evaluates Candidate 1 vs Candidate 2 in Order 1 `(1 as A, 2 as B)` and Order 2 `(2 as A, 1 as B)`.
2. **Real-Time Position Bias Detector:** Instantly flags `🚨 POSITION BIAS DETECTED` if the judge's verdict flips when order swaps.
3. **Order-Averaged Mitigation Filter:** Displays verified verdict if both orders agree, or safely rejects the pair as `UNDECIDED` if contradictory.
4. **Pre-Loaded Benchmark Cases:** 1-click loading of real cases from the benchmark:
   - **Case 1 (p015):** Severe Position Bias Flip (AA Flip on DP Subsequence Sum).
   - **Case 2 (p062):** Consistent Agreement (Greedy $O(N)$ vs DP $O(N^2)$).
   - **Case 3 (p014):** Algorithmic Complexity Comparison under Distance Constraints.
5. **Benchmark Visualizer:** Live interactive tabs displaying all publication figures (Figs 1–5), numerical tables, and findings.

---

## 🖥️ Viewing the Slide Deck (14 Slides)

The slide deck is provided in three production-ready formats:
1. **Interactive HTML5 Presentation:**
   ```bash
   open slides/index.html
   ```
   Navigate with Left/Right Arrow keys or Spacebar. Press `F` for fullscreen.
2. **Marp / Markdown Deck:**
   `slides/presentation.md` (can be exported to PDF via Marp CLI or opened in VS Code).
3. **LaTeX Beamer Deck:**
   `slides/presentation.tex` (ready for Overleaf or `pdflatex`).

---

## 📊 Step-by-Step Pipeline Reproduction

### 1. Data Sampling & Perturbation (P2)
```bash
# Sample 150 balanced pairs from CodeJudgeBench (seed 42)
python -m src.sample_data

# Create AST-equivalent comment-padded pairs for Exp 2
python -m src.perturb
```

### 2. Running Batch Experiments (P1)
Batch runs can be executed locally with `--backend mock` or on Colab/Kaggle with GPU:
```bash
# Run Exp 1 Pairwise Baseline
python -m src.run_experiment --exp exp1 --model 3b --backend vllm

# Run Exp 3 Pointwise Auditing
python -m src.run_experiment --exp exp3 --model 3b --backend vllm

# Run Exp 4b Debias Prompt & Exp 4c Vote@3
python -m src.run_experiment --exp exp4b --model 3b --backend vllm
python -m src.run_experiment --exp exp4c --model 3b --backend vllm
```

### 3. Computing Metrics, Tables & Plots (P2)
```bash
# Calculate statistical metrics, Wilson CIs, and McNemar significance
python -m src.metrics

# Render publication-grade figures (PNG and PDF)
python -m src.plots
```

---

## 💡 Viva Voce & Defense Cheat Sheet

| Question | Recommended Answer |
|---|---|
| **Why only test correctness, and not security or readability?** | Security and readability lack deterministic, executable ground truth in benchmark datasets. By focusing strictly on functional correctness, our ground truth is 100% verified by automated compiler and unit test execution, eliminating human subjective noise. |
| **Why no human annotations?** | Human annotations introduce subjective variance and inter-annotator disagreement. Test-suite pass/fail is the gold standard for software engineering auditing. |
| **Are 150 pairs (300 programs) sufficient?** | Yes. We evaluated 2,400 total calls with 95% Wilson confidence intervals. The position gap (+92 pp) is so massive that the statistical effect is overwhelming ($p < 0.001$). Paired comparisons use exact McNemar tests on identical calls. |
| **Why focus on $\le 7\text{B}$ models?** | Large proprietary models (GPT-4) are too slow, costly, and privacy-invasive for automated local CI/CD pre-commit hooks. Small open models represent the real frontier for local edge deployment. |
| **How does this differ from "Bias in the Loop"?** | "Bias in the Loop" focused primarily on length bias in 70B+ proprietary models on open-ended text. We provide the first systematic audit of position and leniency bias in lightweight coder models on strict unit-test verified code. |
| **Why does False Acceptance Rate (FAR) matter more than accuracy?** | In code auditing, a False Rejection (FRR) is an annoyance (a human re-checks valid code), but a False Acceptance (FAR) is a critical security and functional failure (buggy or vulnerable code silently merges to production). An FAR of 76% means the judge is dangerous. |
| **Can prompting or sampling fix position bias?** | No. Our experiments showed that debias prompts ($p = 0.453$) and majority voting ($p = 0.555$) had no statistically significant effect. The bias is rooted in structural token attention representations, requiring architectural dual-order averaging. |

---

## 👥 Team Responsibilities (P1 - P4)

- **P1 (Setup & Batch Runs):** Colab/Kaggle infrastructure, `src/judge.py`, `src/run_experiment.py`, 2,400 inference calls, Section IV setup text.
- **P2 (Data & Metrics):** `src/sample_data.py`, `src/perturb.py`, `src/metrics.py`, `src/plots.py`, Tables 1–5, Figures 1–5, Section V results text.
- **P3 (Demo, Slides & Repo):** `app/app.py` (Gradio), `app/web_demo.py` (standalone server), 3 ready-made demo cases, `demo/DEMO_SCRIPT.md` (3-4 min video guide), 14-slide deck (`slides/index.html`, `slides/presentation.md`, `slides/presentation.tex`), and master `README.md`.
- **P4 (Report & Synthesis):** IEEE 10-page conference paper draft, system design diagram, limitations analysis, and viva defense synthesis.

