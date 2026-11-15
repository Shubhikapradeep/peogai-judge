# 🎬 3 to 4 Minute Live Demo Video Script & Walkthrough

**Project:** Evaluative Benchmarking of LLM-as-a-Judge for Automated Code Auditing (Problem Statement 3)  
**Target Duration:** 3 minutes 30 seconds (3:30)  
**Presenter(s):** Team P3  
**Recommended Screen Recording Tool:** macOS QuickTime Player (Cmd + Shift + 5) or OBS Studio  
**Resolution:** 1080p (1920x1080) or 1440p  

---

## ⏱️ Video Timeline Overview

| Timestamp | Phase | What to Show on Screen | Key Spoken Points |
|---|---|---|---|
| **0:00 – 0:35** | **1. Context & Motivation** | Title Slide / Web App Header | Problem Statement 3, why small LLMs (3B/7B) matter, and why LLM-as-a-judge cannot be trusted naively. |
| **0:35 – 1:35** | **2. Exposing Severe Position Bias (Case 1)** | Tab 1: Load Example 1 (p015). Run Dual-Order Evaluation. | Show Order 1 (picks A) and Order 2 (also picks A). Highlight 🚨 **Position Bias Detected**. Explain the AA flip. |
| **1:35 – 2:20** | **3. Order-Averaging Mitigation & Consistent Case (Case 2)** | Show Order-Averaging Filter on Case 1. Then switch to Example 2 (p062) and run it. | Explain how order-averaging safely filters out contradictory pairs. Show Case 2 where greedy solution is chosen in both orders. |
| **2:20 – 3:05** | **4. Empirical Benchmark Results** | Tab 2: Show Figures 1, 2, 3 and Table 1. | 2,400 calls summary: 96% first-position bias, 76% pointwise false acceptance rate (FAR), why debias prompts failed. |
| **3:05 – 3:35** | **5. Conclusion & Takeaway** | Architecture tab & concluding summary | Key takeaway: small LLM judges require mandatory permutation checks before any CI/CD deployment. |

---

## 🎙️ Detailed Spoken Script with On-Screen Cues

### [0:00 – 0:35] Part 1: Context & Core Question
* **[ACTION]**: Screen shows the Web App header: *Evaluative Benchmarking of LLM-as-a-Judge for Automated Code Auditing*. Camera/Voice clear.
* **[SPOKEN]**:  
  > "Hello everyone. Today we are presenting our evaluative benchmark of LLM-as-a-Judge for automated code auditing.  
  > Automated code review is increasingly critical in modern CI/CD pipelines. But can lightweight, open-source models—specifically 3B and 7B parameter models running on local developer hardware like this MacBook M3 Pro—reliably evaluate code correctness?  
  > More importantly: does their verdict depend on the code's actual logic, or simply on the order in which solutions are presented? Let's find out."

---

### [0:35 – 1:35] Part 2: Exposing Severe Position Bias (Example 1)
* **[ACTION]**: Click dropdown **Load Ready-Made Benchmark Case** and select **Example 1: Longest Subsequence Sum (p015)**.
* **[ACTION]**: Point out Candidate 1 (Correct $O(N \cdot \text{target})$ DP) vs Candidate 2 (Incorrect Dictionary implementation with state explosion).
* **[ACTION]**: Ensure Backend is set to `Ollama (Qwen2.5-Coder-3B)` (or `Benchmark Replay`) and click **🚀 Run Dual-Order Evaluation**.
* **[SPOKEN]**:  
  > "Here we have a dynamic programming problem from LeetCode. Candidate 1 is an optimal 1D DP array. Candidate 2 is a flawed dictionary implementation that hits memory limits and duplicate state bugs.  
  > When we evaluate in automated code auditing, we normally present two solutions: Answer A and Answer B.  
  > Watch what happens when our system tests **both permutations simultaneously**!  
  > In Order 1, Candidate 1 is placed in position A, and Candidate 2 is placed in position B. The model outputs token `A`—selecting Candidate 1.  
  > But now look at Order 2: we swap the positions! Candidate 2 is placed in position A, and Candidate 1 in position B. The model *again* outputs token `A`—this time selecting the broken code!  
  > Our system flags this instantly: **🚨 Position Bias Detected!**  
  > In 92% of the pairs in our benchmark, this model simply voted for whichever code appeared first on its screen, irrespective of correctness."

---

### [1:35 – 2:20] Part 3: Order-Averaging Fix & Consistent Case (Example 2)
* **[ACTION]**: Point the mouse to the **Order-Averaging Filter** result box for Example 1.
* **[SPOKEN]**:  
  > "If this judge were deployed naively in CI/CD, bad code would be approved simply because it was submitted as Option A.  
  > Our fix is **Order-Averaging**: a verdict is only valid if the judge agrees across both presentation orders. Because Order 1 and Order 2 contradicted each other, the system marks this pair as **Undecided**, filtering it out safely."
* **[ACTION]**: Switch dropdown to **Example 2: Beautiful Binary String (p062)**. Click **Run Dual-Order Evaluation**.
* **[SPOKEN]**:  
  > "Now let's examine Example 2. Here, Candidate 1 is an optimal $O(N)$ greedy check, while Candidate 2 is a quadratic $O(N^2)$ DP that times out.  
  > When we run the dual-order test: Order 1 selects `A` (Candidate 1), and Order 2 selects `B` (still Candidate 1!).  
  > Both orders agree on the same underlying code. The system verifies this with **✅ Verdicts Consistent Across Orders**, and order-averaging confirms Candidate 1 as the true winner."

---

### [2:20 – 3:05] Part 4: Benchmark Findings Across 2,400 Calls
* **[ACTION]**: Switch to **Tab 2: Empirical Benchmark Results**. Scroll smoothly through Figures 1, 2, and 3.
* **[SPOKEN]**:  
  > "This isn't just an isolated anomaly. We ran 2,400 rigorous evaluations across Google Colab T4 GPUs and our local setup. Here are the three critical findings:  
  > 1. **Massive Position Bias Gap (Fig 1):** Qwen2.5-Coder-3B chose position A in **96%** of all calls. When the correct code was A, accuracy was **98.7%**. When it was B, accuracy plummeted to **6.7%**—a shocking **92 percentage point gap**! Scaling to 7B only narrowed this gap slightly to 83 percentage points.  
  > 2. **Alarming False Acceptance Rate (Fig 2):** In pointwise auditing—checking one code snippet at a time—the judge exhibited a **76% False Acceptance Rate**. It accepted roughly 8 out of 10 buggy submissions due to strong leniency bias.  
  > 3. **Prompt Fixes Do Not Work (Fig 3):** Explicitly warning the judge to ignore order and length with a debiasing prompt changed nothing (p=0.453). Sampling 3 votes at temperature 0.7 also failed (p=0.555). Only **dual-order averaging** reliably eliminates position bias, raising accuracy to **83.3%** on decided pairs."

---

### [3:05 – 3:35] Part 5: Architecture & Conclusion
* **[ACTION]**: Switch to **Tab 3: System Architecture & Setup**.
* **[SPOKEN]**:  
  > "In summary: small open-source models under 7B cannot be trusted as standalone code reviewers without architectural safeguards. Naive single-order prompts produce random chance accuracy and dangerous false approvals.  
  > Dual-order evaluation and order-averaging provide a lightweight, deterministic defense that can be dropped into any code auditing harness today.  
  > Thank you!"

---

## 📋 Backup & Two-Device Checklist

- [ ] Export demo video as `.mp4` (H.264, AAC audio).
- [ ] Name the file: `PEOGAI_Demo_Video_P3.mp4`.
- [ ] **Device 1:** Local storage on laptop (`peogai-judge/demo/PEOGAI_Demo_Video_P3.mp4`).
- [ ] **Device 2:** Phone / Tablet / USB drive / Cloud Drive (Google Drive / iCloud / OneDrive).
- [ ] Test playback with audio on both devices before viva presentation!

