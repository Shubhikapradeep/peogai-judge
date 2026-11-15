"""
app/web_demo.py -- Zero-dependency standalone web demo server (using standard library http.server).
Serves an interactive web application on http://localhost:7860 even without Gradio installed!
"""

from __future__ import annotations

import cgi
import http.server
import json
import os
import socketserver
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.app import (
    DEMO_EXAMPLES,
    FIGURES_DIR,
    TABLES_DIR,
    read_table_markdown,
    run_evaluation,
)

PORT = 7860

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PEOGAI - LLM-as-a-Judge Code Auditing Demo</title>
<style>
  :root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --danger: #dc2626;
    --success: #16a34a;
    --warning: #d97706;
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --text-muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg); color: var(--text); line-height: 1.5; padding: 24px; max-width: 1280px; margin: 0 auto; }
  header { margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
  h1 { font-size: 1.75rem; color: #1e293b; margin-bottom: 6px; }
  .subtitle { color: var(--text-muted); font-size: 0.95rem; }
  .tabs { display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 2px solid var(--border); }
  .tab-btn { padding: 10px 18px; border: none; background: none; font-weight: 600; cursor: pointer; color: var(--text-muted); font-size: 0.95rem; }
  .tab-btn.active { color: var(--primary); border-bottom: 3px solid var(--primary); margin-bottom: -2px; }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  label { display: block; font-weight: 600; margin-bottom: 6px; font-size: 0.9rem; color: #334155; }
  select, textarea, input { width: 100%; padding: 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; margin-bottom: 14px; }
  textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; resize: vertical; }
  button.primary { background: var(--primary); color: white; border: none; padding: 12px 24px; border-radius: 6px; font-weight: 600; cursor: pointer; width: 100%; font-size: 1rem; transition: background 0.15s; }
  button.primary:hover { background: var(--primary-dark); }
  .alert { padding: 14px 18px; border-radius: 8px; margin-bottom: 16px; border-left: 4px solid; }
  .alert-warning { background: #fffbeb; border-color: var(--warning); color: #92400e; }
  .alert-danger { background: #fef2f2; border-color: var(--danger); color: #991b1b; }
  .alert-success { background: #f0fdf4; border-color: var(--success); color: #166534; }
  .verdict-box { background: #f1f5f9; padding: 14px; border-radius: 8px; margin-bottom: 12px; }
  .verdict-box h4 { margin-bottom: 6px; }
  .stat-badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
  .badge-danger { background: #fee2e2; color: #b91c1c; }
  .badge-success { background: #dcfce7; color: #15803d; }
  table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.88rem; }
  th, td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
  th { background: #f1f5f9; }
  img.chart { width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; margin-bottom: 16px; }
  pre { background: #f8fafc; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.82rem; border: 1px solid var(--border); }
</style>
</head>
<body>
<header>
  <h1>🔍 Evaluative Benchmarking of LLM-as-a-Judge for Automated Code Auditing</h1>
  <div class="subtitle">Course: UE24AM343AB5 | Problem Statement 3 | Demonstration System</div>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('demo')">🧪 Interactive Dual-Order Judge</button>
  <button class="tab-btn" onclick="showTab('results')">📊 Benchmark Results & Findings</button>
  <button class="tab-btn" onclick="showTab('architecture')">🏗️ Architecture & Pipeline</button>
</div>

<!-- TAB 1: DEMO -->
<div id="tab-demo" class="tab-pane active">
  <div class="grid-2">
    <div class="card">
      <label for="example-select">⚡ Load Benchmark Case</label>
      <select id="example-select" onchange="loadCase()">
        <option value="0">Example 1 (Severe Position Bias Flip): Longest Subsequence Sum</option>
        <option value="1">Example 2 (Consistent Agreement): Beautiful Binary String</option>
        <option value="2">Example 3 (Algorithmic Complexity): Min Absolute Difference with Distance</option>
      </select>
      <div id="case-info" style="font-size:0.85rem; color:var(--text-muted); margin-bottom:12px;"></div>

      <label for="problem">Problem Description</label>
      <textarea id="problem" rows="7"></textarea>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
        <div>
          <label for="cand1">Candidate Solution 1</label>
          <textarea id="cand1" rows="12"></textarea>
        </div>
        <div>
          <label for="cand2">Candidate Solution 2</label>
          <textarea id="cand2" rows="12"></textarea>
        </div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px;">
        <div>
          <label for="backend">Backend Model</label>
          <select id="backend">
            <option value="ollama">Ollama (Qwen2.5-Coder-3B Local)</option>
            <option value="benchmark_replay" selected>Benchmark Replay / Offline (3B Empirical)</option>
            <option value="mock">Mock Backend (Pure Position Bias)</option>
          </select>
        </div>
        <div>
          <label for="prompt_strat">Prompt Strategy</label>
          <select id="prompt_strat">
            <option value="pairwise_v1" selected>Standard Pairwise (pairwise_v1)</option>
            <option value="pairwise_debias">Debias Prompt (pairwise_debias)</option>
          </select>
        </div>
      </div>

      <button class="primary" onclick="runEval()">🚀 Run Dual-Order Evaluation</button>
    </div>

    <div class="card">
      <h3 style="margin-bottom:14px;">Evaluation Output</h3>
      <div id="status-badge" class="alert alert-warning" style="display:none;"></div>

      <div id="fixed-verdict" class="verdict-box" style="display:none; border-left:4px solid var(--primary);"></div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
        <div id="order1-res" class="verdict-box">
          <h4>Order 1: [1 as A, 2 as B]</h4>
          <p style="color:var(--text-muted); font-size:0.85rem;">Click 'Run Dual-Order Evaluation' to execute.</p>
        </div>
        <div id="order2-res" class="verdict-box">
          <h4>Order 2: [2 as A, 1 as B]</h4>
          <p style="color:var(--text-muted); font-size:0.85rem;">Click 'Run Dual-Order Evaluation' to execute.</p>
        </div>
      </div>

      <div id="bias-explanation" style="margin-top:14px; font-size:0.9rem; line-height:1.6;"></div>
      <div id="meta-info" style="margin-top:12px; font-size:0.8rem; color:var(--text-muted);"></div>
    </div>
  </div>
</div>

<!-- TAB 2: RESULTS -->
<div id="tab-results" class="tab-pane">
  <div class="card" style="margin-bottom:20px;">
    <h2>Executive Findings from 2,400 Model Evaluations</h2>
    <ul style="margin: 14px 20px; line-height: 1.8;">
      <li><strong>Severe First-Position Bias (Exp 1):</strong> In pairwise evaluation, Qwen2.5-Coder-3B picked Position A in <strong>96.0%</strong> of calls (7B in <strong>91.7%</strong>). When correct code appeared as A, accuracy was <strong>98.7%</strong>; when placed as B, accuracy crashed to <strong>6.7%</strong> (+92.0 pp gap).</li>
      <li><strong>Widespread Order Disagreement:</strong> Consistency across orders was only <strong>8.0%</strong> for 3B and <strong>15.3%</strong> for 7B. In <strong>138 of 150 pairs (92.0%)</strong>, the 3B model flipped its pick simply because presentation order swapped (AA flip).</li>
      <li><strong>Dangerous False Acceptance in Pointwise Auditing (Exp 3):</strong> In single-code auditing, 3B exhibited a <strong>76.0% False Acceptance Rate (FAR)</strong> (72.0% for 7B). Leniency bias caused the judge to approve ~79% of all submissions. In production auditing, 3 out of 4 buggy programs would slip past.</li>
      <li><strong>Order-Averaging is the Only Robust Mitigation (Exp 4a):</strong> Requiring both orders to agree boosts accuracy to <strong>83.3%</strong> (3B) and eliminates position bias by construction, at the cost of coverage (8.0% decided).</li>
      <li><strong>Prompting & Sampling Fixes Fail (Exp 4b, 4c):</strong> Instructing the model to ignore order did not reduce bias (%A remained 98.3%, p=0.453). Sampling 3 completions at T=0.7 did not improve consistency (p=0.555), proving the bias is <strong>systematic representation bias, not stochastic noise</strong>.</li>
    </ul>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>Fig 1: Position Bias Gap (A vs B)</h3>
      <img class="chart" src="/figures/f1_position_bias.png" alt="Fig 1 Position Bias">
    </div>
    <div class="card">
      <h3>Fig 2: Pointwise Auditing Error Rates (FAR & FRR)</h3>
      <img class="chart" src="/figures/f2_pointwise.png" alt="Fig 2 Pointwise">
    </div>
    <div class="card">
      <h3>Fig 3: Bias Mitigation Comparison</h3>
      <img class="chart" src="/figures/f3_fixes.png" alt="Fig 3 Fixes">
    </div>
    <div class="card">
      <h3>Fig 4: Verbosity Perturbation Effect</h3>
      <img class="chart" src="/figures/f4_verbosity.png" alt="Fig 4 Verbosity">
    </div>
  </div>
</div>

<!-- TAB 3: ARCHITECTURE -->
<div id="tab-architecture" class="tab-pane">
  <div class="card">
    <h2>System Design & Evaluation Pipeline</h2>
    <pre>
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
(results/metrics.py)             Order-Averaged Filter
   │
   ▼
IEEE Report & Slide Deck
    </pre>
  </div>
</div>

<script>
const examples = EXAMPLES_DATA_JSON;

function showTab(tabName) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + tabName).classList.add('active');
  event.target.classList.add('active');
}

function loadCase() {
  const idx = document.getElementById('example-select').value;
  const ex = examples[idx];
  if (!ex) return;
  document.getElementById('problem').value = ex.problem;
  document.getElementById('cand1').value = ex.candidate_1;
  document.getElementById('cand2').value = ex.candidate_2;
  document.getElementById('case-info').innerHTML = `<strong>Category:</strong> ${ex.category} | <strong>Expected:</strong> ${ex.expected_behavior}`;
}

async function runEval() {
  const payload = {
    problem: document.getElementById('problem').value,
    candidate_1: document.getElementById('cand1').value,
    candidate_2: document.getElementById('cand2').value,
    backend: document.getElementById('backend').value,
    prompt_strategy: document.getElementById('prompt_strat').value
  };

  const badge = document.getElementById('status-badge');
  badge.style.display = 'block';
  badge.className = 'alert alert-warning';
  badge.innerText = 'Evaluating in both orders...';

  try {
    const res = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    badge.className = data.bias_detected ? 'alert alert-danger' : 'alert alert-success';
    badge.innerHTML = `<strong>${data.status_box}</strong>`;

    document.getElementById('fixed-verdict').style.display = 'block';
    document.getElementById('fixed-verdict').innerHTML = data.fixed_verdict.replace(/### /g, '<h3>').replace(/\n/g, '<br>');

    document.getElementById('order1-res').innerHTML = data.ord1_display.replace(/### /g, '<h4>').replace(/\n/g, '<br>');
    document.getElementById('order2-res').innerHTML = data.ord2_display.replace(/### /g, '<h4>').replace(/\n/g, '<br>');

    document.getElementById('bias-explanation').innerHTML = data.bias_status.replace(/### /g, '<h4>').replace(/\n/g, '<br>');
    document.getElementById('meta-info').innerText = data.meta_metrics.replace(/\\*\\*/g, '').replace(/`/g, '');
  } catch(err) {
    badge.className = 'alert alert-danger';
    badge.innerText = 'Error running evaluation: ' + err;
  }
}

// Initial load
window.onload = loadCase;
</script>
</body>
</html>
"""


class DemoHTTPHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            html = HTML_PAGE.replace("EXAMPLES_DATA_JSON", json.dumps(DEMO_EXAMPLES))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif url.path.startswith("/figures/"):
            fname = os.path.basename(url.path)
            img_path = FIGURES_DIR / fname
            if img_path.exists() and img_path.suffix.lower() == ".png":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(img_path.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/evaluate":
            length = int(self.headers.get("content-length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)

            status_box, bias_status, fixed_verdict, ord1_display, ord2_display, meta_metrics = run_evaluation(
                data.get("problem", ""),
                data.get("candidate_1", ""),
                data.get("candidate_2", ""),
                data.get("backend", "benchmark_replay"),
                data.get("prompt_strategy", "pairwise_v1")
            )

            res = {
                "status_box": status_box,
                "bias_detected": "Position Bias Detected" in status_box,
                "bias_status": bias_status,
                "fixed_verdict": fixed_verdict,
                "ord1_display": ord1_display,
                "ord2_display": ord2_display,
                "meta_metrics": meta_metrics,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress routine log clutter
        pass


def run_standalone_server(port=PORT):
    with socketserver.TCPServer(("", port), DemoHTTPHandler) as httpd:
        print(f"[*] Standalone Demo Server running at http://localhost:{port}")
        print("    Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server stopped.")


if __name__ == "__main__":
    run_standalone_server()
