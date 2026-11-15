"""
judge.py -- single entry point for every LLM-as-a-judge call in the project.

    from src.judge import load_backend, judge
    backend = load_backend("hf", "3b")              # or "vllm", "ollama", "mock"
    out = judge(backend, "pairwise_v1", problem, code_a=a, code_b=b)
    out -> {"verdict": "A" | "B" | None, "raw": "...", "parse_ok": True, ...}

Prompt types (file in prompts/<type>.txt; built-in fallback if missing):
    pairwise_v1, pairwise_debias  -> verdict "A" / "B"
    pointwise_v1                  -> verdict "correct" / "incorrect"

Prompt placeholders: {problem}, {answer_a}, {answer_b} (pairwise), {code} (pointwise).
The demo (app/app.py) uses this same module with backend "ollama".
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts"

# ---------------------------------------------------------------- models
MODELS = {
    # key: (HF id for transformers+bnb, HF id for vLLM, Ollama tag)
    "3b": ("Qwen/Qwen2.5-Coder-3B-Instruct",
           "Qwen/Qwen2.5-Coder-3B-Instruct",
           "qwen2.5-coder:3b"),
    "7b": ("Qwen/Qwen2.5-Coder-7B-Instruct",          # loaded as 4-bit NF4 via bitsandbytes
           "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",      # 4-bit AWQ for vLLM
           "qwen2.5-coder:7b"),
}

SYSTEM_MSG = "You are a strict code reviewer. Follow the output format exactly."

# ---------------------------------------------------------------- prompts
# Fallbacks so P1 can run before P2's prompt files land. P2's files override these.
DEFAULT_PROMPTS = {
    "pairwise_v1": """You are given a programming problem and two candidate solutions, Answer A and Answer B.
Exactly one of them is correct (passes all hidden unit tests).

### Problem
{problem}

### Answer A
{answer_a}

### Answer B
{answer_b}

Which answer is correct? Output only a single letter: A or B.""",

    "pairwise_debias": """You are given a programming problem and two candidate solutions, Answer A and Answer B.
Exactly one of them is correct (passes all hidden unit tests).
Important: the order in which the answers appear is random and carries no information.
Do not prefer an answer because it comes first or second, or because it is longer, has more
comments, or looks more polished. Judge only whether the logic solves the problem for all inputs.

### Problem
{problem}

### Answer A
{answer_a}

### Answer B
{answer_b}

Which answer is correct? Output only a single letter: A or B.""",

    "pointwise_v1": """You are given a programming problem and one candidate solution.
Decide whether the solution is correct, i.e. it would pass all hidden unit tests.

### Problem
{problem}

### Candidate solution
{code}

Output only one word: correct or incorrect.""",
}


def prompt_kind(prompt_type: str) -> str:
    return "pointwise" if prompt_type.startswith("pointwise") else "pairwise"


def load_prompt(prompt_type: str) -> str:
    f = PROMPT_DIR / f"{prompt_type}.txt"
    if f.exists():
        return f.read_text()
    if prompt_type in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[prompt_type]
    raise FileNotFoundError(f"No prompt file {f} and no built-in default for '{prompt_type}'")


def build_prompt(prompt_type: str, problem: str, code_a: str = None, code_b: str = None,
                 code: str = None) -> str:
    tpl = load_prompt(prompt_type)
    # str.replace, not .format(): code is full of { } braces.
    out = tpl.replace("{problem}", problem.strip())
    if prompt_kind(prompt_type) == "pairwise":
        out = out.replace("{answer_a}", code_a.strip()).replace("{answer_b}", code_b.strip())
    else:
        out = out.replace("{code}", (code if code is not None else code_a).strip())
    return out


# ---------------------------------------------------------------- parsing
_PAIR_PATTERNS = [
    r"^\W*([AB])\W*$",                                      # "A", "**B**", "(A)."
    r"^\W*(?:answer)?\s*([AB])\b",                          # "A is correct", "Answer B ..."
    r"(?:answer|verdict|choice|correct(?: one| answer)?)\s*(?:is|:)?\s*\**\(?([AB])\b",
]


def parse_pairwise(text: str):
    t = text.strip()
    for p in _PAIR_PATTERNS:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # last resort: exactly one distinct standalone letter in the whole output
    letters = set(re.findall(r"\b([AB])\b", t))
    if len(letters) == 1:
        return letters.pop()
    return None


def parse_pointwise(text: str):
    t = text.strip().lower()
    first = re.match(r"^\W*(incorrect|correct|wrong|right|yes|no)\b", t)
    if first:
        w = first.group(1)
        return "correct" if w in ("correct", "right", "yes") else "incorrect"
    has_inc = re.search(r"\b(incorrect|not correct|wrong)\b", t)
    has_cor = re.search(r"(?<!in)\bcorrect\b", t) and not re.search(r"not correct", t)
    if has_inc and not has_cor:
        return "incorrect"
    if has_cor and not has_inc:
        return "correct"
    return None


def parse_verdict(prompt_type: str, text: str):
    return parse_pointwise(text) if prompt_kind(prompt_type) == "pointwise" else parse_pairwise(text)


# ---------------------------------------------------------------- backends
class Backend:
    name = "base"
    model_id = ""

    def generate(self, prompts: list[str], temperature: float = 0.0, n: int = 1,
                 max_new_tokens: int = 16, seed: int = 42) -> list[list[str]]:
        """Return one list of n completions per prompt."""
        raise NotImplementedError


class HFBackend(Backend):
    """transformers + bitsandbytes. 7B is loaded in 4-bit NF4; 3B in fp16."""
    name = "hf"

    def __init__(self, model_key: str, four_bit: bool | None = None, max_input_tokens: int = 16384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.model_id = MODELS[model_key][0]
        four_bit = (model_key == "7b") if four_bit is None else four_bit
        self.quant = "4-bit NF4 (bitsandbytes)" if four_bit else "fp16"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map="auto", torch_dtype=torch.float16)
        if four_bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw).eval()
        self.max_input_tokens = max_input_tokens

    def generate(self, prompts, temperature=0.0, n=1, max_new_tokens=16, seed=42):
        torch = self.torch
        texts = [self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        # No truncation: cutting the prompt would drop the final output instruction.
        # Qwen2.5-Coder handles 32k context; longest pair here is ~7k tokens.
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        if enc["input_ids"].shape[1] > self.max_input_tokens:
            print(f"warning: prompt of {enc['input_ids'].shape[1]} tokens exceeds {self.max_input_tokens}")
        torch.manual_seed(seed)
        gen_kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                      num_return_sequences=n)
        if temperature > 0:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kw)
        new = out[:, enc["input_ids"].shape[1]:]
        dec = self.tok.batch_decode(new, skip_special_tokens=True)
        return [dec[i * n:(i + 1) * n] for i in range(len(prompts))]


class VLLMBackend(Backend):
    """vLLM. 7B uses the official AWQ 4-bit checkpoint."""
    name = "vllm"

    def __init__(self, model_key: str, max_model_len: int = 12288, gpu_mem: float = 0.90):
        from vllm import LLM
        self.model_id = MODELS[model_key][1]
        self.quant = "4-bit AWQ" if "AWQ" in self.model_id else "fp16"
        self.llm = LLM(model=self.model_id, dtype="float16", max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_mem,
                       quantization="awq" if "AWQ" in self.model_id else None)

    def generate(self, prompts, temperature=0.0, n=1, max_new_tokens=16, seed=42):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                            max_tokens=max_new_tokens, seed=seed)
        msgs = [[{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": p}]
                for p in prompts]
        outs = self.llm.chat(msgs, sp, use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]


class OllamaBackend(Backend):
    """Local Ollama server (demo on the MacBook). `ollama pull qwen2.5-coder:3b` first."""
    name = "ollama"

    def __init__(self, model_key: str = "3b", host: str = None):
        self.model_id = MODELS[model_key][2]
        self.quant = "Ollama default (Q4_K_M GGUF)"
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def _one(self, prompt, temperature, max_new_tokens, seed):
        body = json.dumps({
            "model": self.model_id, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_MSG},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "num_predict": max_new_tokens,
                        "seed": seed, "num_ctx": 12288},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"]

    def generate(self, prompts, temperature=0.0, n=1, max_new_tokens=16, seed=42):
        return [[self._one(p, temperature, max_new_tokens, seed + k) for k in range(n)]
                for p in prompts]


class MockBackend(Backend):
    """No GPU. Always answers 'A' / 'correct' (a maximally position-biased judge).
    Use it to dry-run the pipeline and resume logic."""
    name = "mock"

    def __init__(self, model_key: str = "mock"):
        self.model_id = f"mock-{model_key}"
        self.quant = "none"

    def generate(self, prompts, temperature=0.0, n=1, max_new_tokens=16, seed=42):
        return [["correct" if "Candidate solution" in p else "A"] * n for p in prompts]


def load_backend(backend: str, model_key: str = "3b", **kw) -> Backend:
    return {"hf": HFBackend, "vllm": VLLMBackend, "ollama": OllamaBackend,
            "mock": MockBackend}[backend](model_key, **kw)


# ---------------------------------------------------------------- public API
def judge(backend: Backend, prompt_type: str, problem: str, code_a: str = None,
          code_b: str = None, code: str = None, temperature: float = 0.0, n: int = 1,
          max_new_tokens: int = 16) -> dict:
    """Judge one item. For n > 1 also returns the majority verdict."""
    return judge_batch(backend, prompt_type,
                       [dict(problem=problem, code_a=code_a, code_b=code_b, code=code)],
                       temperature=temperature, n=n, max_new_tokens=max_new_tokens)[0]


def judge_batch(backend: Backend, prompt_type: str, items: list[dict], temperature: float = 0.0,
                n: int = 1, max_new_tokens: int = 16, seed: int = 42) -> list[dict]:
    prompts = [build_prompt(prompt_type, **it) for it in items]
    t0 = time.time()
    gens = backend.generate(prompts, temperature=temperature, n=n,
                            max_new_tokens=max_new_tokens, seed=seed)
    dt = (time.time() - t0) / max(len(prompts), 1)
    results = []
    for samples in gens:
        verdicts = [parse_verdict(prompt_type, s) for s in samples]
        res = {"verdict": verdicts[0], "raw": samples[0], "parse_ok": verdicts[0] is not None,
               "sec_per_call": round(dt, 3)}
        if n > 1:
            valid = [v for v in verdicts if v is not None]
            top = Counter(valid).most_common()
            # majority over parsed samples; a tie (incl. 1-1 with one failure) -> None
            maj = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else None
            res.update(verdict=maj, parse_ok=maj is not None, samples=verdicts, raw_samples=samples)
        results.append(res)
    return results


def judge_both_orders(backend: Backend, problem: str, code_1: str, code_2: str,
                      prompt_type: str = "pairwise_v1") -> dict:
    """Demo helper: run (1,2) and (2,1); map verdicts back to answer identity;
    order-averaged verdict is defined only when both orders agree."""
    r12, r21 = judge_batch(backend, prompt_type, [
        dict(problem=problem, code_a=code_1, code_b=code_2),
        dict(problem=problem, code_a=code_2, code_b=code_1)])
    pick12 = {"A": 1, "B": 2}.get(r12["verdict"])
    pick21 = {"A": 2, "B": 1}.get(r21["verdict"])
    consistent = pick12 is not None and pick12 == pick21
    return {"order_12": r12, "order_21": r21, "pick_order_12": pick12, "pick_order_21": pick21,
            "position_bias": (pick12 is not None and pick21 is not None and pick12 != pick21),
            "final": pick12 if consistent else None}
