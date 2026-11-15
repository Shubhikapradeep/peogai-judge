| Exp | Model | Checkpoint | Quant | Backend | GPU | Calls | Time (min) | s/call | Sessions |
|---|---|---|---|---|---|---|---|---|---|
| exp1 | 3B | Qwen/Qwen2.5-Coder-3B-Instruct | fp16 | vllm | Tesla T4 | 300 | 8.5 | 1.71 | 1 |
| exp1 | 7B | Qwen/Qwen2.5-Coder-7B-Instruct-AWQ | 4-bit AWQ | vllm | Tesla T4 | 300 | 15.2 | 3.04 | 1 |
| exp2 | 3B | Qwen/Qwen2.5-Coder-3B-Instruct | fp16 | vllm | Tesla T4 | 300 | 9.6 | 1.93 | 1 |
| exp3 | 3B | Qwen/Qwen2.5-Coder-3B-Instruct | fp16 | vllm | Tesla T4 | 300 | 3.8 | 0.75 | 1 |
| exp3 | 7B | Qwen/Qwen2.5-Coder-7B-Instruct-AWQ | 4-bit AWQ | vllm | Tesla T4 | 300 | 6.8 | 1.36 | 1 |
| exp4b | 3B | Qwen/Qwen2.5-Coder-3B-Instruct | fp16 | vllm | Tesla T4 | 300 | 8.6 | 1.71 | 1 |
| exp4b | 7B | Qwen/Qwen2.5-Coder-7B-Instruct-AWQ | 4-bit AWQ | vllm | Tesla T4 | 300 | 15.2 | 3.04 | 1 |
| exp4c | 3B | Qwen/Qwen2.5-Coder-3B-Instruct | fp16 | vllm | Tesla T4 | 300 | 8.6 | 1.72 | 1 |

Total: 2400 judge calls, 1.27 GPU-hours
