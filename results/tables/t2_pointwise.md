**Exp 3: pointwise judging of all 300 answers**

| Model | Accuracy | FAR | FRR | % said correct | Both of pair right | Parse fail |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | 53.7 [48, 59] | 82.0 [75, 87] | 10.7 [7, 17] | 85.7 | 13.3 | 0.0 |
| Qwen2.5-Coder-7B (4-bit) | 56.3 [51, 62] | 72.0 [64, 79] | 15.3 [10, 22] | 78.3 | 18.0 | 0.0 |

_Values in %. Brackets: 95% Wilson CI. Parse failures count as wrong. FAR: wrong code accepted. FRR: correct code rejected._
