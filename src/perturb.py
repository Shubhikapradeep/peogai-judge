"""
perturb.py -- Exp 2 input: pad the INCORRECT answer with a header comment and
step comments, without changing its logic.

    python -m src.perturb --data data/pairs_150.jsonl --out data/pairs_150_verbose.jsonl

Safety check: only comment lines are inserted, and the AST of the padded code must equal
the AST of the original (comments never reach the AST). If that check fails, or the
original doesn't parse, we fall back to the header comment only (still logic-preserving).
"""
import argparse
import ast
import json
import re

HEADER = [
    "# Solution overview:",
    "# This implementation carefully handles all edge cases, including empty input,",
    "# single-element cases and the maximum constraints given in the problem.",
    "# Time complexity is optimal for the given limits; the approach has been",
    "# verified against the sample cases and additional corner cases.",
]
STEP_COMMENTS = [
    (r"def ", "# Helper/entry function: validated against all constraints"),
    (r"for |while ", "# Iterate over the input; loop bounds checked for off-by-one errors"),
    (r"if |elif ", "# Handle this case explicitly to cover edge conditions"),
    (r"return ", "# Return the verified final result"),
]


def same_ast(a, b):
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return False


def pad(code: str) -> str:
    out = HEADER[:] + [""]
    in_string = False
    for line in code.split("\n"):
        # crude triple-quote tracking: never insert inside a multi-line string
        if not in_string:
            stripped = line.lstrip()
            for pat, comment in STEP_COMMENTS:
                if re.match(pat, stripped):
                    out.append(line[: len(line) - len(stripped)] + comment)
                    break
        out.append(line)
        if (line.count('"""') + line.count("'''")) % 2 == 1:
            in_string = not in_string
    padded = "\n".join(out)
    if same_ast(code, padded):
        return padded
    return "\n".join(HEADER) + "\n\n" + code  # header-only fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pairs_150.jsonl")
    ap.add_argument("--out", default="data/pairs_150_verbose.jsonl")
    args = ap.parse_args()
    n = full = 0
    with open(args.data) as f, open(args.out, "w") as g:
        for line in f:
            r = json.loads(line)
            r["incorrect_verbose"] = pad(r["incorrect"])
            full += r["incorrect_verbose"].count("\n") > len(HEADER) + r["incorrect"].count("\n") + 1
            n += 1
            g.write(json.dumps(r) + "\n")
    print(f"wrote {n} rows to {args.out} ({full} with step comments, {n - full} header-only)")


if __name__ == "__main__":
    main()
