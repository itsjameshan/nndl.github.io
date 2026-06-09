"""Make a notebook finish quickly on CPU by shrinking ONLY training-length knobs
(epoch counts, iteration/step caps, and explicit training for-loops). Model shapes
(n_layer, n_embd, hidden_size, batch_size, block_size, ...) are left untouched, so
tensor shapes and asserts that depend on them still hold.

Usage: python cpufast.py <input.ipynb> <output.ipynb>

The original notebooks are never modified; run_tests.sh writes a throwaway copy
beside each notebook (so the kernel's cwd still resolves ../datasets) and executes
that. This is a CPU smoke-run: it proves every notebook runs end-to-end. Models are
under-trained on purpose; for real results use a GPU and the unmodified notebooks.
"""
import json
import re
import sys

CAP_EPOCHS = 2     # epoch-style knobs -> at most this many
CAP_ITERS = 30     # iteration/step knobs -> at most this many
RANGE_MIN = 150    # only shrink training for-loops with at least this many steps
RANGE_CAP = 30

EPOCH_NAMES = r"(?:num_epochs|n_epochs|max_epochs|epochs)"
ITER_NAMES = (
    r"(?:max_iters|max_iter|max_steps|num_steps|n_steps|num_iters|n_iters|"
    r"total_steps|num_training_steps|train_steps)"
)
# training-loop variable names — `for step in range(500)` etc.
LOOP_VARS = r"(?:step|steps|epoch|epochs|it|iter|iteration|t)"


def _cap_kw(m, cap):
    return f"{m.group('k')}{m.group('eq')}{min(int(m.group('v')), cap)}"


def _cap_range(m):
    n = int(m.group("n"))
    if n >= RANGE_MIN:
        return f"{m.group('pre')}{min(n, RANGE_CAP)})"
    return m.group(0)


def transform(src: str) -> str:
    src = re.sub(rf"(?P<k>\b{EPOCH_NAMES})(?P<eq>\s*=\s*)(?P<v>\d+)",
                 lambda m: _cap_kw(m, CAP_EPOCHS), src)
    src = re.sub(rf"(?P<k>\b{ITER_NAMES})(?P<eq>\s*=\s*)(?P<v>\d+)",
                 lambda m: _cap_kw(m, CAP_ITERS), src)
    src = re.sub(rf"(?P<pre>\bfor\s+{LOOP_VARS}\s+in\s+range\(\s*)(?P<n>\d+)\s*\)",
                 _cap_range, src)
    return src


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        s = cell["source"]
        joined = s if isinstance(s, str) else "".join(s)
        cell["source"] = transform(joined)
        cell["outputs"] = []
        cell["execution_count"] = None
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
