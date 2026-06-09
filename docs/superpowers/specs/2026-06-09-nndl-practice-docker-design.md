# nndl-practice CPU Docker image for Windows 10 — design

Date: 2026-06-09

## Goal

A self-contained Docker image that runs every notebook in the `pytorch/` part of
[nndl/nndl-practice](https://github.com/nndl/nndl-practice) (17 notebooks, 10 chapters),
usable on a Windows 10 machine via Docker Desktop + WSL2, plus a test harness that proves
all 17 run end-to-end.

## Verified facts (from repo inspection at commit `e429d53`)

- `pytorch/` has 17 runnable notebooks; `legacy/` has student stubs (out of scope).
- Dependency truth across all 17 notebooks:
  - Covered by `pytorch/requirements.txt`: torch, torchvision, numpy, pandas, matplotlib,
    scikit-learn, jupyter, nbconvert, pytest.
  - **Missing → added by this image:** `torch_geometric`, `networkx` (chap9 only).
  - No API keys, no `transformers`, no `openai`. Every model is built from scratch in pure
    torch ⇒ fully offline-capable once deps + data are baked.
- Data (gitignored in the repo) is fetched by `pytorch/datasets/download.py` (boston, cifar10
  raw batches, imdb, lcqmc, optional bert vocab) plus torchvision MNIST/CIFAR10
  (`~/.cache/torch_data`), PyG Cora/PROTEINS (chap9 `data/`), and tinyshakespeare (chap10).
- chap6/chap8 degrade gracefully when imdb/lcqmc are absent (they print a notice and skip,
  they do not error). bert vocab is optional (char-level fallback).

## Decisions (locked with user)

1. Build **and** test here on this Mac; hand off a proven image.
2. Scope: `pytorch/` — all 17 notebooks. Legacy stubs excluded.
3. Compute: **CPU only** (portable to any Win10 Docker Desktop; no GPU required).
4. Verify depth: **both** — repo `pytest` sanity suite *and* nbconvert end-to-end execution
   of every notebook.

## Architecture

Build context: `nndl-practice-docker/`

```
nndl-practice-docker/
  src/                  # shallow clone of nndl-practice @ e429d53 (code only, no data)
  Dockerfile            # python:3.11-slim, arch-detected CPU torch, repo + datasets baked
  prefetch.py           # bakes torchvision MNIST/CIFAR10, PyG Cora/PROTEINS, tinyshakespeare
  run_tests.sh          # pytest + nbconvert-execute all 17 -> /workspace/report
  docker-compose.yml    # `up` = Jupyter Lab :8888 ; `run test` = full verify
  .dockerignore
  README.md             # Windows 10 usage + how to switch to GPU
build_and_test.sh       # host: build arm64 (verify) + amd64 (deliver), save image tar
```

### Image (Dockerfile)

- Base `python:3.11-slim-bookworm` + `ca-certificates`, `libgomp1` (torch runtime).
- torch install is arch-detected: `x86_64` → `--index-url download.pytorch.org/whl/cpu`
  (CPU-only, avoids the ~2GB CUDA wheel); `aarch64` → default PyPI (already CPU-only).
- `pip install -r requirements.txt` + `torch_geometric networkx jupyterlab`.
- `COPY src/` then bake data:
  - `download.py --only=boston,cifar10,imdb` (global mirrors, must succeed).
  - `download.py --only=lcqmc,bert_vocab || true` (CN mirrors, best-effort; chap8 degrades).
  - `prefetch.py` (torchvision + PyG + tinyshakespeare).
- CMD = Jupyter Lab on `0.0.0.0:8888`, token disabled, workdir `pytorch/`.

### Test harness (run_tests.sh, in-container)

1. `pytest pytorch/tests -q` (the maintainer's 10 sanity files).
2. Execute each of the 17 notebooks via `jupyter nbconvert --to notebook --execute`
   (output to throwaway, kernel cwd = notebook dir so `../datasets` resolves). Capture
   PASS/FAIL + wall-time → `report/results.md` + per-notebook logs. Exit nonzero if any fail.

### Build + verify strategy

Notebook code is arch-independent (pure Python/torch). So:

- **arm64 native** image → run the full 17-notebook suite + pytest at native speed = the real
  proof.
- **amd64** image (Win10) → same Dockerfile; smoke-verify boot + imports + pytest + a couple
  notebooks. Identical recipe + CPU torch ⇒ same result on Win10.

## Risks / honest limits

1. DL notebooks (chap5/6/8/9/10) train on CPU; could be slow even native. Generous timeouts,
   real times reported. Anything > ~30 min is flagged before touching epochs. No faked passes.
2. CN mirrors (lcqmc/bert vocab) may be slow/unreachable from here; chap8 still runs (graceful
   degrade + char-level fallback).
3. Delivered amd64 image is smoke-verified, not full-run, unless the slow QEMU path is requested.

## Out of scope

- GPU/CUDA image (documented in README as a swap, not built/tested).
- `legacy/` notebooks (intentional student stubs).
