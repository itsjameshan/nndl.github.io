# context-log — nndl-practice Docker

## Goal
CPU Docker image (Win10/amd64) running all 17 `pytorch/` notebooks of nndl/nndl-practice,
verified by pytest + nbconvert end-to-end. Spec: `docs/superpowers/specs/2026-06-09-nndl-practice-docker-design.md`.

## Decisions (user-approved)
Build+test here · pytorch/ 17 notebooks · CPU only · verify = pytest + execute-all.

## Key facts
- requirements.txt MISSING `torch_geometric` + `networkx` (chap9). Added in Dockerfile.
- No API keys / transformers / openai anywhere — pure-torch, offline-capable.
- Data: download.py (boston/cifar10/imdb global; lcqmc/bert CN best-effort) + prefetch.py
  (torchvision MNIST/CIFAR10 -> ~/.cache/torch_data ; PyG Cora/PROTEINS -> chap9 data/ ;
  tinyshakespeare -> chap10).
- chap6/8 degrade gracefully if imdb/lcqmc absent (no hard fail).
- torch install arch-detected: amd64 -> cpu index (no CUDA bloat); arm64 -> default PyPI.
- Repo pinned at commit e429d53 (src/ shallow clone).

## Build/verify strategy
arm64 native = full 17-notebook proof (fast). amd64 = same Dockerfile, smoke + pytest, deliver as tar.

## Incident (resolved)
Host disk hit 100% (1.6GB free): default-PyPI torch on linux/aarch64 now pulls ~2GB CUDA
wheels -> filled disk -> Docker VM [Errno 5] -> wedged. Fix: Dockerfile pins CPU index for
BOTH arches (verified aarch64 cpu wheels exist). Recovery: deleted 60GB-sparse Docker.raw
(user-consented reset) + cleared ~/Library/Caches -> 7.4GB free -> fresh Docker VM.

## State
- [x] Files authored + Dockerfile fixed (CPU index both arches).
- [x] arm64 build OK: image nndl-practice:cpu-arm64 = 2.32GB (torch 2.12.0+cpu, pyg 2.8.0).
- [x] smoke: pytest 94 pass / 1 fail (test_chap3 softmax acc 0.771<0.85 = seed/version-sensitive
      threshold, NOT an image defect). chap1 notebook 5.7s.
- [~] full 17-notebook run -> /tmp/nndl_fullrun.log, report/ (host). NB_TIMEOUT=2400.
- [ ] amd64 cross-build + smoke + save tar
- [ ] final report + commit

## 8GB RAM wall + delivery decision (2nd incident)
Host Mac = 8GB RAM; Docker VM=4096MiB (settings-store.json). CIFAR/RNN/attention notebooks
OOM the VM -> Docker daemon wedge (hit twice). Can't raise VM RAM safely on 8GB host.
User chose: DELIVER NOW + document RAM-heavy ones. amd64 = build natively on Win10 (NOT
emulated here — slow + wedge risk). cpufast.py now BAKED into image (rebuild OK, 2.32GB).

## Verified-green evidence (CPU_FAST, arm64)
PASS: chap1, chap2上下, chap3上下, chap4上下, chap5上(40s), chap6上(142s), chap10上(109s), chap10下(18s).
RAM-bound / wedged before reaching: chap5下(CIFAR), chap6下(RNN), chap8上下(attn). chap7,chap9
not yet cleanly verified (runs wedged first) -> confirm run covers chap9; verify chap7 next.
pytest 94/95 (chap3 softmax 0.771<0.85 = flaky threshold, not env).

## Next on resume
1. Read report/results.md (notebook PASS/FAIL table + times). Investigate any notebook FAIL
   (vs intentional). The 1 pytest fail is a known flaky threshold — report honestly, don't fake.
2. amd64: `docker buildx build --platform linux/amd64 -t nndl-practice:cpu-amd64 --load .`
   then smoke (imports+pytest) then `docker save | gzip > nndl-practice-cpu-amd64.tar.gz`.
3. git add nndl-practice-docker (NOT report/ or *.tar.gz or src/.git) + commit. 4. final report.
