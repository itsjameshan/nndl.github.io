#!/usr/bin/env bash
# Host orchestration.
#
#   ./build_and_test.sh            # build the image + CPU_FAST-verify all 17 notebooks
#   ./build_and_test.sh quick      # build + verify a fast low-RAM subset only
#   ./build_and_test.sh amd64      # ALSO cross-build the linux/amd64 image + save a tarball
#
# Notes for this machine (8 GB RAM Apple Silicon):
#   - The native build + CPU_FAST verify is the supported path here.
#   - Give Docker Desktop >= 6 GB (Settings > Resources) or the CIFAR/RNN/attention
#     notebooks will OOM the VM.
#   - The amd64 path cross-builds under QEMU emulation: slow and memory-hungry here.
#     On the actual Windows 10 box just run `docker compose build` instead — it builds
#     amd64 natively in ~10-15 min with no emulation.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-full}"
TAG=nndl-practice:cpu-arm64
mkdir -p report

echo "== build (native) =="
docker build -t "$TAG" .

echo "== verify (CPU_FAST) =="
if [ "$MODE" = "quick" ]; then
  docker run --rm -e CPU_FAST=1 -e SKIP_PYTEST=1 -e NB_TIMEOUT=600 \
    -e ONLY="实践基础 线性模型 图神经网络 大语言模型与智能体" \
    -v "$PWD/report:/workspace/report" "$TAG" bash /workspace/run_tests.sh
else
  docker run --rm -e CPU_FAST=1 -e NB_TIMEOUT=1500 \
    -v "$PWD/report:/workspace/report" "$TAG" bash /workspace/run_tests.sh
fi
echo "report written to ./report/results.md"

if [ "$MODE" = "amd64" ]; then
  echo "== cross-build linux/amd64 (emulated — slow) =="
  docker buildx build --platform linux/amd64 -t nndl-practice:cpu-amd64 --load .
  echo "== smoke-test amd64 =="
  docker run --rm --platform linux/amd64 nndl-practice:cpu-amd64 bash -c \
    "python -c 'import torch,torchvision,torch_geometric,networkx,sklearn,pandas,matplotlib;print(\"imports OK\",torch.__version__)' \
     && cd pytorch && python -m pytest tests -q"
  echo "== save amd64 image -> nndl-practice-cpu-amd64.tar.gz =="
  docker save nndl-practice:cpu-amd64 | gzip > nndl-practice-cpu-amd64.tar.gz
  ls -lh nndl-practice-cpu-amd64.tar.gz
fi
