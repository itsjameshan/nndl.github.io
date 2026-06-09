# syntax=docker/dockerfile:1
# 《神经网络与深度学习：案例与实践》(第二版) PyTorch — CPU image that runs all 17
# pytorch/ notebooks of nndl/nndl-practice offline (legacy/ and PaddlePaddle are out of scope).
# Builds for both linux/amd64 (Windows 10 target) and linux/arm64 (local verify).
FROM python:3.11-slim-bookworm

# torch needs libgomp1 at runtime; ca-certificates for dataset downloads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace/nndl-practice

# 1) CPU-only PyTorch for BOTH arches. The default PyPI wheel now bundles ~2GB of
#    NVIDIA CUDA libs on linux/aarch64 as well as x86_64, so always pin to the CPU
#    index (it carries both x86_64 and aarch64 cp311 cpu wheels).
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.7" "torchvision>=0.22"

# 2) Repo requirements + the two deps chap9 needs that requirements.txt omits.
COPY src/pytorch/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install "torch_geometric>=2.5" networkx jupyterlab

# 3) Repo code (datasets are gitignored; baked below).
COPY src/ /workspace/nndl-practice/

# 4) Bake datasets so the image runs with no network.
#    Global mirrors must succeed; CN mirrors are best-effort (chap8 degrades gracefully).
RUN python pytorch/datasets/download.py --only=boston,cifar10,imdb
RUN python pytorch/datasets/download.py --only=lcqmc,bert_vocab || true
COPY prefetch.py /workspace/prefetch.py
RUN python /workspace/prefetch.py

COPY run_tests.sh /workspace/run_tests.sh
COPY cpufast.py /workspace/cpufast.py
RUN chmod +x /workspace/run_tests.sh

EXPOSE 8888
WORKDIR /workspace/nndl-practice/pytorch
# Default: Jupyter Lab, no token (bind to localhost on the host side).
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
