"""Bake the datasets that aren't covered by datasets/download.py so the image
runs fully offline:

  - torchvision MNIST + CIFAR10  -> ~/.cache/torch_data  (chap5 uses download=True)
  - PyG Cora + PROTEINS          -> chap9 data/          (notebook root='data/...')
  - tinyshakespeare.txt          -> chap10 dir           (notebook fetches if absent)

Run inside the image during build, from anywhere. Fails loud: these come from
global mirrors and are required for chap5 / chap9 / chap10 to actually execute.
"""
import os
import urllib.request
from pathlib import Path

PYT = Path("/workspace/nndl-practice/pytorch")


def chapdir(prefix: str) -> Path:
    hits = sorted(PYT.glob(prefix + "*"))
    if not hits:
        raise SystemExit(f"[prefetch] no chapter dir matching {prefix!r}")
    return hits[0]


# 1) torchvision MNIST + CIFAR10 into the cache path the notebooks use.
cache = os.path.expanduser("~/.cache/torch_data")
os.makedirs(cache, exist_ok=True)
from torchvision import datasets  # noqa: E402

print("[prefetch] MNIST ...", flush=True)
datasets.MNIST(cache, train=True, download=True)
datasets.MNIST(cache, train=False, download=True)
print("[prefetch] CIFAR10 ...", flush=True)
datasets.CIFAR10(cache, train=True, download=True)
datasets.CIFAR10(cache, train=False, download=True)

# 2) PyG datasets. The chap9 notebook uses root='data/Planetoid' relative to its
#    own directory, so download with that cwd.
c9 = chapdir("chap9")
os.chdir(c9)
print(f"[prefetch] PyG Cora + PROTEINS into {c9}/data ...", flush=True)
from torch_geometric.datasets import Planetoid, TUDataset  # noqa: E402

Planetoid(root="data/Planetoid", name="Cora")
TUDataset(root="data/TUDataset", name="PROTEINS")

# 3) tinyshakespeare for chap10.
c10 = chapdir("chap10")
dest = c10 / "tinyshakespeare.txt"
if not dest.exists():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("[prefetch] tinyshakespeare ...", flush=True)
    urllib.request.urlretrieve(url, dest)

print("[prefetch] done", flush=True)
