# scripts/download_reranker.py
"""下载本地精排模型(ONNX)到 cache/。

默认下载 bge-reranker-base(多语言, 中文精排效果最佳, ~279MB),
经 hf-mirror 加速。用法:
    python scripts/download_reranker.py            # bge-reranker-base
    python scripts/download_reranker.py ms-marco   # ms-marco-MiniLM-L-6-v2(英文, ~91MB)

模型目录不入 git(cache/ 已 gitignore), 换机器后先跑本脚本。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIRROR = "https://hf-mirror.com"

# (repo, 本地目录, onnx 文件名)
MODELS = {
    "bge": (
        "Xenova/bge-reranker-base",
        ROOT / "cache" / "bge-reranker-base",
        "onnx/model_quantized.onnx",
    ),
    "ms-marco": (
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ROOT / "cache" / "ms-marco-onnx",
        "onnx/model.onnx",
    ),
}

TOKENIZER_FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt",
]


def download(repo: str, dest: Path, onnx_path: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "onnx").mkdir(exist_ok=True)
    base = f"{MIRROR}/{repo}/resolve/main"

    def _curl(url: str, out: Path) -> None:
        print(f"  ↓ {url.split('/resolve/')[0].split('/')[-1]}/{out.name}")
        subprocess.run(
            ["curl", "-sL", "--max-time", "900", "-o", str(out), url],
            check=True,
        )

    _curl(f"{base}/{onnx_path}", dest / "onnx" / Path(onnx_path).name)
    for f in TOKENIZER_FILES:
        out = dest / f
        if not out.exists():
            _curl(f"{base}/{f}", out)
    print(f"✅ {repo} → {dest}")


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "bge"
    if key not in MODELS:
        print(f"可选: {list(MODELS)}"); sys.exit(1)
    repo, dest, onnx_path = MODELS[key]
    download(repo, dest, onnx_path)


if __name__ == "__main__":
    main()
