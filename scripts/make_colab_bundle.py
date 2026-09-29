"""Colab GPU 학습용 묶음(zip)을 만든다 (docs/colab.md).

코드(tripin_ai, scripts/finetune.py)와 학습에 필요한 데이터만 담는다.
⚠️ AI Hub 데이터가 들어 있다. 본인 Colab·Google Drive에서만 쓰고 공유·공개하지 않는다(재배포 금지).

사용 예:
  python scripts/make_colab_bundle.py   → data/interim/colab/tripin_colab.zip
"""
import zipfile
from pathlib import Path

FILES = [
    *Path("tripin_ai").rglob("*.py"),
    Path("scripts/finetune.py"),
    Path("pyproject.toml"),
    Path("data/interim/eval/attractions.csv"),
    Path("data/interim/eval/travelers.csv"),
    Path("data/interim/eval/visits.csv"),
    Path("data/interim/emb/mminilm-l12-v1/attractions_v0.csv"),
    Path("data/interim/emb/mminilm-l12-v1/attractions_v1.csv"),   # 설명 포함 (백엔드 DB 설명 약 4천 곳)
    Path("data/interim/emb/mminilm-l12-v1/travelers_aihub-v1.csv"),
]


def main() -> None:
    out = Path("data/interim/colab/tripin_colab.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in FILES:
            if "__pycache__" not in path.parts:
                zf.write(path, path.as_posix())
    print(f"{out} ({out.stat().st_size / 1e6:.1f}MB, 파일 {len(FILES)}개)")


if __name__ == "__main__":
    main()
