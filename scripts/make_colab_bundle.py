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
    Path("data/interim/emb/mminilm-l12-v1/attractions_v1.csv"),   # 설명 포함
    Path("data/interim/emb/mminilm-l12-v1/attractions_v2.csv"),   # 설명 + 콘텐츠 유형 (서비스 템플릿 2)
    Path("data/interim/emb/mminilm-l12-v1/travelers_aihub-v1.csv"),
    Path("data/reference/tourapi_lcls_codes.csv"),            # 코스 테마 이름
    # 코스 쌍 실험(--course-pairs). 수집 전이면 빠진다.
    Path("data/raw/tourapi/courses/courses.csv"),
    Path("data/raw/tourapi/courses/stops.csv"),
    Path("data/raw/tourapi/courses/overviews.csv"),
]


def main() -> None:
    out = Path("data/interim/colab/tripin_colab.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in FILES:
            if "__pycache__" not in path.parts and path.exists():
                zf.write(path, path.as_posix())
    print(f"{out} ({out.stat().st_size / 1e6:.1f}MB, 파일 {len(FILES)}개)")


if __name__ == "__main__":
    main()
