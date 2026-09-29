"""AI Hub 여행로그 다운로드(.zip.part*)를 풀어 data/raw/aihub/<권역>/<TL|VL>/ 아래에 정리한다.

zip 안 파일명이 CP949 한글이라 깨지므로 영문 접두어만 남긴다(예: tn_visit_area_info_E.csv).
권역은 파일명 끝 글자로 구분한다: E 수도권, F 동부권, G 서부권, H 제주도 및 도서지역.

사용 예:
  python scripts/extract_aihub.py --src data/raw --dest data/raw/aihub
"""
import argparse
import re
import zipfile
from pathlib import Path

REGIONS = {"E": "capital", "F": "east", "G": "west", "H": "jeju"}


def merge_parts(first_part: Path) -> Path:
    """aihubshell이 나눈 .zip.part0, part1 … 을 하나의 zip으로 합친다."""
    base = first_part.with_name(first_part.name.split(".part")[0])
    parts = sorted(first_part.parent.glob(base.name + ".part*"), key=lambda p: int(p.name.rsplit("part", 1)[1]))
    if not base.exists():
        with base.open("wb") as out:
            for part in parts:
                out.write(part.read_bytes())
    return base


def ascii_name(member: str) -> str:
    stem = Path(member).name.removesuffix(".csv")
    tokens = [t for t in stem.split("_") if re.fullmatch(r"[a-zA-Z]+", t)]
    return "_".join(tokens) + ".csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw")
    parser.add_argument("--dest", default="data/raw/aihub")
    args = parser.parse_args()

    for first_part in sorted(Path(args.src).rglob("*_csv.zip.part0")):
        split = "TL" if first_part.name.startswith("TL_") else "VL"
        archive = merge_parts(first_part)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
            suffix = next((re.search(r"_([EFGH])\.csv$", m.filename) for m in members
                           if re.search(r"_([EFGH])\.csv$", m.filename)), None)
            if suffix is None:
                print(f"권역을 알 수 없어 건너뜀: {first_part}")
                continue
            out_dir = Path(args.dest) / REGIONS[suffix.group(1)] / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for member in members:
                (out_dir / ascii_name(member.filename)).write_bytes(zf.read(member))
        print(f"{REGIONS[suffix.group(1)]}/{split}: {len(members)}개 CSV")


if __name__ == "__main__":
    main()
