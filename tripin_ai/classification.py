"""TourAPI 새 분류체계(lclsSystm1~3) 코드 → 이름. 관광지 문장 템플릿이 분류명을 쓴다.

코드표는 data/reference/tourapi_lcls_codes.csv (collect_tourapi.py --class-codes로 받는다).
"""
import csv
from functools import lru_cache
from pathlib import Path

CODES_PATH = Path(__file__).resolve().parents[1] / "data" / "reference" / "tourapi_lcls_codes.csv"


@lru_cache(maxsize=1)
def class_code_names() -> dict[str, str]:
    names = {}
    with CODES_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for level in (1, 2, 3):
                names[row[f"lclsSystm{level}Cd"]] = row[f"lclsSystm{level}Nm"]
    return names


def class_names_of(*codes: str | None) -> list[str]:
    """알 수 없는 코드는 건너뛴다(TourAPI가 분류를 추가해도 서버가 실패하지 않게)."""
    names = class_code_names()
    return [names[c] for c in codes if c and c in names]
