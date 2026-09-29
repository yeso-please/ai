"""관광지 설명(TourAPI overview) 모으기 → 관광지 템플릿 v1 입력 (#4, #6).

설명 출처(먼저 있는 것을 쓴다)
  1. 백엔드 DB에 적재된 설명: data/interim/descriptions_backend.csv (contentid, description)
     만들기: docker exec tripin-local-postgres psql -U tripin_local -d tripin_local -c "\\copy (select
             source_content_id as contentid, description from app.attractions where source_system='TOUR_API'
             and description is not null and description<>'') to stdout with csv header" > data/interim/descriptions_backend.csv
  2. TourAPI 추천코스 지점 설명(subdetailoverview): data/raw/tourapi/courses/stops.csv
     코스 수집(collect_tourapi_courses.py)에서 덤으로 받은 것. 관광지 상세 조회 호출을 아낄 수 있다.

출력
  data/interim/descriptions_combined.csv  (contentid, description, source)
  data/interim/emb/<기성 버전>/attractions_v1.csv  (--write-texts: 파인튜닝·Colab 묶음이 읽는 v1 문장)

사용 예:
  python scripts/build_descriptions.py --write-texts
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SOURCES = [
    ("backend", Path("data/interim/descriptions_backend.csv"), "contentid", "description"),
    ("course_stop", Path("data/raw/tourapi/courses/stops.csv"), "subcontentid", "subdetailoverview"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/interim/descriptions_combined.csv")
    parser.add_argument("--write-texts", action="store_true", help="attractions_v1.csv도 다시 만든다")
    parser.add_argument("--base-version", default="mminilm-l12-v1")
    args = parser.parse_args()

    frames = []
    for name, path, id_col, text_col in SOURCES:
        if not path.exists():
            print(f"건너뜀(파일 없음): {path}")
            continue
        df = pd.read_csv(path, dtype=str)[[id_col, text_col]].dropna()
        df = df.rename(columns={id_col: "contentid", text_col: "description"})
        df = df[df.description.str.strip().str.len() > 0].drop_duplicates("contentid")
        frames.append(df.assign(source=name))
        print(f"{name}: {len(df)}곳")
    combined = pd.concat(frames).drop_duplicates("contentid", keep="first")
    combined.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"합계 {len(combined)}곳 → {args.out}")

    if args.write_texts:
        from embed_offline import attraction_texts, load_descriptions

        texts = attraction_texts("v1", load_descriptions(args.out)).reset_index(drop=True)
        v0 = pd.read_csv(Path("data/interim/emb") / args.base_version / "attractions_v0.csv", dtype=str)
        assert (texts.contentid.values == v0.contentid.values).all(), "v0와 관광지 순서가 달라졌다"
        path = Path("data/interim/emb") / args.base_version / "attractions_v1.csv"
        texts.to_csv(path, index=False, encoding="utf-8-sig")
        combined[["contentid"]].to_csv("data/interim/described_contentids.csv", index=False)
        print(f"v1 문장 {len(texts)}곳 (설명 포함 {texts.contentid.isin(combined.contentid).sum()}곳) → {path}")


if __name__ == "__main__":
    main()
