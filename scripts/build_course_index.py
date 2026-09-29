"""관광공사 추천코스 색인 만들기 (#7). 서버의 /explanations가 비슷한 코스를 문체 예시로 검색할 때 쓴다.

코스 벡터 = 코스 지점 관광지 문장(템플릿 v1: 이름·분류·지역·설명)의 벡터 평균. 서버와 **같은 모델**로 만들어야
우리 코스 벡터와 비교할 수 있다(모델을 바꾸면 색인도 다시 만든다).

입력: data/raw/tourapi/courses/{courses,stops,overviews}.csv (collect_tourapi_courses.py),
      data/interim/emb/<기성 버전>/attractions_v1.csv (지점 관광지 문장)
출력: data/interim/course_index/<모델 버전>.npz  (TourAPI 공개 데이터만 담는다)

사용 예:
  .venv/Scripts/python scripts/build_course_index.py
  .venv/Scripts/python scripts/build_course_index.py --model-name data/interim/models/<버전>/final --model-version <버전>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tripin_ai.encoder import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION, Encoder  # noqa: E402

COURSE_DIR = Path("data/raw/tourapi/courses")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--texts", default="data/interim/emb/mminilm-l12-v1/attractions_v1.csv")
    parser.add_argument("--out-dir", default="data/interim/course_index")
    args = parser.parse_args()

    courses = pd.read_csv(COURSE_DIR / "courses.csv", dtype=str)
    stops = pd.read_csv(COURSE_DIR / "stops.csv", dtype=str).fillna("")
    overviews = {}
    if (COURSE_DIR / "overviews.csv").exists():
        o = pd.read_csv(COURSE_DIR / "overviews.csv", dtype=str).fillna("")
        overviews = dict(zip(o.course_id, o.overview.str.replace(r"<[^>]+>", " ", regex=True)))
    codes = pd.read_csv("data/reference/tourapi_lcls_codes.csv", dtype=str)
    theme_names = dict(zip(codes.lclsSystm3Cd, codes.lclsSystm3Nm))
    texts = dict(pd.read_csv(args.texts, dtype=str)[["contentid", "text"]].values)

    encoder = Encoder(args.model_name, args.model_version)
    stop_texts = [texts.get(s.subcontentid) or s.subname for s in stops.itertuples()]
    stop_vectors = encoder.encode(stop_texts, show_progress=True)
    stops = stops.assign(row=range(len(stops)))

    rows = {"course_ids": [], "titles": [], "themes": [], "overviews": [], "stop_names": [], "stop_descriptions": []}
    vectors = []
    by_course = dict(tuple(stops.groupby("course_id")))
    for c in courses.itertuples():
        g = by_course.get(c.contentid)
        if g is None or g.empty:
            continue   # 지점을 아직 수집하지 않은 코스
        v = stop_vectors[g.row.values].mean(axis=0)
        vectors.append(v / (np.linalg.norm(v) + 1e-12))
        rows["course_ids"].append(c.contentid)
        rows["titles"].append(c.title)
        rows["themes"].append(theme_names.get(c.lclsSystm3, ""))
        rows["overviews"].append(" ".join(str(overviews.get(c.contentid, "")).split()))
        rows["stop_names"].append("|".join(g.subname.str.replace("|", " ")))
        rows["stop_descriptions"].append("|".join(g.subdetailoverview.str.replace("|", " ").str.slice(0, 300)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model_version}.npz"
    np.savez_compressed(path, vectors=np.array(vectors, dtype="<f4"), model_version=args.model_version,
                        **{k: np.array(v, dtype=object) for k, v in rows.items()})
    with_overview = sum(1 for o in rows["overviews"] if o)
    print(f"코스 {len(vectors)}개 색인 (소개글 있음 {with_overview}개) → {path}")
    if with_overview == 0:
        print("⚠️ 코스 소개글(overview)을 아직 수집하지 않아 문체 예시로 쓸 코스가 없다. collect_tourapi_courses.py를 이어서 돌린다.")


if __name__ == "__main__":
    main()
