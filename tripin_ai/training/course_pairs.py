"""TourAPI 추천코스로 만드는 보조 학습 쌍 (#6 실험 A).

관광공사가 만든 코스는 "의도 문장(테마·제목·설명) → 그 의도에 맞는 관광지들"이다. 설문 문장 → 만족한 관광지와
구조가 같아서, AI Hub 설문에 없는 표현(힐링, 탁 트인, 가족과 함께 …)과 관광지를 잇는 데 쓸 수 있다.

mode
  real      코스의 실제 지점들을 양성으로
  shuffled  통제군: 지점을 같은 시군구의 무작위 관광지로 바꾼다(지점 수·지역은 같고 코스의 의미만 없앤다).
            real만 오르면 "학습 데이터가 늘어서"가 아니라 "코스의 의미" 덕분이라고 말할 수 있다.

평가 여행자(AI Hub)는 쓰지 않으므로 평가 누설이 없다. 평가는 계속 AI Hub로 한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.data import EvalData

COURSE_DIR = Path("data/raw/tourapi/courses")
OVERVIEW_MAX_CHARS = 200


def course_texts(course_dir: Path = COURSE_DIR, class_names: dict[str, str] | None = None) -> dict[str, str]:
    """코스 id → 의도 문장. 회원 문장처럼 "…를 원하는 여행자"로 시작해 문장 분포를 맞춘다."""
    courses = pd.read_csv(course_dir / "courses.csv", dtype=str)
    overviews = {}
    if (course_dir / "overviews.csv").exists():
        o = pd.read_csv(course_dir / "overviews.csv", dtype=str).dropna(subset=["overview"])
        overviews = dict(zip(o.course_id, o.overview.str.replace(r"<[^>]+>", " ", regex=True)))
    class_names = class_names or {}
    texts = {}
    for c in courses.itertuples():
        theme = class_names.get(c.lclsSystm3, "")
        parts = [f"{theme}를 원하는 여행자." if theme else "", f"{c.title}."]
        overview = " ".join(str(overviews.get(c.contentid, "")).split())[:OVERVIEW_MAX_CHARS]
        texts[c.contentid] = " ".join(p for p in parts + [overview] if p)
    return texts


def build_course_triplets(data: EvalData, attraction_texts: list[str], mode: str = "real",
                          course_dir: Path = COURSE_DIR, class_names: dict[str, str] | None = None,
                          seed: int = 0) -> pd.DataFrame:
    if mode not in ("real", "shuffled"):
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    stops = pd.read_csv(course_dir / "stops.csv", dtype=str)
    texts = course_texts(course_dir, class_names)
    index_of = {c: i for i, c in enumerate(data.attractions.contentid)}
    key_of = (data.attractions.lDongRegnCd + "-" + data.attractions.lDongSignguCd).fillna("").values
    candidate_set = {int(i) for arr in data.candidates.values() for i in arr}

    rows = []
    for course_id, g in stops.groupby("course_id"):
        if course_id not in texts:
            continue
        items = [index_of[s] for s in g.subcontentid if s in index_of and index_of[s] in candidate_set]
        for item in items:
            pool = data.candidates.get(key_of[item], np.array([], int))
            pool = pool[~np.isin(pool, items)]
            if len(pool) == 0:
                continue
            positive = item if mode == "real" else int(rng.choice(pool))
            negative = int(rng.choice(pool[pool != positive])) if (pool != positive).any() else None
            if negative is None:
                continue
            rows.append({"anchor": texts[course_id], "positive": attraction_texts[positive],
                         "negative": attraction_texts[negative]})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
