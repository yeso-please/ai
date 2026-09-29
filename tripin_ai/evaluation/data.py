"""평가 데이터(#3 산출물) 로딩과 여행자 단위 분할."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path("data/interim/eval")
FESTIVAL, SHOPPING = "15", "38"


@dataclass
class EvalData:
    attractions: pd.DataFrame            # 행 위치 = 관광지 인덱스 (임베딩 행 순서와 같게)
    travelers: pd.DataFrame
    visits: pd.DataFrame                 # travel_id, a_idx, s(만족도), key(시군구)
    candidates: dict[str, np.ndarray]    # 시군구 → 후보 관광지 인덱스
    fold_of: dict[str, int]              # travel_id → 평가 묶음 번호 (매칭 방문이 있는 여행자만)
    n_folds: int


def load(attraction_order: list[str] | None = None, include_shopping: bool = False,
         n_folds: int = 5, seed: int = 42, eval_dir: Path = EVAL_DIR) -> EvalData:
    attractions = pd.read_csv(eval_dir / "attractions.csv", dtype=str)
    if attraction_order is not None:
        attractions = attractions.set_index("contentid").loc[attraction_order].reset_index()
    excluded = {FESTIVAL} if include_shopping else {FESTIVAL, SHOPPING}
    index_of = {c: i for i, c in enumerate(attractions.contentid)}
    key = (attractions.lDongRegnCd + "-" + attractions.lDongSignguCd).fillna("").values
    # 시군구 코드가 없는 관광지는 어느 지역 후보에도 넣지 않는다.
    is_candidate = ~attractions.contenttypeid.isin(excluded).values & (key != "")
    candidates = {k: np.where(is_candidate & (key == k))[0] for k in np.unique(key[is_candidate])}

    travelers = pd.read_csv(eval_dir / "travelers.csv", dtype=str)
    visits = pd.read_csv(eval_dir / "visits.csv", dtype=str)
    visits = visits[~visits.contenttypeid.isin(excluded) & visits.contentid.isin(index_of)]
    visits = visits.drop_duplicates(["travel_id", "contentid"])
    visits = pd.DataFrame({
        "travel_id": visits.travel_id.values,
        "a_idx": visits.contentid.map(index_of).values,
        "s": pd.to_numeric(visits.satisfaction).values,
        "key": (visits.lDongRegnCd + "-" + visits.lDongSignguCd).fillna("").values,
    }).dropna(subset=["s"])

    # 여행자 단위 k겹 분할. 같은 여행자의 방문은 한 묶음에만 들어간다.
    ids = np.array(sorted(visits.travel_id.unique()))
    order = np.random.default_rng(seed).permutation(len(ids))
    fold_of = {ids[i]: int(n % n_folds) for n, i in enumerate(order)}
    return EvalData(attractions, travelers, visits, candidates, fold_of, n_folds)


def popularity(visits: pd.DataFrame, n_attractions: int) -> np.ndarray:
    """학습 묶음의 매칭 방문 수 (평가 묶음 정답이 새지 않게 학습 묶음만 넘긴다)."""
    return np.bincount(visits.a_idx.astype(int), minlength=n_attractions).astype(float)
