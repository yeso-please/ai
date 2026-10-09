"""학습 데이터: (여행자 설문 문장, 만족한 관광지 문장, 어려운 음성 관광지 문장).

어려운 음성은 모델이 "유명한 곳"이 아니라 취향을 배우게 하려고 고른다.
  1순위: 같은 여행자가 같은 시군구에서 3점 이하를 준 곳 (실제로 덜 만족한 곳)
  2순위: 같은 시군구에서 인기(방문 수)가 거의 같은 안 간 곳 (평가 주2와 같은 기준)
  3순위: 같은 시군구의 안 간 곳 아무 데나
배치 안의 다른 예시들도 음성으로 쓰인다(MultipleNegativesRankingLoss).

negatives="taste"이면 1순위 다음에 "설문이 가장 다른 여행자가 같은 시군구에서 만족한 곳"을 쓴다.
음성도 누군가 만족한 곳이라 장소 매력은 비슷해지고, 모델은 설문 차이로만 정답과 음성을 가를 수 있다.
평가의 설문 섞기 대조군(ShuffledSurveyScorer)에서 개인화 몫이 1~2%p에 그친 것을 키우려는 설정이다.
"""
import numpy as np
import pandas as pd

from ..evaluation.data import EvalData, popularity
from ..evaluation.runner import popularity_matched


# 서비스 온보딩이 받는 여행 스타일 문항(1~7점). 동기는 코드 집합으로 비교한다.
SURVEY_STYLES = (1, 3, 5, 6)
OTHER_TASTE_SAMPLE = 30


def survey_profiles(travelers: pd.DataFrame) -> dict[str, tuple[np.ndarray, frozenset]]:
    """travel_id → (스타일 점수 배열(없으면 중립 4), 동기 코드 집합)."""
    profiles = {}
    for r in travelers.itertuples():
        styles = np.array([float(getattr(r, f"style_{i}")) if pd.notna(getattr(r, f"style_{i}")) else 4.0
                           for i in SURVEY_STYLES])
        motives = frozenset(str(getattr(r, f"motive_{i}")) for i in range(1, 4) if pd.notna(getattr(r, f"motive_{i}")))
        profiles[r.travel_id] = (styles, motives)
    return profiles


def survey_distance(a: tuple[np.ndarray, frozenset], b: tuple[np.ndarray, frozenset]) -> float:
    """0(같은 설문)~1. 스타일 점수 차이(최대 6점 × 문항 수로 나눔)와 동기 자카드 거리의 평균."""
    style = float(np.abs(a[0] - b[0]).sum()) / (6.0 * len(SURVEY_STYLES))
    union = a[1] | b[1]
    motive = 1.0 - len(a[1] & b[1]) / len(union) if union else 0.0
    return (style + motive) / 2


def build_triplets(data: EvalData, train_ids: set[str], traveler_texts: dict[str, str],
                   attraction_texts: list[str], seed: int = 0, min_satisfaction: int = 4,
                   negatives: str = "hard") -> pd.DataFrame:
    if negatives not in ("hard", "taste"):
        raise ValueError(f"알 수 없는 음성 방식: {negatives}")
    rng = np.random.default_rng(seed)
    train_visits = data.visits[data.visits.travel_id.isin(train_ids)]
    pop = popularity(train_visits, len(data.attractions))
    rows, sources = [], {"low_rated": 0, "other_taste": 0, "pop_matched": 0, "same_region": 0}
    if negatives == "taste":
        profiles = survey_profiles(data.travelers[data.travelers.travel_id.isin(train_ids)])
        liked = train_visits[train_visits.s >= min_satisfaction]
        liked_in_region = {key: (g.travel_id.to_numpy(), g.a_idx.astype(int).to_numpy())
                           for key, g in liked.groupby("key")}
    for travel_id, g in train_visits.groupby("travel_id"):
        if travel_id not in traveler_texts:
            continue
        visited = set(g.a_idx.astype(int))
        for item, s, key in zip(g.a_idx.astype(int), g.s, g.key):
            if s < min_satisfaction:
                continue
            low = g[(g.key == key) & (g.s <= 3)].a_idx.astype(int).tolist()
            cands = data.candidates.get(key, np.array([], int))
            cands = cands[~np.isin(cands, list(visited))]
            other = other_taste_negative(travel_id, item, key, visited, liked_in_region, profiles, rng)                 if negatives == "taste" and not low else None
            if low:
                negative, source = int(rng.choice(low)), "low_rated"
            elif other is not None:
                negative, source = other, "other_taste"
            else:
                matched = cands[popularity_matched(pop[item], pop[cands])]
                if len(matched):
                    negative, source = int(rng.choice(matched)), "pop_matched"
                elif len(cands):
                    negative, source = int(rng.choice(cands)), "same_region"
                else:
                    continue
            sources[source] += 1
            rows.append({"anchor": traveler_texts[travel_id], "positive": attraction_texts[item],
                         "negative": attraction_texts[negative]})
    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    frame.attrs["negative_sources"] = sources
    return frame


def other_taste_negative(travel_id: str, item: int, key: str, visited: set[int], liked_in_region: dict,
                         profiles: dict, rng: np.random.Generator) -> int | None:
    """같은 시군구에서 다른 여행자가 만족한 곳 중, 설문이 가장 다른 여행자의 것. 이 여행자가 간 곳은 뺀다."""
    if key not in liked_in_region or travel_id not in profiles:
        return None
    others, items = liked_in_region[key]
    ok = (others != travel_id) & ~np.isin(items, list(visited)) & (items != item)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return None
    idx = rng.choice(idx, min(OTHER_TASTE_SAMPLE, len(idx)), replace=False)
    me = profiles[travel_id]
    dist = np.array([survey_distance(me, profiles[others[i]]) if others[i] in profiles else -1.0 for i in idx])
    best = np.flatnonzero(dist == dist.max())
    if dist.max() < 0:
        return None
    return int(items[idx[rng.choice(best)]])
