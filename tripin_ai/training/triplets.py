"""학습 데이터: (여행자 설문 문장, 만족한 관광지 문장, 어려운 음성 관광지 문장).

어려운 음성은 모델이 "유명한 곳"이 아니라 취향을 배우게 하려고 고른다.
  1순위: 같은 여행자가 같은 시군구에서 3점 이하를 준 곳 (실제로 덜 만족한 곳)
  2순위: 같은 시군구에서 인기(방문 수)가 거의 같은 안 간 곳 (평가 주2와 같은 기준)
  3순위: 같은 시군구의 안 간 곳 아무 데나
배치 안의 다른 예시들도 음성으로 쓰인다(MultipleNegativesRankingLoss).
"""
import numpy as np
import pandas as pd

from ..evaluation.data import EvalData, popularity
from ..evaluation.runner import popularity_matched


def build_triplets(data: EvalData, train_ids: set[str], traveler_texts: dict[str, str],
                   attraction_texts: list[str], seed: int = 0, min_satisfaction: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train_visits = data.visits[data.visits.travel_id.isin(train_ids)]
    pop = popularity(train_visits, len(data.attractions))
    rows, sources = [], {"low_rated": 0, "pop_matched": 0, "same_region": 0}
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
            if low:
                negative, source = int(rng.choice(low)), "low_rated"
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
