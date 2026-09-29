"""여행자 단위 k겹 교차검증으로 모든 방식을 같은 표본에서 평가한다.

조건
  cold: 설문만 쓴다(서비스 신규 회원). 방문 기록이 필요한 방식(협업 필터링 등)은 빠진다.
  warm: 매칭 방문이 4곳 이상인 여행자의 방문 절반을 공개(history)하고 나머지 절반으로 평가한다.

지표 (단위마다 분자·분모 기록 → 여행자 단위 부트스트랩)
  주1 취향 쌍 비교        같은 여행자가 간 곳 중 더 만족한 곳이 더 높은 점수를 받는 비율 (전체 / 5점 vs 3점 이하)
  주2 인기 맞춘 후보 AUC   만족한 곳을 같은 시군구에서 인기(방문 수)가 거의 같은 안 간 곳들보다 높게 두는 비율
  주3 덜 알려진 정답 적중@10, 추천 상위 10의 평균 인기 백분위(낮을수록 덜 알려진 곳)
  참고 방문 적중@10, Recall@10, nDCG@10 (인기 편향이 큰 방문 예측)
"""
from collections import defaultdict

import numpy as np

from .data import EvalData, popularity
from .metrics import auc, ndcg_at_k, pair_correct

MIN_CANDIDATES = 20
K = 10
# 주2 인기 맞추기: 방문 수 7 이하는 정확히 같게, 그보다 많으면 log2 차이 0.25 이내(약 ±19%).
# 구간을 넓게 잡으면 같은 구간 안에서도 인기 차이가 남아 인기가 유리해진다(0/1/2~3/4~7/8~15/16+ 구간일 때 인기 AUC 61%).
EXACT_MATCH_MAX_VISITS = 7
RELATIVE_WINDOW = 0.25
NEGATIVES_PER_POSITIVE = 20
MIN_NEGATIVES = 5
LONG_TAIL_MAX_VISITS = 2

METRICS = {
    "m1_pair_all": "주1 취향 쌍 비교 (만족도 차이가 있는 모든 쌍)",
    "m1_pair_5v3": "주1 취향 쌍 비교 (5점 vs 3점 이하)",
    "m2_pop_matched_auc": "주2 인기 맞춘 후보 AUC",
    "m3_tail_hit": "주3 덜 알려진 정답 적중@10",
    "m3_top_pop_pct": "주3 추천 상위 10의 평균 인기 백분위 (낮을수록 덜 알려진 곳)",
    "ref_hit": "참고 방문 적중@10",
    "ref_recall": "참고 Recall@10",
    "ref_ndcg": "참고 nDCG@10",
}


def _pairs(sats: dict[int, float]):
    """만족도가 다른 (더 만족한 곳, 덜 만족한 곳) 쌍."""
    items = list(sats)
    for i in items:
        for j in items:
            if sats[i] > sats[j]:
                yield i, j


def popularity_matched(target: float, candidates: np.ndarray) -> np.ndarray:
    if target <= EXACT_MATCH_MAX_VISITS:
        return candidates == target
    return np.abs(np.log2(np.clip(candidates, 1, None)) - np.log2(target)) <= RELATIVE_WINDOW


def evaluate(data: EvalData, scorers: list, seed: int = 0, warm_min_visits: int = 4, folds: list[int] | None = None):
    rng = np.random.default_rng(seed)
    units = defaultdict(lambda: defaultdict(list))   # (조건, 지표) → 방식 → [(여행자, 분자, 분모)]
    n_attractions = len(data.attractions)
    fold = data.visits.travel_id.map(data.fold_of).values

    for f in (folds if folds is not None else range(data.n_folds)):
        train = data.visits[fold != f]
        pop = popularity(train, n_attractions)
        for scorer in scorers:
            scorer.fit(train, fold=f)

        for travel_id, g in data.visits[fold == f].groupby("travel_id"):
            visited = dict(zip(g.a_idx.astype(int), g.s))
            keys = dict(zip(g.a_idx.astype(int), g.key))
            conditions = {"cold": (None, visited)}
            if len(visited) >= warm_min_visits:
                items = rng.permutation(list(visited))
                shown = items[: len(items) // 2]
                conditions["warm"] = ([(int(i), visited[i]) for i in shown], {int(i): visited[i] for i in items[len(items) // 2:]})

            for condition, (history, hidden) in conditions.items():
                shown_items = {i for i, _ in history} if history else set()

                # 주1: 숨겨진(평가 대상) 방문끼리의 만족도 쌍
                pairs = list(_pairs(hidden))
                # 주2: 만족한 곳마다 같은 시군구·같은 인기 구간의 안 간 곳을 음성으로
                m2_units = []
                for item, s in hidden.items():
                    if s < 4:
                        continue
                    cands = data.candidates.get(keys[item], np.array([], int))
                    negs = cands[popularity_matched(pop[item], pop[cands]) & ~np.isin(cands, list(visited))]
                    if len(negs) >= MIN_NEGATIVES:
                        m2_units.append((item, rng.choice(negs, min(NEGATIVES_PER_POSITIVE, len(negs)), replace=False)))
                # 참고·주3: 시군구마다 후보 순위
                region_units = []
                for key in {keys[i] for i in hidden}:
                    cands = data.candidates.get(key, np.array([], int))
                    cands = cands[~np.isin(cands, list(shown_items))]
                    if len(cands) < MIN_CANDIDATES:
                        continue
                    gains = np.array([2.0 if hidden.get(c, 0) >= 5 else 1.0 if hidden.get(c, 0) >= 4 else 0.0 for c in cands])
                    if gains.sum() > 0:
                        region_units.append((cands, gains, rng.random(len(cands))))

                for scorer in scorers:
                    name = scorer.name
                    # 모든 단위에 같은 점수를 쓰도록 필요한 관광지를 한 번에 점수 매긴다.
                    needed = set(hidden)
                    for item, negs in m2_units:
                        needed.update(negs.tolist())
                    for cands, _, _ in region_units:
                        needed.update(cands.tolist())
                    needed = np.array(sorted(needed), dtype=int)
                    values = scorer.score(travel_id, needed, history)
                    if values is None:
                        continue
                    sc = dict(zip(needed.tolist(), np.asarray(values, float).tolist()))

                    if pairs:
                        correct = [pair_correct(sc[i], sc[j]) for i, j in pairs]
                        units[(condition, "m1_pair_all")][name].append((travel_id, sum(correct), len(correct)))
                        strong = [pair_correct(sc[i], sc[j]) for i, j in pairs if hidden[i] == 5 and hidden[j] <= 3]
                        if strong:
                            units[(condition, "m1_pair_5v3")][name].append((travel_id, sum(strong), len(strong)))
                    for item, negs in m2_units:
                        units[(condition, "m2_pop_matched_auc")][name].append(
                            (travel_id, auc(sc[item], np.array([sc[n] for n in negs])), 1.0))
                    for cands, gains, noise in region_units:
                        scores = np.array([sc[c] for c in cands])
                        order = np.lexsort((noise, -scores))
                        top = order[:K]
                        hit = float(gains[top].sum() > 0)
                        units[(condition, "ref_hit")][name].append((travel_id, hit, 1.0))
                        units[(condition, "ref_recall")][name].append((travel_id, float((gains[top] > 0).sum()), float((gains > 0).sum())))
                        units[(condition, "ref_ndcg")][name].append((travel_id, ndcg_at_k(gains[top], gains, K), 1.0))
                        pct = (pop[cands][:, None] < pop[cands][top][None, :]).mean(axis=0)  # 후보 중 자기보다 인기 낮은 곳의 비율
                        units[(condition, "m3_top_pop_pct")][name].append((travel_id, float(pct.mean()), 1.0))
                        tail = pop[cands] <= LONG_TAIL_MAX_VISITS
                        tail_gold = tail & (gains > 0)
                        if tail_gold.any() and tail.sum() >= MIN_CANDIDATES:
                            t_idx = np.where(tail)[0]
                            t_top = t_idx[np.lexsort((noise[t_idx], -scores[t_idx]))[:K]]
                            units[(condition, "m3_tail_hit")][name].append((travel_id, float(tail_gold[t_top].any()), 1.0))
    return units
