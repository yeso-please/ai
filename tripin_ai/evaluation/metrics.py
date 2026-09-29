"""평가 지표와 여행자 단위 부트스트랩.

지표는 "단위(unit)"마다 (분자, 분모)로 기록하고, 여행자 단위로 다시 뽑아(부트스트랩) 신뢰구간을 낸다.
같은 여행자의 쌍·후보는 서로 독립이 아니기 때문이다.
"""
import numpy as np


def pair_correct(high: float, low: float) -> float:
    """더 만족한 곳(high)의 점수가 높으면 1, 같으면 0.5, 낮으면 0."""
    return 1.0 if high > low else 0.5 if high == low else 0.0


def auc(positive: float, negatives: np.ndarray) -> float:
    """정답 하나가 음성들보다 높은 비율 (동점은 절반)."""
    negatives = np.asarray(negatives)
    return float((negatives < positive).mean() + 0.5 * (negatives == positive).mean())


def top_k(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """점수 상위 k의 위치. 동점은 무작위로 푼다(인기 0처럼 동점이 많은 경우 순서 편향을 막는다)."""
    order = np.lexsort((rng.random(len(scores)), -np.asarray(scores)))
    return order[:k]


def ndcg_at_k(ranked_gains: np.ndarray, all_gains: np.ndarray, k: int) -> float:
    """ranked_gains: 추천 상위 k의 이득, all_gains: 후보 전체의 이득."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((np.asarray(ranked_gains[:k]) * discounts[:len(ranked_gains[:k])]).sum())
    ideal = np.sort(np.asarray(all_gains))[::-1][:k]
    idcg = float((ideal * discounts[:len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap(units: dict[str, list[tuple[str, float, float]]], baseline: str | None,
              n_boot: int = 1000, seed: int = 0) -> dict[str, dict[str, float]]:
    """units: 방식 → [(여행자, 분자, 분모), …]. 모든 방식이 같은 여행자 집합을 쓴다고 가정한다.

    반환: 방식 → {value, lo, hi, n_travelers, diff, diff_lo, diff_hi} (diff는 baseline 대비, 짝지은 부트스트랩)
    """
    methods = list(units)
    travelers = sorted({t for rows in units.values() for t, _, _ in rows})
    position = {t: i for i, t in enumerate(travelers)}
    num = np.zeros((len(methods), len(travelers)))
    den = np.zeros((len(methods), len(travelers)))
    for m, method in enumerate(methods):
        for traveler, n, d in units[method]:
            num[m, position[traveler]] += n
            den[m, position[traveler]] += d

    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(travelers), size=(n_boot, len(travelers)))
    boot_num = num[:, samples].sum(axis=2)   # (방식, n_boot)
    boot_den = den[:, samples].sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = boot_num / boot_den
    point = num.sum(axis=1) / np.maximum(den.sum(axis=1), 1e-12)

    result = {}
    base = methods.index(baseline) if baseline in methods else None
    for m, method in enumerate(methods):
        lo, hi = np.nanpercentile(boot[m], [2.5, 97.5])
        entry = {"value": float(point[m]), "lo": float(lo), "hi": float(hi),
                 "n_travelers": int((den[m] > 0).sum()), "n_units": float(den[m].sum())}
        if base is not None and m != base:
            diff = boot[m] - boot[base]
            d_lo, d_hi = np.nanpercentile(diff, [2.5, 97.5])
            entry.update(diff=float(point[m] - point[base]), diff_lo=float(d_lo), diff_hi=float(d_hi))
        result[method] = entry
    return result
