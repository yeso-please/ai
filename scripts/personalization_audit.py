"""개인화 몫 점검: 장소만 보는 모델 vs 장소 + 설문 상호작용 모델 (CPU, 학습 라벨 누출 없음).

질문 두 가지에 답한다(2026-10-24, 교수 피드백 후속).
  1. NDCG@10 ≈ 0.108(E5)은 장소 특성만으로 얼마나 나오나? 실패한 작은 순위 모델은 버그인가 목표 불일치인가?
  2. 설문(스타일 4문항 + 동기)을 장소 특성과 상호작용시키면 개인화 몫(본인 설문 − 남의 설문)이 얼마나 생기나?
     → 지금 데이터로 얻을 수 있는 개인화의 현실적 상한 추정.

평가는 E5 5겹과 같은 여행자 단위 겹(assign_folds), 같은 후보(시군구, 20곳 이상), 같은 지표(holdout.query_metrics)다.
텍스트 특성은 사전학습 MiniLM(만족도 학습 없음) 관광지 벡터를 PCA로 줄여 쓴다.

  python scripts/personalization_audit.py            # 결과: data/interim/eval_results/personalization-audit-v1/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_validate_t4 import assign_folds
from personal_ranker import profile_features
from train_combined_t4 import TEXTS, TRAIN_2022, TRAIN_2023, mapped_visits
from tripin_ai.evaluation.data import load
from tripin_ai.evaluation.holdout import query_metrics, summarize

OUT = Path("data/interim/eval_results/personalization-audit-v1")
TEXT_VECTORS = Path("data/interim/emb/mminilm-l12-v1/attractions_v2.npy")
MIN_CANDIDATES, NEG_PER_QUERY, TEXT_DIMS, INTER_TEXT_DIMS, SEED = 20, 15, 32, 8, 42


def prepare():
    a = pd.read_csv(TEXTS, dtype=str).fillna("")
    base = load(a.contentid.tolist())
    frames = []
    for year, folder in [("2022", TRAIN_2022), ("2023", TRAIN_2023)]:
        t = pd.read_csv(folder / "travelers.csv", dtype=str)
        t["year"], t["stratum"] = year, year + "-" + t.region.fillna("unknown")
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    folds = assign_folds(t)
    v = pd.concat([mapped_visits(f / "visits.csv", dict(zip(a.contentid, range(len(a)))))
                   for f in [TRAIN_2022, TRAIN_2023]], ignore_index=True)
    text = np.load(TEXT_VECTORS).astype(np.float32)
    if len(text) != len(a):
        raise ValueError("Text vectors are not aligned with attraction texts")
    return a, base, t, folds, v, text


def item_features(attractions, text):
    """장소 쪽 특성: 유형·분류 원-핫 + 텍스트 PCA. 장소 ID는 쓰지 않는다(새 장소에도 일반화)."""
    cats = []
    for field in ["contenttypeid", "lclsSystm1", "lclsSystm2", "lclsSystm3"]:
        values = attractions[field].fillna("").astype(str)
        cats.append(pd.get_dummies(values, prefix=field, dtype=np.float32))
    onehot = pd.concat(cats, axis=1)
    coarse = pd.concat(cats[:3], axis=1).to_numpy(np.float32)   # 상호작용용(대·중분류·유형)
    pcs = PCA(TEXT_DIMS, random_state=SEED).fit_transform(text).astype(np.float32)
    pcs /= pcs.std(axis=0, keepdims=True) + 1e-6
    return onehot.to_numpy(np.float32), coarse, pcs


def queries(visits, candidates, travel_ids):
    out = []
    for (uid, city), group in visits[visits.travel_id.isin(travel_ids)].groupby(["travel_id", "key"], sort=True):
        cand = candidates.get(city, np.array([], dtype=int))
        if len(cand) < MIN_CANDIDATES:
            continue
        ratings = {int(i): float(s) for i, s in zip(group.a_idx, group.s) if int(i) in set(cand.tolist())}
        if ratings:
            out.append((uid, city, cand, ratings))
    return out


def design(items, users, item_rows, user_rows, personal, chunk=40000):
    """[장소 특성] (+ [설문 ⊗ (유형·대·중분류, 텍스트 PCA 앞 8개)]). 메모리 때문에 나눠 만든다."""
    onehot, coarse, pcs = items
    parts = []
    for s in range(0, len(item_rows), chunk):
        ir, ur = item_rows[s:s + chunk], user_rows[s:s + chunk]
        place = np.hstack([onehot[ir], pcs[ir]])
        if personal:
            side = np.hstack([coarse[ir], pcs[ir][:, :INTER_TEXT_DIMS]])
            inter = np.einsum("ni,nj->nij", users[ur], side).reshape(len(ir), -1)
            place = np.hstack([place, inter])
        parts.append(sparse.csr_matrix(place))
    return sparse.vstack(parts).tocsr()


def visit_training_rows(qs, user_index, rng):
    """방문(1) vs 같은 시군구 미방문 표본(0). 미방문을 '싫음'이 아니라 '선택되지 않음'으로만 쓴다."""
    items, users, labels = [], [], []
    for uid, _, cand, ratings in qs:
        visited = np.fromiter(ratings, dtype=int)
        pool = np.setdiff1d(cand, visited)
        neg = rng.choice(pool, min(NEG_PER_QUERY, len(pool)), replace=False)
        for group, label in [(visited, 1), (neg, 0)]:
            items.extend(group.tolist()); users.extend([user_index[uid]] * len(group)); labels.extend([label] * len(group))
    return np.array(items), np.array(users), np.array(labels)


def pair_training_rows(qs, user_index):
    """같은 여행자·시군구 방문지 쌍(만족도 높은 쪽 − 낮은 쪽). 장소 매력과 개인 취향이 섞인 신호."""
    hi, lo, users = [], [], []
    for uid, _, _, ratings in qs:
        keys = list(ratings)
        for i in keys:
            for j in keys:
                if ratings[i] > ratings[j]:
                    hi.append(i); lo.append(j); users.append(user_index[uid])
    return np.array(hi), np.array(lo), np.array(users)


def fit_logistic(x, y, c=0.1):
    model = LogisticRegression(C=c, max_iter=400, solver="lbfgs")
    model.fit(x, y)
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, base, t, folds, v, text = prepare()
    users = profile_features(t)
    user_index = dict(zip(t.travel_id, range(len(t))))
    year_of = dict(zip(t.travel_id, t.year))
    items = item_features(base.attractions, text)
    rows, pair_rows, notes = [], [], []
    for fold in range(5):
        rng = np.random.default_rng(SEED + fold)
        train_ids = {u for u, f in folds.items() if f != fold}
        test_ids = {u for u, f in folds.items() if f == fold}
        train_q, test_q = queries(v, base.candidates, train_ids), queries(v, base.candidates, test_ids)

        # 방문 빈도(인기) 기준선: 학습 겹 방문 수
        counts = np.zeros(len(a))
        np.add.at(counts, v[v.travel_id.isin(train_ids)].a_idx.astype(int).to_numpy(), 1)

        # 1) 방문 예측: 장소만 / 장소 + 설문 상호작용
        it, us, y = visit_training_rows(train_q, user_index, rng)
        place = fit_logistic(design(items, users, it, us, False), y)
        personal = fit_logistic(design(items, users, it, us, True), y)

        # 2) 만족도 쌍: 차이 특성으로 로지스틱(절편 없음) — 장소만 / 장소 + 설문
        hi, lo, pu = pair_training_rows(train_q, user_index)
        sign = rng.integers(0, 2, len(hi)) * 2 - 1          # 반반 뒤집어 두 클래스를 만든다
        def pair_x(personal_terms):
            return (design(items, users, hi, pu, personal_terms) - design(items, users, lo, pu, personal_terms)).multiply(sign[:, None]).tocsr()
        pair_place = LogisticRegression(C=0.1, max_iter=400, fit_intercept=False).fit(pair_x(False), sign > 0)
        pair_personal = LogisticRegression(C=0.1, max_iter=400, fit_intercept=False).fit(pair_x(True), sign > 0)

        # 남의 설문: 같은 시험 겹 안에서 고정 시드로 섞는다(장소 쪽 신호는 그대로, 개인 대응만 깨진다)
        test_users = sorted({uid for uid, *_ in test_q})
        shuffled_of = dict(zip(test_users, np.random.default_rng(1000 + fold).permutation(test_users)))

        def score(model, cand, uid, personal_terms, as_pair=False):
            u = np.full(len(cand), user_index[uid])
            x = design(items, users, cand, u, personal_terms)
            return x @ model.coef_.ravel() if as_pair else model.decision_function(x)

        for uid, city, cand, ratings in test_q:
            other = shuffled_of[uid]
            methods = {
                "popularity": np.log1p(counts[cand]),
                "place_only": score(place, cand, uid, False),
                "place+survey": score(personal, cand, uid, True),
                "shuffled": score(personal, cand, other, True),
            }
            for name, s in methods.items():
                model_metrics, random_metrics = query_metrics(cand, ratings, s)
                for metric, value in model_metrics.items():
                    rows.append(dict(travel_id=uid, year=year_of[uid], method=name, metric=metric, value=value))
                if name == "popularity":
                    for metric, value in random_metrics.items():
                        rows.append(dict(travel_id=uid, year=year_of[uid], method="random", metric=metric, value=value))
            # 방문지끼리 만족도 순서만 학습한 모델(실패한 작은 순위 모델과 같은 목표)도 전체 후보 지표로 잰다
            pair_methods = {
                "pair_place_only": score(pair_place, cand, uid, False, True),
                "pair_place+survey": score(pair_personal, cand, uid, True, True),
                "shuffled": score(pair_personal, cand, other, True, True),
            }
            for name, s in pair_methods.items():
                m, _ = query_metrics(cand, ratings, s)
                for metric, value in m.items():
                    pair_rows.append(dict(travel_id=uid, year=year_of[uid], method=name, metric=metric, value=value))
        notes.append(dict(fold=fold, train_queries=len(train_q), test_queries=len(test_q),
                          visit_rows=int(len(y)), pairs=int(len(hi))))
        print(notes[-1], flush=True)

    summary, _ = summarize(rows)
    pair_summary, _ = summarize(pair_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    pair_summary.to_csv(OUT / "pair_summary.csv", index=False)
    (OUT / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    view = pd.concat([summary, pair_summary])
    view = view[view.subset == "combined"]
    print(view[["metric", "method", "value", "lo", "hi", "n_travelers", "diff_vs_shuffled", "diff_vs_shuffled_lo",
                "diff_vs_shuffled_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
