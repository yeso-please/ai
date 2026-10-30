"""여행기 피드백을 다음 추천에 반영하는 방법을 오프라인으로 비교한다. CPU.

질문: 백엔드가 실제로 저장하는 여행기 신호(여행 단위 만족도 1~5 + 그 여행에서 간 장소)만으로
      다음 지역의 추천 순위가 좋아지는가? 어떤 반영 방식이 맞는가?

데이터 (AI Hub 2022·2023, lgbm_ranker.py와 같은 분할·후보·특성)
  - 순위 모델은 TL로만 학습한다. 평가는 TL에 없는 VL 여행자 중 두 개 이상의 시군구를 방문한 사람.
  - AI Hub는 응답자당 여행이 하나라서, 방문 순서가 가장 이른 시군구를 ‘지난 여행(여행기)’,
    나머지 시군구를 ‘다음 여행’으로 본다(한 여행 안의 대리 실험 — 한계로 보고한다).
  - 여행기 신호는 서비스와 같은 정보만 쓴다: 그 지역 방문지 + 여행 단위 만족도(방문지 만족도 평균을 반올림한 1~5).
    장소별 만족도는 서비스에 없으므로 쓰지 않는다(oracle 비교에만 사용).

방법 (점수 = z(순위 모델) + λ · w · z(조정), 같은 질의 안에서 표준화)
  base        순위 모델만 (현재 서비스)
  rocchio     지난 여행 방문지 E5 벡터 중심과의 코사인 (Rocchio 적합성 피드백)
  category    지난 여행 방문지의 중분류 비율을 전체 비율로 수축(empirical Bayes)한 뒤 log-lift
  both        rocchio + category
  w: 여행 단위 만족도의 가중치. 비대칭(기본) w = clip(0.5 + 0.25·(s−3), 0, 1) — 불만족은 반영을 약하게 할 뿐 밀어내지 않음.
     대칭(대조) w = (s−3)/2 — 불만족한 여행의 모든 장소를 ‘싫어함’으로 밀어냄.
  대조군
  rocchio_shuffled  남의 여행기 중심을 넣음(효과가 그 사람의 기록 때문인지 확인)
  oracle_place      장소별 실제 만족도로 가중한 중심(서비스에 없는 정보, 상한 참고)
  λ는 TL 안쪽 검증 여행자(순위 모델 학습에 쓰지 않음)에서 고른다.

  python scripts/diary_feedback_eval.py   # 결과: data/interim/eval_results/diary-feedback-v1/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lgbm_ranker import Features, build_queries, e5_vectors, fit, prepare, SEED
from personal_ranker import profile_features
from train_combined_t4 import TRAIN_2022, TRAIN_2023
from tripin_ai.evaluation.holdout import query_metrics

OUT = Path("data/interim/eval_results/diary-feedback-v1")
LAMBDAS = [0.1, 0.2, 0.3, 0.5, 0.8]
DIAG_LAMBDAS = [-0.1, -0.2, -0.3, -0.5]   # 선택에 쓰지 않는 진단용
PRIOR_K = 5.0            # category 수축 강도(가상 관측 수)
CAT_FIELD = "lclsSystm2"


def z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def weight(s, mode):
    return float(np.clip(0.5 + 0.25 * (s - 3), 0, 1)) if mode == "asym" else (s - 3) / 2


def region_order(t_ids):
    """여행자별 시군구 방문 순서(가장 이른 visit_order)."""
    frames = []
    for folder in [TRAIN_2022, TRAIN_2023]:
        raw = pd.read_csv(folder / "visits.csv", dtype=str, usecols=["travel_id", "visit_order", "lDongRegnCd",
                                                                     "lDongSignguCd"])
        frames.append(raw[raw.travel_id.isin(t_ids)])
    raw = pd.concat(frames)
    raw["key"] = raw.lDongRegnCd + "-" + raw.lDongSignguCd
    raw["order"] = pd.to_numeric(raw.visit_order, errors="coerce")
    return raw.groupby(["travel_id", "key"]).order.min().to_dict()


def episodes(queries, order):
    """여행자별 (지난 여행 질의, [다음 여행 질의들])."""
    by_user = {}
    for q in queries:
        by_user.setdefault(q[0], []).append(q)
    out = []
    for uid, qs in by_user.items():
        if len(qs) < 2:
            continue
        qs = sorted(qs, key=lambda q: order.get((uid, q[1]), 1e9))
        out.append((uid, qs[0], qs[1:]))
    return out


class Adjust:
    def __init__(self, feats, cats, pop_share):
        self.feats, self.cats, self.pop = feats, cats, pop_share

    def centroid(self, items, weights=None):
        m = self.feats.item_matrix[np.fromiter(items, int)]
        w = np.ones(len(m)) if weights is None else np.asarray(weights, float)
        c = (m * w[:, None]).sum(0)
        n = np.linalg.norm(c)
        return c / n if n > 0 else c

    def rocchio(self, cand, centroid):
        return self.feats.item_matrix[cand] @ centroid

    def category(self, cand, hist_items):
        counts = pd.Series(self.cats[np.fromiter(hist_items, int)]).value_counts()
        n = counts.sum()
        lift = np.zeros(len(cand))
        for j, c in enumerate(self.cats[cand]):
            p0 = self.pop.get(c, 0.0)
            if not c or p0 <= 0:
                continue
            pu = (counts.get(c, 0) + PRIOR_K * p0) / (n + PRIOR_K)
            lift[j] = np.log(pu / p0)
        return lift


def score_episode(ep, model, idx, feats, adj, lam, mode, donor=None):
    """다음 여행 질의마다 방법별 점수."""
    uid, (_, _, _, hist_r), targets = ep
    hist_items = list(hist_r)
    s_trip = int(np.clip(round(np.mean(list(hist_r.values()))), 1, 5))
    w = weight(s_trip, mode)
    c_trip = adj.centroid(hist_items)
    c_oracle = adj.centroid(hist_items, [max(0.0, (hist_r[i] - 2) / 3) for i in hist_items])
    out = []
    for _, city, cand, ratings in targets:
        base = z(model.predict(feats.rows(uid, cand)[:, idx], num_iteration=model.best_iteration))
        r = z(adj.rocchio(cand, c_trip))
        c = z(adj.category(cand, hist_items))
        m = {"base": base, "rocchio": base + lam * w * r, "category": base + lam * w * c,
             "both": base + lam * w * (r + c) / 2,
             "oracle_place": base + lam * z(adj.rocchio(cand, c_oracle))}
        if donor is not None:
            m["rocchio_shuffled"] = base + lam * w * z(adj.rocchio(cand, donor))
        out.append((city, cand, ratings, m, s_trip))
    return out


def evaluate(eps, model, idx, feats, adj, lam, mode, donors=None):
    rows = []
    for k, ep in enumerate(eps):
        donor = donors[ep[0]] if donors else None
        for city, cand, ratings, methods, s_trip in score_episode(ep, model, idx, feats, adj, lam, mode, donor):
            for name, scores in methods.items():
                metrics, _ = query_metrics(cand, ratings, scores, seed=SEED + k)
                for metric, value in metrics.items():
                    rows.append(dict(travel_id=ep[0], method=name, metric=metric, value=value, s_trip=s_trip))
    return pd.DataFrame(rows)


def paired(frame, metric="ndcg_at_10", control="base", n_boot=2000):
    users = frame[frame.metric == metric].groupby(["travel_id", "method"]).value.mean().unstack()
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, len(users), size=(n_boot, len(users)))
    out = []
    for m in users.columns:
        d = (users[m] - users[control]).to_numpy()
        bd = d[boot].mean(1)
        out.append(dict(metric=metric, method=m, value=users[m].mean(), diff_vs_base=d.mean(),
                        lo=np.quantile(bd, .025), hi=np.quantile(bd, .975), n_travelers=len(users)))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, base, t, v = prepare()
    tl, vl = set(t.loc[t.split == "TL", "travel_id"]), set(t.loc[t.split == "VL", "travel_id"])
    rng = np.random.default_rng(SEED)
    respondents = t[t.split == "TL"].traveler_id.unique()
    valid_people = set(rng.choice(respondents, len(respondents) // 5, replace=False))
    inner_valid = set(t.loc[t.traveler_id.isin(valid_people) & (t.split == "TL"), "travel_id"])
    train_q = build_queries(v, base.candidates, tl - inner_valid)
    valid_q = build_queries(v, base.candidates, inner_valid)
    test_q = build_queries(v, base.candidates, vl)
    needed_items, needed_users = set(), {q[0] for q in train_q + valid_q + test_q}
    for _, _, cand, _ in train_q + valid_q + test_q:
        needed_items.update(cand.tolist())
    item_vec, user_vec = e5_vectors(a, t, needed_items, needed_users)
    keep = lambda qs: [q for q in qs if q[0] in user_vec]
    train_q, valid_q, test_q = keep(train_q), keep(valid_q), keep(test_q)
    feats = Features(a, base.attractions, item_vec, user_vec, profile_features(t),
                     dict(zip(t.travel_id, range(len(t)))))
    model, idx = fit(train_q, valid_q, feats, "B", feats.names, np.random.default_rng(SEED))
    print("base model iterations", model.best_iteration, flush=True)

    cats = base.attractions[CAT_FIELD].fillna("").to_numpy()
    # 사전 비율 = 추천 후보 전체(카탈로그)의 중분류 비율. 서비스도 추천 가능 관광지 전체로 같은 값을 계산한다.
    catalog = np.unique(np.concatenate([c for c in base.candidates.values()]))
    pop = pd.Series(cats[catalog]).value_counts(normalize=True).to_dict()
    adj = Adjust(feats, cats, pop)

    order = region_order(set(q[0] for q in valid_q + test_q))
    valid_eps, test_eps = episodes(valid_q, order), episodes(test_q, order)
    print("episodes valid", len(valid_eps), "test", len(test_eps), flush=True)

    # λ 선택: TL 안쪽 검증 여행자, 방법별로 NDCG@10 평균이 가장 높은 값
    tuning = []
    for lam in LAMBDAS:
        f = evaluate(valid_eps, model, idx, feats, adj, lam, "asym")
        s = paired(f)
        s["lambda"] = lam
        tuning.append(s)
    tuning = pd.concat(tuning)
    # 진단: 반대 방향(지난 여행과 다른 곳을 올림). 한 여행 안의 대리 실험이 ‘다양성 추구’로 왜곡됐는지 본다.
    diag = []
    for lam in DIAG_LAMBDAS:
        f = evaluate(valid_eps, model, idx, feats, adj, lam, "asym")
        s = paired(f)
        s["lambda"] = lam
        diag.append(s)
    pd.concat(diag).to_csv(OUT / "diagnostic_negative_valid.csv", index=False)
    print(pd.concat(diag).pivot(index="lambda", columns="method", values="value").round(4).to_string(), flush=True)
    tuning.to_csv(OUT / "tuning_valid.csv", index=False)
    best = tuning[tuning.method == "both"].sort_values("value").iloc[-1]["lambda"]
    print("selected lambda", best, flush=True)

    test_users = [ep[0] for ep in test_eps]
    perm = np.random.default_rng(1000).permutation(len(test_eps))
    donors = {test_eps[i][0]: adj.centroid(list(test_eps[j][1][3])) for i, j in enumerate(perm)}
    results = []
    for mode in ["asym", "sym"]:
        f = evaluate(test_eps, model, idx, feats, adj, best, mode, donors)
        for metric in ["ndcg_at_10", "hit_at_10", "recall_at_10"]:
            s = paired(f, metric)
            s["weighting"] = mode
            results.append(s)
        if mode == "asym":
            f.to_csv(OUT / "rows_test_asym.csv", index=False)
            low = f[f.s_trip <= 3]
            if low.travel_id.nunique() >= 30:
                s = paired(low)
                s["weighting"] = "asym · 지난 여행 만족도 ≤3"
                results.append(s)
        else:
            low = f[f.s_trip <= 3]
            if low.travel_id.nunique() >= 30:
                s = paired(low)
                s["weighting"] = "sym · 지난 여행 만족도 ≤3"
                results.append(s)
    results = pd.concat(results)
    results.to_csv(OUT / "summary_test.csv", index=False)
    s_trip = pd.Series([int(np.clip(round(np.mean(list(ep[1][3].values()))), 1, 5)) for ep in test_eps])
    (OUT / "run.json").write_text(json.dumps(dict(
        lambda_selected=float(best), lambdas=LAMBDAS, prior_k=PRIOR_K, category_field=CAT_FIELD,
        base_iterations=model.best_iteration, valid_episodes=len(valid_eps), test_episodes=len(test_eps),
        test_target_queries=int(sum(len(ep[2]) for ep in test_eps)),
        trip_satisfaction_distribution=s_trip.value_counts().sort_index().to_dict()), ensure_ascii=False, indent=2),
        encoding="utf-8")
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
