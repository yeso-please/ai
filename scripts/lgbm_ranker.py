"""E5 위에 LightGBM(LambdaMART) 순위 모델을 얹는 실험. CPU.

설계 (2026-10-27 확정안)
  - E5(`e5-base-rating-2022-2023-t4-all-v2`)는 AI Hub TL로만 학습됐다. 그래서 LightGBM도 TL로 학습하고
    TL에 한 번도 나오지 않은 VL 여행자로 평가한다(E5 VL 평가와 같은 분할·후보·지표).
  - 특성: E5 코사인 유사도, 설문 원값 13개, 관광지 유형·분류 3단계(범주형), 설명 임베딩 PCA, 설명 길이.
    트리가 설문 × 장소 상호작용을 학습한다.
  - 학습 목표 두 가지를 같은 조건에서 비교한다.
      A: 같은 여행자·시군구의 방문지끼리 실제 만족도 순서
      B: 같은 여행자·시군구의 후보 전체(미방문 표본 포함). 방문은 만족도 등급, 미방문은 0(불만족이 아니라 미관측)
  - 개인화 확인: 설문 특성을 뺀 모델(장소만), 남의 설문을 넣은 대조군(shuffled).
  - 하이퍼파라미터는 고정, 반복 수만 TL 안쪽 검증(20% 여행자)으로 정한다. VL은 마지막 평가에만 쓴다.

  python scripts/lgbm_ranker.py   # 결과: data/interim/eval_results/lgbm-ranker-v1/
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from personal_ranker import profile_features
from train_combined_t4 import TEXTS, TRAIN_2022, TRAIN_2023, mapped_visits, profile_texts
from tripin_ai.evaluation.data import load
from tripin_ai.evaluation.holdout import query_metrics, summarize

MODEL_DIR = Path("data/interim/models/e5-base-rating-2022-2023-t4-all-v2/final")
OUT = Path("data/interim/eval_results/lgbm-ranker-v1")
CACHE = OUT / "cache"
MIN_CANDIDATES, NEG_PER_QUERY, PCA_DIMS, SEED = 20, 60, 16, 42
CAT_FIELDS = ["contenttypeid", "lclsSystm1", "lclsSystm2", "lclsSystm3"]
SURVEY = ["style_1", "style_3", "style_5", "style_6"] + [f"motive_{i}" for i in range(1, 10)]
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              label_gain=[0, 1, 3, 7, 15], verbose=-1, seed=SEED, num_threads=0)


def prepare():
    a = pd.read_csv(TEXTS, dtype=str).fillna("")
    base = load(a.contentid.tolist())
    frames = []
    for year, folder in [("2022", TRAIN_2022), ("2023", TRAIN_2023)]:
        t = pd.read_csv(folder / "travelers.csv", dtype=str)
        t["year"] = year
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    if set(t.loc[t.split == "TL", "traveler_id"]) & set(t.loc[t.split == "VL", "traveler_id"]):
        raise ValueError("Same respondent in TL and VL")
    index_of = dict(zip(a.contentid, range(len(a))))
    v = pd.concat([mapped_visits(f / "visits.csv", index_of) for f in [TRAIN_2022, TRAIN_2023]], ignore_index=True)
    return a, base, t, v


def build_queries(v, candidates, ids):
    out = []
    for (uid, city), group in v[v.travel_id.isin(ids)].groupby(["travel_id", "key"], sort=True):
        cand = candidates.get(city, np.array([], dtype=int))
        if len(cand) < MIN_CANDIDATES:
            continue
        allowed = set(cand.tolist())
        ratings = {int(i): float(s) for i, s in zip(group.a_idx, group.s) if int(i) in allowed}
        if ratings:
            out.append((uid, city, cand, ratings))
    return out


def e5_vectors(a, t, needed_items, needed_users):
    """학습된 E5로 관광지(passage)·여행자(query) 벡터. 평가와 같은 접두어·최대 길이. 캐시한다."""
    CACHE.mkdir(parents=True, exist_ok=True)
    item_path, user_path = CACHE / "items.npy", CACHE / "users.npy"
    meta = CACHE / "meta.json"
    key = dict(items=sorted(map(int, needed_items)), users=sorted(needed_users))
    if meta.exists() and json.loads(meta.read_text()) == key:
        return np.load(item_path, allow_pickle=True).item(), np.load(user_path, allow_pickle=True).item()
    from sentence_transformers import SentenceTransformer
    info = json.loads((MODEL_DIR / "training_info.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(str(MODEL_DIR), device="cpu")
    model.max_seq_length = info["max_seq_length"]
    items = key["items"]
    av = model.encode(["passage: " + a.text.iloc[i] for i in items], batch_size=64, normalize_embeddings=True,
                      show_progress_bar=True, convert_to_numpy=True)
    texts = profile_texts(t[t.travel_id.isin(needed_users)])
    users = [u for u in key["users"] if texts.get(u)]
    tv = model.encode(["query: " + texts[u] for u in users], batch_size=128, normalize_embeddings=True,
                      show_progress_bar=True, convert_to_numpy=True)
    item_map, user_map = dict(zip(items, av.astype(np.float32))), dict(zip(users, tv.astype(np.float32)))
    np.save(item_path, item_map, allow_pickle=True)
    np.save(user_path, user_map, allow_pickle=True)
    meta.write_text(json.dumps(key))
    return item_map, user_map


class Features:
    def __init__(self, a, attractions, item_vec, user_vec, survey, user_index):
        self.item_vec, self.user_vec, self.survey, self.user_index = item_vec, user_vec, survey, user_index
        self.item_matrix = np.zeros((len(a), 768), np.float32)
        for i, vec in item_vec.items():
            self.item_matrix[int(i)] = vec
        self.cats = np.column_stack([attractions[f].fillna("").astype("category").cat.codes.to_numpy()
                                     for f in CAT_FIELDS]).astype(np.int32)
        ids = sorted(item_vec)
        pca = PCA(PCA_DIMS, random_state=SEED).fit(np.stack([item_vec[i] for i in ids]))
        self.pcs = {i: p for i, p in zip(ids, pca.transform(np.stack([item_vec[i] for i in ids])).astype(np.float32))}
        self.desc_len = a.text.str.len().to_numpy(np.float32)
        self.names = (["e5_cos"] + SURVEY + CAT_FIELDS + [f"desc_pc{i}" for i in range(PCA_DIMS)] + ["desc_len"])

    def rows(self, uid, items, survey_of=None):
        """survey_of: 설문(과 E5 사용자 벡터)을 가져올 여행자. 대조군은 남의 것을 넣는다."""
        who = survey_of or uid
        u = self.user_vec[who]
        e5 = (self.item_matrix[items] @ u)[:, None]
        s = np.repeat(self.survey[self.user_index[who]][None, :], len(items), axis=0)
        pcs = np.stack([self.pcs[int(i)] for i in items])
        return np.hstack([e5, s, self.cats[items].astype(np.float32), pcs, self.desc_len[items][:, None]])


def training_set(qs, feats, rng, objective):
    xs, ys, groups = [], [], []
    for uid, _, cand, ratings in qs:
        if objective == "A":
            items = np.fromiter(ratings, dtype=int)
            if len(items) < 2 or len(set(ratings.values())) < 2:
                continue
            labels = np.array([ratings[int(i)] - 1 for i in items])
        else:
            visited = np.fromiter(ratings, dtype=int)
            pool = np.setdiff1d(cand, visited)
            neg = rng.choice(pool, min(NEG_PER_QUERY, len(pool)), replace=False)
            items = np.concatenate([visited, neg])
            labels = np.concatenate([[ratings[int(i)] - 1 for i in visited], np.zeros(len(neg))])
        xs.append(feats.rows(uid, items)); ys.append(labels.astype(int)); groups.append(len(items))
    return np.vstack(xs), np.concatenate(ys), np.array(groups)


def fit(train_q, valid_q, feats, objective, use_cols, rng):
    xt, yt, gt = training_set(train_q, feats, rng, objective)
    xv, yv, gv = training_set(valid_q, feats, rng, objective)   # 반복 수는 각 목표의 안쪽 검증으로 고른다
    cat_idx = [i for i, n in enumerate(use_cols) if n in CAT_FIELDS]
    cols = [feats.names.index(n) for n in use_cols]
    dtrain = lgb.Dataset(xt[:, cols], yt, group=gt, feature_name=use_cols, categorical_feature=cat_idx)
    dvalid = lgb.Dataset(xv[:, cols], yv, group=gv, reference=dtrain)
    model = lgb.train(PARAMS, dtrain, num_boost_round=1500, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, cols


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

    needed_items = set()
    for _, _, cand, _ in train_q + valid_q + test_q:
        needed_items.update(cand.tolist())
    needed_users = {q[0] for q in train_q + valid_q + test_q}
    item_vec, user_vec = e5_vectors(a, t, needed_items, needed_users)
    keep = lambda qs: [q for q in qs if q[0] in user_vec]
    train_q, valid_q, test_q = keep(train_q), keep(valid_q), keep(test_q)
    survey = profile_features(t)
    feats = Features(a, base.attractions, item_vec, user_vec, survey, dict(zip(t.travel_id, range(len(t)))))

    place_cols = CAT_FIELDS + [f"desc_pc{i}" for i in range(PCA_DIMS)] + ["desc_len"]
    variants = {
        "lgbm_B_full": ("B", feats.names),
        "lgbm_B_place_only": ("B", place_cols),
        "lgbm_B_no_e5": ("B", [n for n in feats.names if n != "e5_cos"]),
        "lgbm_A_full": ("A", feats.names),
    }
    models, log = {}, {}
    for name, (objective, cols) in variants.items():
        model, idx = fit(train_q, valid_q, feats, objective, cols, np.random.default_rng(SEED))
        models[name] = (model, idx)
        log[name] = dict(objective=objective, features=cols, best_iteration=model.best_iteration)
        print(name, log[name]["best_iteration"], flush=True)

    counts = np.zeros(len(a))
    np.add.at(counts, v[v.travel_id.isin(tl)].a_idx.astype(int).to_numpy(), 1)
    test_users = sorted({q[0] for q in test_q})
    shuffled_of = dict(zip(test_users, np.random.default_rng(1000).permutation(test_users)))
    year_of = dict(zip(t.travel_id, t.year))
    rows = []
    for qi, (uid, city, cand, ratings) in enumerate(test_q):
        x = feats.rows(uid, cand)
        xs = feats.rows(uid, cand, survey_of=shuffled_of[uid])
        methods = {"e5_only": x[:, 0], "shuffled_e5": xs[:, 0], "popularity_tl": np.log1p(counts[cand])}
        for name, (model, idx) in models.items():
            methods[name] = model.predict(x[:, idx], num_iteration=model.best_iteration)
        full, idx = models["lgbm_B_full"]
        methods["shuffled"] = full.predict(xs[:, idx], num_iteration=full.best_iteration)
        for name, scores in methods.items():
            m, r = query_metrics(cand, ratings, scores, seed=SEED + qi)
            for metric, value in m.items():
                rows.append(dict(travel_id=uid, year=year_of[uid], method=name, metric=metric, value=value))
            if name == "e5_only":
                for metric, value in r.items():
                    rows.append(dict(travel_id=uid, year=year_of[uid], method="random", metric=metric, value=value))

    summary, _ = summarize(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    # 처음 보는 관광지: TL에서 한 번도 방문되지 않은 후보만으로 다시 평가(인기를 외웠는지, 콘텐츠로 일반화되는지)
    cold_rows = []
    for qi, (uid, city, cand, ratings) in enumerate(test_q):
        cold = cand[counts[cand] == 0]
        cold_ratings = {i: r for i, r in ratings.items() if counts[i] == 0}
        if len(cold) < MIN_CANDIDATES or not cold_ratings:
            continue
        x = feats.rows(uid, cold)
        xs = feats.rows(uid, cold, survey_of=shuffled_of[uid])
        methods = {"e5_only": x[:, 0]}
        for name in ["lgbm_B_full", "lgbm_B_place_only"]:
            model, idx = models[name]
            methods[name] = model.predict(x[:, idx], num_iteration=model.best_iteration)
        full, idx = models["lgbm_B_full"]
        methods["shuffled"] = full.predict(xs[:, idx], num_iteration=full.best_iteration)
        for name, scores in methods.items():
            m, r = query_metrics(cold, cold_ratings, scores, seed=SEED + qi)
            for metric, value in m.items():
                cold_rows.append(dict(travel_id=uid, year=year_of[uid], method=name, metric=metric, value=value))
            if name == "e5_only":
                for metric, value in r.items():
                    cold_rows.append(dict(travel_id=uid, year=year_of[uid], method="random", metric=metric, value=value))
    cold_summary, _ = summarize(cold_rows)
    cold_summary.to_csv(OUT / "summary_cold_items.csv", index=False)
    cv = cold_summary[(cold_summary.subset == "combined") & cold_summary.metric.isin(["ndcg_at_10", "hit_at_10"])]
    print("=== cold items (never visited in TL)")
    print(cv[["metric", "method", "value", "lo", "hi", "n_travelers", "diff_vs_random"]].to_string(index=False))
    model, idx = models["lgbm_B_full"]
    gain = pd.Series(model.feature_importance("gain"), index=[feats.names[i] for i in idx]).sort_values(ascending=False)
    gain.to_csv(OUT / "feature_importance_gain.csv")
    (OUT / "run.json").write_text(json.dumps(dict(variants=log, train_queries=len(train_q), valid_queries=len(valid_q),
                                                  test_queries=len(test_q), test_travelers=len(test_users),
                                                  params=PARAMS, neg_per_query=NEG_PER_QUERY), ensure_ascii=False,
                                             indent=2), encoding="utf-8")
    view = summary[summary.subset == "combined"]
    print(view[["metric", "method", "value", "lo", "hi", "n_travelers", "diff_vs_shuffled", "diff_vs_shuffled_lo",
                "diff_vs_shuffled_hi"]].to_string(index=False))
    print(gain.head(15).to_string())


if __name__ == "__main__":
    main()
