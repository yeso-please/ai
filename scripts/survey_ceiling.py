"""설문 정보의 상한 확인 (#6). 문장 임베딩 없이 설문 값 × 관광지 분류를 GBDT로 직접 학습한다.

학습한 문장 임베딩(주2 54.8%)이 모델의 한계인지, 설문에 담긴 정보의 한계인지 가늠한다.
  - GBDT도 비슷한 수준에서 멈추면 → 한계는 설문 쪽. 성능을 올리려면 온보딩 문항을 바꿔야 한다.
  - GBDT가 훨씬 높으면 → 문장 임베딩 쪽에 개선 여지가 크다.

학습 목표는 평가 주2와 같다: 같은 시군구에서 "만족한 곳(4점 이상)" vs "인기가 거의 같은 안 간 곳 / 3점 이하 준 곳".
인기·지역은 입력에 넣지 않는다(취향 정보만). 평가는 여행자 단위 5겹, 콜드 스타트만.

변형
  A 설문: 여행 스타일 1~8, 동기(다중 선택)
  B 설문+: A + 선호 시도 3곳, 동반 형태, 성별, 연령대

사용 예:
  .venv/Scripts/python scripts/survey_ceiling.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_fold_vectors  # noqa: E402
from tripin_ai.evaluation import data as eval_data  # noqa: E402
from tripin_ai.evaluation.data import popularity  # noqa: E402
from tripin_ai.evaluation.metrics import bootstrap  # noqa: E402
from tripin_ai.evaluation.runner import METRICS, evaluate, popularity_matched  # noqa: E402
from tripin_ai.evaluation.scorers import FoldTasteScorer, PopularityScorer, RandomScorer, Scorer, TasteScorer  # noqa: E402

NEGATIVES_PER_POSITIVE = 4
ATTRACTION_CATEGORICAL = ["contenttypeid", "lclsSystm1", "lclsSystm2", "lclsSystm3"]


def traveler_features(travelers: pd.DataFrame, extended: bool) -> pd.DataFrame:
    t = travelers.set_index("travel_id")
    feats = pd.DataFrame(index=t.index)
    for i in range(1, 9):
        feats[f"style_{i}"] = pd.to_numeric(t[f"style_{i}"], errors="coerce")
    motives = t[["motive_1", "motive_2", "motive_3"]].apply(pd.to_numeric, errors="coerce")
    for code in range(1, 11):
        feats[f"motive_{code}"] = (motives == code).any(axis=1).astype(float)
    if extended:
        for i in range(1, 4):
            feats[f"like_sido_{i}"] = t[f"like_sgg_{i}"].str[:2].astype("category").cat.codes
        feats["companion"] = t["companion"].astype("category").cat.codes
        feats["gender"] = t["gender"].astype("category").cat.codes
        feats["age_grp"] = pd.to_numeric(t["age_grp"], errors="coerce")
    return feats


def attraction_features(attractions: pd.DataFrame) -> pd.DataFrame:
    # 범주형은 정수 코드로 바꿔 GBDT의 범주형 분할을 쓴다(최대 255개 범주).
    return pd.DataFrame({c: attractions[c].astype("category").cat.codes for c in ATTRACTION_CATEGORICAL})


class SurveyGBDTScorer(Scorer):
    def __init__(self, name: str, data, extended: bool, seed: int = 0):
        self.name, self.data, self.seed = name, data, seed
        self.t_feats = traveler_features(data.travelers, extended)
        self.a_feats = attraction_features(data.attractions).values
        self.columns = list(self.t_feats.columns) + ATTRACTION_CATEGORICAL
        self.categorical = [c in ATTRACTION_CATEGORICAL or c.startswith(("like_sido", "companion", "gender")) for c in self.columns]
        self.model = None

    def _rows(self, travel_id: str, items: np.ndarray) -> np.ndarray:
        t = np.repeat(self.t_feats.loc[[travel_id]].values, len(items), axis=0)
        return np.hstack([t, self.a_feats[items]])

    def fit(self, train_visits, fold=None):
        rng = np.random.default_rng(self.seed + (fold or 0))
        pop = popularity(train_visits, len(self.data.attractions))
        X, y = [], []
        for travel_id, g in train_visits.groupby("travel_id"):
            if travel_id not in self.t_feats.index:
                continue
            visited = set(g.a_idx.astype(int))
            items, labels = [], []
            for item, s, key in zip(g.a_idx.astype(int), g.s, g.key):
                if s <= 3:
                    items.append(item), labels.append(0)
                    continue
                if s < 4:
                    continue
                items.append(item), labels.append(1)
                cands = self.data.candidates.get(key, np.array([], int))
                cands = cands[popularity_matched(pop[item], pop[cands]) & ~np.isin(cands, list(visited))]
                for neg in rng.choice(cands, min(NEGATIVES_PER_POSITIVE, len(cands)), replace=False):
                    items.append(int(neg)), labels.append(0)
            if items:
                X.append(self._rows(travel_id, np.array(items)))
                y += labels
        X, y = np.vstack(X), np.array(y)
        self.model = HistGradientBoostingClassifier(categorical_features=self.categorical, max_iter=300,
                                                    learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                                    early_stopping=True, validation_fraction=0.1, random_state=self.seed)
        self.model.fit(X, y)
        print(f"  [{self.name}] fold {fold}: 학습 {len(y)}행 (양성 {int(y.sum())}), 반복 {self.model.n_iter_}")

    def score(self, travel_id, items, history=None):
        if history is not None or travel_id not in self.t_feats.index:
            return None   # 콜드 스타트만
        return self.model.predict_proba(self._rows(travel_id, items))[:, 1]


def main() -> None:
    emb = Path("data/interim/emb/mminilm-l12-v1")
    a_ids = pd.read_csv(emb / "attractions_v0.csv", dtype=str).contentid.tolist()
    t_ids = pd.read_csv(emb / "travelers_aihub-v1.csv", dtype=str).travel_id.tolist()
    data = eval_data.load(a_ids)
    n = len(data.attractions)
    scorers = [
        RandomScorer(42), PopularityScorer(n),
        TasteScorer("기성 취향 임베딩", np.load(emb / "attractions_v0.npy"), np.load(emb / "travelers_aihub-v1.npy"), t_ids),
        FoldTasteScorer("학습한 취향 임베딩", load_fold_vectors("mminilm-l12-ft-b64", range(5))),
        SurveyGBDTScorer("GBDT A: 설문(스타일·동기) × 관광지 분류", data, extended=False),
        SurveyGBDTScorer("GBDT B: 설문 + 선호 시도·동반·성별·연령 × 관광지 분류", data, extended=True),
    ]
    start = time.perf_counter()
    units = evaluate(data, scorers, seed=42)
    print(f"평가 {time.perf_counter() - start:.0f}초")

    lines = ["# 설문 정보의 상한 확인 (콜드 스타트)", "",
             "문장 임베딩 없이 설문 값 × 관광지 분류를 GBDT로 학습해, 학습한 문장 임베딩과 비교한다. "
             "학습 목표는 주2와 같고(만족한 곳 vs 인기가 거의 같은 안 간 곳·3점 이하 준 곳), 인기·지역은 입력에 넣지 않았다. "
             "여행자 단위 5겹, 괄호는 여행자 단위 부트스트랩 95% CI, '학습 임베딩 대비'는 짝지은 차이.", ""]
    for metric in ["m2_pop_matched_auc", "m1_pair_5v3", "m1_pair_all", "ref_hit"]:
        result = bootstrap(units[("cold", metric)], "학습한 취향 임베딩", n_boot=1000, seed=42)
        lines += [f"## {METRICS[metric]}", "", "| 방식 | 값 [95% CI] | 학습 임베딩 대비 [95% CI] |", "|---|---|---|"]
        for method, r in result.items():
            diff = f"{r['diff']:+.1%} [{r['diff_lo']:+.1%}, {r['diff_hi']:+.1%}]" if "diff" in r else ""
            lines.append(f"| {method} | {r['value']:.1%} [{r['lo']:.1%}, {r['hi']:.1%}] | {diff} |")
        lines.append("")
    path = Path("reports/survey_ceiling.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
