"""기성 임베딩의 취향 신호와 인기 편향 확인 (#4 → #5 평가 설계 근거).

"실제로 간 곳을 맞히는가"로 평가하면 유명한 곳에 많이 가는 행동 때문에 인기순이 유리하다(인기 편향).
그래서 방문 예측과 함께, 인기 효과를 줄인 취향 평가를 나란히 본다.

  A. 방문 예측: Validation 여행자의 방문 시군구 안, 상위 10에 만족(4점 이상)한 곳이 있는 비율
     랜덤 / 취향만 / 인기만 / 인기+취향(표준화 가중합)
  B. 알려진 곳 안에서: Training 방문 이력이 있는 후보만, 상위 5
  C. 덜 알려진 정답: Training 방문 2회 이하인 정답만, 후보도 2회 이하만, 상위 10
  D. 취향 쌍 비교: 한 여행자가 실제로 간 곳 중 5점 준 곳과 3점 이하 준 곳을 구별하는 정확도
     (이미 간 곳끼리 비교하므로 "유명해서 갔다"는 효과가 크게 줄어든다)

인기는 Training 분할의 매칭 방문 수로 잰다(Validation 정답이 새지 않게).
학습하지 않은 모델이므로 D는 Training+Validation 전체를 쓴다.

사용 예:
  .venv/Scripts/python scripts/popularity_bias_probe.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path("data/interim/eval")
EXCLUDED_TYPES = {"15", "38"}


def ci(p: float, n: int) -> str:
    return f"±{1.96 * np.sqrt(p * (1 - p) / n):.1%}"


def table(title: str, results: dict[str, list[float]], note: str) -> list[str]:
    n = len(next(iter(results.values())))
    rows = [f"### {title}", "", note, "", f"| 방식 | 비율 (n={n}) |", "|---|---|"]
    for name, values in results.items():
        m = float(np.mean(values))
        rows.append(f"| {name} | {m:.1%} {ci(m, len(values))} |")
    return rows + [""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", default="mminilm-l12-v1")
    parser.add_argument("--attraction-template", default="v0")
    parser.add_argument("--traveler-template", default="aihub-v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = Path("data/interim/emb") / args.model_version
    a_vec = np.load(base / f"attractions_{args.attraction_template}.npy")
    a_ids = pd.read_csv(base / f"attractions_{args.attraction_template}.csv", dtype=str)
    t_vec = np.load(base / f"travelers_{args.traveler_template}.npy")
    t_ids = pd.read_csv(base / f"travelers_{args.traveler_template}.csv", dtype=str)
    attractions = pd.read_csv(EVAL_DIR / "attractions.csv", dtype=str).set_index("contentid").loc[a_ids.contentid].reset_index()
    travelers = pd.read_csv(EVAL_DIR / "travelers.csv", dtype=str).set_index("travel_id").loc[t_ids.travel_id].reset_index()
    visits = pd.read_csv(EVAL_DIR / "visits.csv", dtype=str)
    visits["s"] = pd.to_numeric(visits.satisfaction)

    a_index = {c: i for i, c in enumerate(attractions.contentid)}
    t_index = {t: i for i, t in enumerate(travelers.travel_id)}
    train_ids = set(travelers.travel_id[travelers.split == "TL"])
    popularity = attractions.contentid.map(visits[visits.travel_id.isin(train_ids)].contentid.value_counts()).fillna(0).values
    is_candidate = ~attractions.contenttypeid.isin(EXCLUDED_TYPES).values
    region_key = (attractions.lDongRegnCd + "-" + attractions.lDongSignguCd).values
    rng = np.random.default_rng(args.seed)

    positives = visits[(visits.s >= 4) & ~visits.contenttypeid.isin(EXCLUDED_TYPES)
                       & visits.travel_id.isin(travelers.travel_id[travelers.split == "VL"])]
    positives = positives.assign(key=positives.lDongRegnCd + "-" + positives.lDongSignguCd)

    a_res = {"랜덤": [], "취향 임베딩만": [], "인기만": [], "인기+취향 (λ=0.25)": [], "인기+취향 (λ=0.5)": [], "인기+취향 (λ=1.0)": []}
    b_res = {"랜덤": [], "인기순": [], "취향순": []}
    c_res = {"랜덤": [], "인기순": [], "취향순": []}
    hit = lambda gold, chosen: float(bool(gold & set(chosen)))  # noqa: E731
    for (travel_id, key), group in positives.groupby(["travel_id", "key"]):
        cands = np.where(is_candidate & (region_key == key))[0]
        gold = {a_index[c] for c in group.contentid if c in a_index}
        if len(cands) < 20 or not gold:
            continue
        taste = a_vec[cands] @ t_vec[t_index[travel_id]]
        pop = popularity[cands]
        a_res["랜덤"].append(hit(gold, rng.choice(cands, 10, replace=False)))
        a_res["취향 임베딩만"].append(hit(gold, cands[np.argsort(-taste)[:10]]))
        a_res["인기만"].append(hit(gold, cands[np.lexsort((-taste, -pop))[:10]]))
        tz = (taste - taste.mean()) / (taste.std() + 1e-9)
        pz = (np.log1p(pop) - np.log1p(pop).mean()) / (np.log1p(pop).std() + 1e-9)
        for lam in (0.25, 0.5, 1.0):
            a_res[f"인기+취향 (λ={lam})"].append(hit(gold, cands[np.argsort(-(pz + lam * tz))[:10]]))

        known = pop > 0
        gold_known = gold & set(cands[known])
        if known.sum() >= 10 and gold_known:
            k_cands, k_taste, k_pop = cands[known], taste[known], pop[known]
            b_res["랜덤"].append(hit(gold_known, rng.choice(k_cands, 5, replace=False)))
            b_res["인기순"].append(hit(gold_known, k_cands[np.argsort(-k_pop)[:5]]))
            b_res["취향순"].append(hit(gold_known, k_cands[np.argsort(-k_taste)[:5]]))

        tail = pop <= 2
        gold_tail = {g for g in gold if popularity[g] <= 2}
        if gold_tail and tail.sum() >= 20:
            t_cands, t_taste, t_pop = cands[tail], taste[tail], pop[tail]
            c_res["랜덤"].append(hit(gold_tail, rng.choice(t_cands, 10, replace=False)))
            c_res["인기순"].append(hit(gold_tail, t_cands[np.lexsort((-t_taste, -t_pop))[:10]]))
            c_res["취향순"].append(hit(gold_tail, t_cands[np.argsort(-t_taste)[:10]]))

    d_res = {"동전 던지기": [], "취향 임베딩": [], "인기 (방문 수)": []}
    pairs_visits = visits[~visits.contenttypeid.isin(EXCLUDED_TYPES) & visits.contentid.isin(a_index)]
    pairs_visits = pairs_visits.drop_duplicates(["travel_id", "contentid"])
    travelers_with_pairs = 0
    for travel_id, group in pairs_visits.groupby("travel_id"):
        high, low = group[group.s == 5].contentid.tolist(), group[group.s <= 3].contentid.tolist()
        if not high or not low or travel_id not in t_index:
            continue
        travelers_with_pairs += 1
        user = t_vec[t_index[travel_id]]
        for h in high:
            for lo in low:
                d_res["동전 던지기"].append(float(rng.random() < 0.5))
                d_res["취향 임베딩"].append(float(a_vec[a_index[h]] @ user > a_vec[a_index[lo]] @ user))
                ph, pl = popularity[a_index[h]], popularity[a_index[lo]]
                d_res["인기 (방문 수)"].append(1.0 if ph > pl else 0.5 if ph == pl else 0.0)

    tag = f"{args.model_version}_{args.attraction_template}_{args.traveler_template}"
    report = [f"# 인기 편향과 취향 신호 확인: {tag}", "",
              "기성 임베딩(학습 전)이 취향 신호를 갖는지, 평가 방식에 따라 인기순이 얼마나 유리한지 본다. "
              "인기는 Training 분할의 매칭 방문 수. 괄호는 95% 신뢰구간(정규 근사). 축제·쇼핑 제외.", "",
              "## 방문 예측 (인기 편향이 큰 평가)", "",
              *table("A. 방문 시군구 안 상위 10 적중 (Validation)", a_res, "상위 10곳 안에 실제로 만족(4점 이상)한 곳이 하나라도 있는 비율."),
              *table("B. 알려진 곳 안에서 상위 5 적중", b_res, "Training 방문 이력이 있는 후보만 두고, 정답도 그 안에 있을 때."),
              *table("C. 덜 알려진 정답 상위 10 적중", c_res, "Training 방문 2회 이하인 정답만, 후보도 2회 이하인 곳만."),
              "## 취향 평가 (인기 효과를 줄인 평가)", "",
              *table(f"D. 같은 여행자 안에서 만족도 쌍 비교 ({travelers_with_pairs}명)", d_res,
                     "한 여행자가 실제로 간 곳 중 5점 준 곳이 3점 이하 준 곳보다 높은 점수를 받은 비율. 50%가 무작위 수준."),
              "## 해석 (2026-09-29)", "",
              "1. **방문 예측은 인기순이 압도한다.** 사람들은 원래 유명한 곳에 많이 간다(인기 편향). 이 평가를 주 지표로 쓰면 "
              "\"모두에게 대표 명소\"가 최선이 되는데, 이는 취향 반영 랜덤·숨은 여행지라는 서비스 컨셉과 맞지 않는다. "
              "→ 방문 예측은 **참고 지표**로 두고, 취향 평가(D 등)를 주 지표로 쓴다 (#5).",
              "2. **기성 임베딩에는 취향 신호가 없다.** 인기 효과를 줄인 D에서도 무작위 수준이다. 설문 → 벡터 방향은 맞지만"
              "(sanity 리포트), \"이 사람이 어디서 더 만족하는가\"는 담지 못한다. → 실제 여행자 데이터로 학습해야 한다 (#6).",
              "3. **단순 가중합은 인기 신호를 희석한다.** 취향 점수에 신호가 없으니 섞을수록 방문 적중이 떨어진다.",
              "4. **인기는 강한 품질 신호다.** 서비스에는 지금 인기 신호가 없다(TourAPI에 방문자 수가 없음). "
              "순위가 아니라 부적합한 곳을 거르는 **품질 하한선**으로 쓰는 방안을 검토한다.",
              "5. **목표**: 학습한 모델이 D에서 무작위(50%)와 인기(방문 수)를 유의미하게 넘는 것."]
    path = Path("reports") / f"popularity_bias_{tag}.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
