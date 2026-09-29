"""오프라인 임베딩 유사도 검증 (#4). 정량 평가(#5) 전에 임베딩이 상식적으로 동작하는지 본다.

검증
  1. 집단 분리: 설문 응답이 다른 집단은 관련 분류 관광지와의 평균 유사도가 달라야 한다
  2. 부정 표현: "물놀이는 싫다"가 오히려 물놀이 관광지와 가까워지는지
  3. 문장 길이: 모델 최대 토큰 수를 넘어 잘리는 비율
  4. 대략 적중률: Validation 여행자의 방문 시군구 안에서 상위 10 적중률 (임베딩 vs 랜덤, 공식 평가는 #5)
  5. 정성 확인: 여행자 몇 명의 추천 상위 10과 실제 만족한 곳 (개인 단위라 커밋하지 않는 파일로)

출력
  reports/sanity_<모델 버전>_<관광지 템플릿>_<회원 템플릿>.md  집계만 (커밋)
  data/interim/reports/qualitative_*.md                       개인 단위 예시 (커밋 제외)

사용 예:
  .venv/Scripts/python scripts/sanity_check.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tripin_ai.encoder import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION, Encoder  # noqa: E402

EVAL_DIR = Path("data/interim/eval")
CANDIDATE_EXCLUDED_TYPES = {"15", "38"}   # 평가 기본값: 축제·쇼핑 제외
WATER_CLASSES = {"LS020700", "LS020900", "LS021000", "LS021200", "LS021400", "NA010400", "NA020900", "VE020200"}
NEGATION_SENTENCES = {
    "좋아함": "물놀이와 수상 레저를 정말 좋아하는 여행자.",
    "중립(언급 없음)": "여행을 좋아하는 여행자.",
    "싫어함": "물놀이와 수상 레저는 싫어하는 여행자.",
    "제외 조건 나열(현재 백엔드 방식)": "제외 조건: 물놀이",
}


def load(model_version: str, a_tpl: str, t_tpl: str):
    base = Path("data/interim/emb") / model_version
    a_vec = np.load(base / f"attractions_{a_tpl}.npy")
    a_ids = pd.read_csv(base / f"attractions_{a_tpl}.csv", dtype=str)
    t_vec = np.load(base / f"travelers_{t_tpl}.npy")
    t_ids = pd.read_csv(base / f"travelers_{t_tpl}.csv", dtype=str)
    attractions = pd.read_csv(EVAL_DIR / "attractions.csv", dtype=str).set_index("contentid").loc[a_ids.contentid]
    attractions = attractions.reset_index().assign(text=a_ids.text.values)
    travelers = pd.read_csv(EVAL_DIR / "travelers.csv", dtype=str).set_index("travel_id").loc[t_ids.travel_id]
    travelers = travelers.reset_index().assign(text=t_ids.text.values)
    visits = pd.read_csv(EVAL_DIR / "visits.csv", dtype=str)
    return a_vec, attractions, t_vec, travelers, visits


def centroid(vectors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return vectors[mask].mean(axis=0)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled)


def group_separation(a_vec, attractions, t_vec, travelers) -> list[str]:
    """여행자 벡터 · 분류 중심 벡터 = 그 분류 관광지와의 평균 코사인 유사도(모든 벡터가 정규화돼 있으므로)."""
    lcls1 = attractions.lcls1_name.values
    overall = t_vec @ centroid(a_vec, np.ones(len(a_vec), bool))
    score = {name: t_vec @ centroid(a_vec, lcls1 == name) - overall
             for name in ["자연관광", "문화관광", "역사관광", "체험관광", "레저스포츠", "쇼핑"]}
    style = {i: pd.to_numeric(travelers[f"style_{i}"]).values for i in (1, 5)}
    motives = travelers[["motive_1", "motive_2", "motive_3"]].apply(pd.to_numeric).values

    rows = ["| 비교 | 지표(관광지 분류와의 상대 유사도) | 집단 A 평균 | 집단 B 평균 | 효과 크기 d | 기대 방향 |", "|---|---|---|---|---|---|"]

    def add(label, metric_name, metric, mask_a, mask_b, expect):
        a, b = metric[mask_a], metric[mask_b]
        d = cohens_d(a, b)
        ok = "✅" if (d > 0) == (expect == "A>B") else "❌"
        rows.append(f"| {label} (A {mask_a.sum()}명 / B {mask_b.sum()}명) | {metric_name} | {a.mean():+.4f} | {b.mean():+.4f} | {d:+.2f} | {expect} {ok} |")

    add("스타일1 자연 선호(1~2) vs 도시 선호(6~7)", "자연관광 − 문화관광", score["자연관광"] - score["문화관광"],
        style[1] <= 2, style[1] >= 6, "A>B")
    add("스타일5 휴양 선호(1~2) vs 체험 선호(6~7)", "체험관광+레저 − 자연관광",
        (score["체험관광"] + score["레저스포츠"]) / 2 - score["자연관광"], style[5] <= 2, style[5] >= 6, "B>A")
    has = lambda code: (motives == code).any(axis=1)  # noqa: E731
    add("동기 '역사와 문화 탐방' 있음 vs 없음", "역사관광", score["역사관광"], has(8), ~has(8), "A>B")
    add("동기 '운동과 건강' 있음 vs 없음", "레저스포츠", score["레저스포츠"], has(6), ~has(6), "A>B")
    add("동기 '휴식과 재충전' 있음 vs 없음", "자연관광", score["자연관광"], has(2), ~has(2), "A>B")
    return rows


def negation(encoder: Encoder, a_vec, attractions) -> list[str]:
    water = attractions.lclsSystm3.isin(WATER_CLASSES).values
    water_c, other_c = centroid(a_vec, water), centroid(a_vec, ~water)
    vectors = encoder.encode(list(NEGATION_SENTENCES.values()))
    rows = [f"물놀이 관련 관광지 {water.sum()}곳(해수욕장·계곡·워터파크·수상레저)과의 평균 유사도에서 나머지 관광지와의 평균을 뺀 값. 클수록 물놀이 쪽으로 끌린다.", "",
            "| 문장 | 물놀이 쪽 상대 유사도 |", "|---|---|"]
    for (label, sentence), vec in zip(NEGATION_SENTENCES.items(), vectors):
        rows.append(f"| {label}: \"{sentence}\" | {vec @ water_c - vec @ other_c:+.4f} |")
    return rows


def text_length(encoder: Encoder, attractions, travelers) -> list[str]:
    tokenizer, limit = encoder.model.tokenizer, encoder.model.max_seq_length
    rows = [f"모델 최대 입력 {limit} 토큰.", "", "| 대상 | 평균 토큰 | 최대 | 잘리는 비율 |", "|---|---|---|---|"]
    for label, texts in (("관광지", attractions.text), ("여행자", travelers.text)):
        lengths = np.array([len(tokenizer.encode(t)) for t in texts])
        rows.append(f"| {label} | {lengths.mean():.1f} | {lengths.max()} | {(lengths > limit).mean():.1%} |")
    return rows


def rough_hit_rate(a_vec, attractions, t_vec, travelers, visits, qualitative_path: Path, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    cand_mask = ~attractions.contenttypeid.isin(CANDIDATE_EXCLUDED_TYPES).values
    region_key = (attractions.lDongRegnCd + "-" + attractions.lDongSignguCd).values
    index_of = {cid: i for i, cid in enumerate(attractions.contentid)}
    t_index = {tid: i for i, tid in enumerate(travelers.travel_id)}
    positives = visits[(pd.to_numeric(visits.satisfaction) >= 4) & ~visits.contenttypeid.isin(CANDIDATE_EXCLUDED_TYPES)]
    positives = positives[positives.travel_id.isin(travelers.travel_id[travelers.split == "VL"])]
    positives = positives.assign(key=positives.lDongRegnCd + "-" + positives.lDongSignguCd)

    hits_emb, hits_rand, sizes, examples = [], [], [], []
    for (travel_id, key), group in positives.groupby(["travel_id", "key"]):
        cands = np.where(cand_mask & (region_key == key))[0]
        gold = {index_of[c] for c in group.contentid if c in index_of}
        if len(cands) < 20 or not gold:
            continue
        scores = a_vec[cands] @ t_vec[t_index[travel_id]]
        top = cands[np.argsort(-scores)[:10]]
        hits_emb.append(bool(gold & set(top)))
        hits_rand.append(bool(gold & set(rng.choice(cands, 10, replace=False))))
        sizes.append(len(cands))
        if len(examples) < 8 and rng.random() < 0.05:
            ranks = {attractions.title.iat[g]: int((scores > a_vec[g] @ t_vec[t_index[travel_id]]).sum()) + 1 for g in gold}
            examples.append((travel_id, key, travelers.text.iat[t_index[travel_id]],
                             [attractions.title.iat[i] for i in top], ranks, len(cands)))

    qualitative_path.parent.mkdir(parents=True, exist_ok=True)
    with qualitative_path.open("w", encoding="utf-8") as f:
        f.write("# 정성 확인 (개인 단위, 커밋 금지)\n\n")
        for travel_id, key, text, top, ranks, n in examples:
            f.write(f"## {travel_id} / 시군구 {key} (후보 {n}곳)\n\n- 설문 문장: {text}\n- 추천 상위 10: {', '.join(top)}\n")
            f.write(f"- 실제 만족한 곳(순위): {', '.join(f'{k}({v}위)' for k, v in ranks.items())}\n\n")
    return [f"Validation 여행자 × 방문 시군구 {len(hits_emb)}쌍 (후보 20곳 이상), 후보 수 중앙값 {int(np.median(sizes))}곳.", "",
            "| 방식 | 상위 10 안에 만족한 곳이 하나라도 있는 비율 |", "|---|---|",
            f"| 랜덤 | {np.mean(hits_rand):.1%} |", f"| 취향 임베딩 | {np.mean(hits_emb):.1%} |", "",
            "공식 평가(인기순·SVD 기준선, nDCG, 신뢰구간)는 #5에서 한다. 여기서는 랜덤보다 나은지만 본다."]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--attraction-template", default="v0")
    parser.add_argument("--traveler-template", default="aihub-v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    a_vec, attractions, t_vec, travelers, visits = load(args.model_version, args.attraction_template, args.traveler_template)
    encoder = Encoder(args.model_name, args.model_version)
    tag = f"{args.model_version}_{args.attraction_template}_{args.traveler_template}"
    sections = {
        "1. 집단 분리": group_separation(a_vec, attractions, t_vec, travelers),
        "2. 부정 표현": negation(encoder, a_vec, attractions),
        "3. 문장 길이": text_length(encoder, attractions, travelers),
        "4. 대략 적중률 (Validation)": rough_hit_rate(a_vec, attractions, t_vec, travelers, visits,
                                                    Path("data/interim/reports") / f"qualitative_{tag}.md", args.seed),
    }
    report = [f"# 유사도 검증: {tag}", "",
              f"- 모델: `{args.model_name}` ({args.model_version}, {a_vec.shape[1]}차원)",
              f"- 관광지 템플릿 `{args.attraction_template}` {len(a_vec)}곳, 회원 템플릿 `{args.traveler_template}` 여행자 {len(t_vec)}명",
              "- 데이터: AI Hub 국내 여행로그 2023 × TourAPI (docs/data.md)", ""]
    for title, rows in sections.items():
        report += [f"## {title}", "", *rows, ""]
    path = Path("reports") / f"sanity_{tag}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
