"""추천 평가 실행 (#5). 기준선과 취향 모델을 같은 표본·같은 지표로 비교해 리포트를 만든다.

사용 예:
  .venv/Scripts/python scripts/evaluate.py
  .venv/Scripts/python scripts/evaluate.py --model-version mminilm-l12-v1 --attraction-template v0 --traveler-template aihub-v1

출력
  reports/eval_<모델 버전>_<관광지 템플릿>_<회원 템플릿>.md   결과 표 (커밋)
  data/interim/eval_results/<같은 이름>.json                  수치 원본 (다른 실험과 비교용)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tripin_ai.evaluation import data as eval_data  # noqa: E402
from tripin_ai.evaluation.metrics import bootstrap  # noqa: E402
from tripin_ai.evaluation.runner import METRICS, evaluate  # noqa: E402
from tripin_ai.evaluation.scorers import (FoldTasteScorer, ItemSimilarityScorer,  # noqa: E402
                                          PopularityFloorTasteScorer, PopularityScorer, PureSVDScorer,
                                          RandomScorer, TasteScorer)

CONDITIONS = {"cold": "콜드 스타트 (설문만, 서비스 신규 회원)", "warm": "웜 스타트 (방문 절반 공개)"}
BASELINE = "인기"


def load_fold_vectors(version: str, folds) -> dict:
    """finetune.py 산출물: 겹마다 (관광지 벡터 전체, {travel_id: 여행자 벡터})."""
    base = Path("data/interim/emb") / version
    vectors = {}
    for fold in folds:
        if not (base / f"fold{fold}_attractions.npy").exists():
            continue
        ids = pd.read_csv(base / f"fold{fold}_travelers.csv", dtype=str).travel_id
        t_vec = np.load(base / f"fold{fold}_travelers.npy")
        vectors[fold] = (np.load(base / f"fold{fold}_attractions.npy"), dict(zip(ids, t_vec)))
    if not vectors:
        raise SystemExit(f"{base}에 겹별 벡터가 없습니다.")
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", default="mminilm-l12-v1")
    parser.add_argument("--attraction-template", default="v0")
    parser.add_argument("--traveler-template", default="aihub-v1")
    parser.add_argument("--include-shopping", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--finetuned", nargs="*", default=[], help="scripts/finetune.py가 만든 버전들 (겹별 벡터)")
    parser.add_argument("--only-folds", type=int, nargs="*", help="이 겹만 평가 (파일럿용)")
    args = parser.parse_args()

    emb = Path("data/interim/emb") / args.model_version
    a_ids = pd.read_csv(emb / f"attractions_{args.attraction_template}.csv", dtype=str).contentid.tolist()
    t_ids = pd.read_csv(emb / f"travelers_{args.traveler_template}.csv", dtype=str).travel_id.tolist()
    a_vec = np.load(emb / f"attractions_{args.attraction_template}.npy")
    t_vec = np.load(emb / f"travelers_{args.traveler_template}.npy")

    data = eval_data.load(a_ids, include_shopping=args.include_shopping, n_folds=args.folds, seed=args.seed)
    n = len(data.attractions)
    taste = TasteScorer(f"취향 임베딩 ({args.model_version})", a_vec, t_vec, t_ids)
    scorers = [RandomScorer(args.seed), PopularityScorer(n), taste, PopularityFloorTasteScorer(taste, n),
               ItemSimilarityScorer(f"콘텐츠 유사도 ({args.model_version})", a_vec), PureSVDScorer(n)]
    for version in args.finetuned:
        vectors = load_fold_vectors(version, args.only_folds or range(args.folds))
        tuned = FoldTasteScorer(f"학습한 취향 ({version})", vectors)
        scorers += [tuned, PopularityFloorTasteScorer(tuned, n)]

    start = time.perf_counter()
    units = evaluate(data, scorers, seed=args.seed, folds=args.only_folds)
    elapsed = time.perf_counter() - start

    tag = f"{args.model_version}_{args.attraction_template}_{args.traveler_template}" + ("_shop" if args.include_shopping else "")
    if args.finetuned:
        tag += "__" + "+".join(args.finetuned)
    if args.only_folds:
        tag += "_fold" + "".join(map(str, args.only_folds))
    results = {f"{c}/{m}": bootstrap(units[(c, m)], BASELINE, n_boot=args.bootstrap, seed=args.seed)
               for c in CONDITIONS for m in METRICS if (c, m) in units}

    lines = [f"# 추천 평가: {tag}", "",
             f"- 모델 `{args.model_version}`, 관광지 템플릿 `{args.attraction_template}`, 회원 템플릿 `{args.traveler_template}`",
             f"- 데이터: AI Hub 국내 여행로그 2023 × TourAPI (docs/data.md). 매칭 방문이 있는 여행자 {len(data.fold_of)}명, "
             f"여행자 단위 {args.folds}겹 교차검증 (시드 {args.seed})"
             + (f", **평가한 겹: {args.only_folds}**" if args.only_folds else ""),
             f"- 후보: 방문 시군구의 TourAPI 관광지, 축제{'' if args.include_shopping else '·쇼핑'} 제외. 인기 = 학습 묶음의 매칭 방문 수",
             f"- 괄호: 여행자 단위 부트스트랩 95% 신뢰구간 ({args.bootstrap}회). '인기 대비'는 같은 표본의 짝지은 차이",
             f"- 실행 시간 {elapsed:.0f}초", ""]
    for condition, label in CONDITIONS.items():
        lines += [f"## {label}", ""]
        for metric, metric_label in METRICS.items():
            result = results.get(f"{condition}/{metric}")
            if not result:
                continue
            any_entry = next(iter(result.values()))
            lines += [f"### {metric_label}", "", f"여행자 {any_entry['n_travelers']}명, 단위 {any_entry['n_units']:.0f}개", "",
                      "| 방식 | 값 [95% CI] | 인기 대비 [95% CI] |", "|---|---|---|"]
            pct = metric != "ref_ndcg"
            fmt = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3f}")
            for method, r in result.items():
                diff = ""
                if "diff" in r:
                    sign = "✅" if r["diff_lo"] > 0 else "❌" if r["diff_hi"] < 0 else "≈"
                    diff = f"{r['diff']:+.1%} [{r['diff_lo']:+.1%}, {r['diff_hi']:+.1%}] {sign}" if pct else \
                        f"{r['diff']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}] {sign}"
                lines.append(f"| {method} | {fmt(r['value'])} [{fmt(r['lo'])}, {fmt(r['hi'])}] | {diff} |")
            lines.append("")
    lines += ["✅ 인기보다 유의미하게 높음, ❌ 유의미하게 낮음, ≈ 차이를 구별할 수 없음 (95% CI 기준). "
              "주3 평균 인기 백분위는 높고 낮음 자체가 좋고 나쁨이 아니다(낮을수록 덜 알려진 곳을 추천).", ""]

    report = Path("reports") / f"eval_{tag}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    raw = Path("data/interim/eval_results") / f"eval_{tag}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
