"""코스 소개 생성 평가 (#7). "관광공사 코스와 비슷한 품질·감성"을 확인한다.

방법: 소개글이 있는 관광공사 추천코스마다 **장소 목록만** 주고 우리 방식으로 제목·소개글을 생성한다.
그 코스 자신은 문체 예시 검색에서 뺀다(자기 소개글을 베낄 수 없게). 그리고
  1) 자동 지표: 검증 실패율, 규칙 문장 대체율, 생성 소개글 ↔ 코스 장소 임베딩 유사도(진짜 소개글과 비교)
  2) 사람 평가지: 진짜 소개글과 생성 소개글을 A/B 무작위로 섞은 블라인드 CSV (정답 키는 따로 저장)

출력 (data/interim/explain_eval/, 커밋 제외)
  generated.csv   코스별 진짜·생성 제목·소개글, 출처, 검증 문제
  blind_sheet.csv 평가자용: 코스 장소, 소개 A/B, 점수 칸 (사실 정확성·자연스러움·가고 싶은 정도 1~5, 어느 쪽이 관광공사 글인지)
  blind_key.csv   A/B 정답
  metrics.md      자동 지표

사용 예:
  GEMINI_API_KEY_FILE=../hidden-travel/config/application-secret.yaml .venv/Scripts/python scripts/explain_eval.py --n 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tripin_ai.classification import class_names_of  # noqa: E402
from tripin_ai.encoder import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION, Encoder  # noqa: E402
from tripin_ai.explain.course_index import CourseIndex  # noqa: E402
from tripin_ai.explain.llm import default_client  # noqa: E402
from tripin_ai.explain.prompts import Place  # noqa: E402
from tripin_ai.explain.service import explain  # noqa: E402
from tripin_ai.templates import region_from_address  # noqa: E402

COURSE_DIR = Path("data/raw/tourapi/courses")
OUT = Path("data/interim/explain_eval")


def course_places(course_id: str, stops: pd.DataFrame, attractions: pd.DataFrame, descriptions: dict) -> list[Place]:
    places = []
    for n, s in enumerate(stops[stops.course_id == course_id].itertuples(), 1):
        a = attractions.get(s.subcontentid)
        places.append(Place(
            id=s.subcontentid, name=s.subname, day=1, order=n,
            class_names=class_names_of(a["lclsSystm1"], a["lclsSystm2"], a["lclsSystm3"]) if a is not None else [],
            region_name=region_from_address(a["addr1"]) if a is not None else "",
            description=descriptions.get(s.subcontentid, s.subdetailoverview or "")))
    return places


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="평가할 코스 수 (Gemini 호출 수)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=4.0, help="호출 간격(초). 무료 등급 분당 한도 보호")
    parser.add_argument("--check-claims", action="store_true", help="소개글 사실 주장을 LLM으로 한 번 더 확인 (호출 2배)")
    args = parser.parse_args()

    llm = default_client()
    if llm is None:
        raise SystemExit("GEMINI_API_KEY 또는 GEMINI_API_KEY_FILE이 필요합니다.")
    index = CourseIndex(Path("data/interim/course_index") / f"{args.model_version}.npz")
    encoder = Encoder(args.model_name, args.model_version)
    stops = pd.read_csv(COURSE_DIR / "stops.csv", dtype=str).fillna("")
    attractions = {r["contentid"]: r for r in pd.read_csv("data/interim/eval/attractions.csv", dtype=str).fillna("").to_dict("records")}
    descriptions = dict(pd.read_csv("data/interim/descriptions_combined.csv", dtype=str)[["contentid", "description"]].values)

    candidates = [i for i, o in enumerate(index.overviews) if o.strip()]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(candidates, min(args.n, len(candidates)), replace=False)

    rows = []
    for i in chosen:
        course_id = index.course_ids[i]
        places = course_places(course_id, stops, attractions, descriptions)
        if len(places) < 2:
            continue
        region = places[0].region_name or ""
        texts = [f"{p.name}. {' > '.join(p.class_names)}. {p.region_name}. {p.description[:300]}" for p in places]
        vectors = encoder.encode(texts)
        result = explain(region, 1, places, vectors, index, llm, exclude_course_ids={course_id},
                         check_claims=args.check_claims)
        real_sim = float(encoder.encode([index.overviews[i]])[0] @ vectors.mean(axis=0))
        gen_sim = float(encoder.encode([result.intro])[0] @ vectors.mean(axis=0))
        rows.append({"course_id": course_id, "places": " → ".join(p.name for p in places),
                     "real_title": index.titles[i], "real_intro": index.overviews[i],
                     "gen_title": result.title, "gen_intro": result.intro,
                     "title_source": result.title_source, "intro_source": result.intro_source,
                     "reason_llm_ratio": np.mean([s == "llm" for s in result.reason_sources.values()]),
                     "problems": json.dumps(result.problems, ensure_ascii=False), "llm_error": result.llm_error,
                     "examples": ",".join(result.example_course_ids),
                     "sim_real_intro": real_sim, "sim_gen_intro": gen_sim,
                     "unsupported_claims": json.dumps(result.problems.get("unsupported_claims", []), ensure_ascii=False)})
        print(f"{course_id}: {result.intro_source} / {result.title}")
        time.sleep(args.sleep)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "generated.csv", index=False, encoding="utf-8-sig")
    swap = rng.random(len(df)) < 0.5
    sheet = pd.DataFrame({
        "no": range(1, len(df) + 1), "places": df.places,
        "A_title": np.where(swap, df.gen_title, df.real_title), "A_intro": np.where(swap, df.gen_intro, df.real_intro),
        "B_title": np.where(swap, df.real_title, df.gen_title), "B_intro": np.where(swap, df.real_intro, df.gen_intro),
        **{c: "" for c in ["A_사실(1-5)", "A_자연스러움(1-5)", "A_가고싶음(1-5)", "B_사실(1-5)", "B_자연스러움(1-5)",
                           "B_가고싶음(1-5)", "관광공사_글은(A/B)"]}})
    sheet.to_csv(OUT / "blind_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"no": sheet.no, "course_id": df.course_id, "관광공사": np.where(swap, "B", "A")}).to_csv(
        OUT / "blind_key.csv", index=False, encoding="utf-8-sig")

    llm_rate = (df.intro_source == "llm").mean()
    lines = ["# 코스 소개 생성 자동 지표", "",
             f"- 코스 {len(df)}개 (관광공사 추천코스, 자기 코스는 문체 예시에서 제외), 생성 모델 `{llm.model}`",
             f"- 소개글이 LLM 생성으로 통과한 비율: {llm_rate:.0%} (나머지는 검증 실패·호출 실패로 규칙 문장)",
             f"- 제목 통과 {(df.title_source == 'llm').mean():.0%}, 장소별 이유 통과 평균 {df.reason_llm_ratio.mean():.0%}",
             f"- 소개글 ↔ 코스 장소 임베딩 유사도: 관광공사 {df.sim_real_intro.mean():.3f} / 생성 "
             f"{df[df.intro_source == 'llm'].sim_gen_intro.mean():.3f} (LLM 통과분)", ""]
    (OUT / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
