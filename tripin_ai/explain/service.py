"""코스 설명 생성 흐름: 비슷한 관광공사 코스 검색 → LLM 생성 → 검증 → 실패한 부분만 규칙 문장으로 대체."""
from dataclasses import dataclass, field

import numpy as np

from .course_index import CourseIndex, course_vector
from .fallback import personal_reason, rule_intro, rule_reason, rule_title
from .llm import LLMClient, LLMUnavailable
from .prompts import PROMPT_VERSION, SYSTEM, Place, build_user_prompt
from .verify import claim_check, verify


@dataclass
class Explanation:
    title: str
    intro: str
    reasons: dict[str, str]
    personal_reasons: dict[str, str]            # 🔒 본인 전용
    title_source: str                            # "llm" | "rule"
    intro_source: str
    reason_sources: dict[str, str]
    example_course_ids: list[str] = field(default_factory=list)
    problems: dict = field(default_factory=dict)
    llm_error: str | None = None
    prompt_version: str = PROMPT_VERSION
    generator_model: str | None = None


def explain(region_name: str, days: int, places: list[Place], place_vectors: np.ndarray | None,
            index: CourseIndex | None, llm: LLMClient | None, exclude_course_ids: set[str] | None = None,
            n_examples: int = 3, check_claims: bool = False) -> Explanation:
    """check_claims: 소개글의 사실 주장을 LLM으로 한 번 더 확인한다(호출 2배). 근거에 없는 주장이 있으면 소개글을 규칙 문장으로."""
    examples = []
    if index is not None and place_vectors is not None and len(place_vectors):
        examples = index.search(course_vector(place_vectors), k=n_examples, exclude=exclude_course_ids)

    result, error, problems = None, None, {}
    if llm is None:
        error = "NO_LLM"
    else:
        try:
            result = llm.generate_json(SYSTEM, build_user_prompt(region_name, days, places, examples))
        except LLMUnavailable as e:
            error = e.code
    if result is not None:
        problems = verify(result, places, examples, extra_facts=f"{days}일 {days - 1}박 {region_name}")
        if problems["format"]:
            result = None
        elif check_claims and not problems["intro"]:
            unsupported = claim_check(llm, str(result.get("intro", "")), places)
            if unsupported:
                problems["intro"].append("UNSUPPORTED_CLAIM")
                problems["unsupported_claims"] = unsupported

    ok = result is not None
    title = str(result["title"]).strip() if ok and not problems["title"] else rule_title(region_name, days, places)
    intro = str(result["intro"]).strip() if ok and not problems["intro"] else rule_intro(region_name, days, places)
    llm_reasons = {str(r["id"]): str(r["reason"]).strip() for r in result["reasons"]} if ok else {}
    reasons, sources = {}, {}
    for p in places:
        good = ok and not problems["reasons"].get(p.id)
        reasons[p.id] = llm_reasons[p.id] if good else rule_reason(p)
        sources[p.id] = "llm" if good else "rule"
    personal = {p.id: text for p in places if (text := personal_reason(p))}
    return Explanation(
        title=title, intro=intro, reasons=reasons, personal_reasons=personal,
        title_source="llm" if ok and not problems["title"] else "rule",
        intro_source="llm" if ok and not problems["intro"] else "rule",
        reason_sources=sources, example_course_ids=[e.course_id for e in examples], problems=problems,
        llm_error=error, generator_model=getattr(llm, "model", None))
