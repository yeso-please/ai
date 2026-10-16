"""한 줄 소개·지역 태그라인 생성: 규칙 태그 → LLM 생성 → 검증. 검증에 걸린 문장은 버린다(null).

코스 소개(explain)와 달리 규칙 문장으로 대신 채우지 않는다. 화면에는 사람이 승인한 문장만 나가므로,
실패한 장소는 다음 배치에서 다시 생성하면 된다. 태그는 문장이 실패해도 규칙 태그를 돌려준다.
"""
from dataclasses import dataclass, field

from ..explain.llm import LLMClient, LLMUnavailable
from ..explain.verify import check_text
from .prompts import (ATTRACTION_SYSTEM, ONE_LINE_MAX, PROMPT_VERSION, REGION_SYSTEM, TAGLINE_MAX, SummaryPlace,
                      build_attraction_prompt, build_region_prompt)
from .tags import attraction_rule_tags, merge_tags, region_rule_tags

SOURCE_SUMMARY = "SOURCE_SUMMARY"
NAME_CATEGORY = "NAME_CATEGORY"


@dataclass
class AttractionSummary:
    id: str
    one_line: str | None
    tags: list[str]
    basis: str
    source: str              # "llm" | "rule"(문장 없음, 규칙 태그만)
    problems: list[str] = field(default_factory=list)


@dataclass
class RegionSummary:
    tagline: str | None
    tags: list[str]
    source: str
    problems: list[str] = field(default_factory=list)


@dataclass
class Batch:
    items: list
    llm_error: str | None = None
    prompt_version: str = PROMPT_VERSION
    generator_model: str | None = None


def _call(llm: LLMClient | None, system: str, user: str) -> tuple[dict | None, str | None]:
    if llm is None:
        return None, "NO_LLM"
    try:
        result = llm.generate_json(system, user)
    except LLMUnavailable as e:
        return None, e.code
    return (result, None) if isinstance(result, dict) else (None, "FORMAT")


def _names_in(text: str, names: list[str]) -> bool:
    return any(len(n) >= 2 and n in text for n in names)


def summarize_attractions(places: list[SummaryPlace], llm: LLMClient | None) -> Batch:
    result, error = _call(llm, ATTRACTION_SYSTEM, build_attraction_prompt(places))
    by_id = {}
    if result is not None and isinstance(result.get("items"), list):
        by_id = {str(r.get("id")): r for r in result["items"] if isinstance(r, dict)}
    elif result is not None:
        error = "FORMAT"
    items = []
    for p in places:
        support = f"{p.name} {p.description}"
        rule = attraction_rule_tags(*p.codes, p.name, p.description)
        basis = SOURCE_SUMMARY if p.has_description else NAME_CATEGORY
        raw = by_id.get(p.id)
        if raw is None:
            items.append(AttractionSummary(p.id, None, merge_tags(rule, [], support), basis, "rule",
                                           [error or "MISSING_ITEM"]))
            continue
        text = " ".join(str(raw.get("oneLine", "")).split())
        problems = check_text(text, ONE_LINE_MAX, [p], [])
        if _names_in(text, [o.name for o in places if o.id != p.id and o.name not in p.name]):
            problems.append("FOREIGN_PLACE")
        ok = not problems
        items.append(AttractionSummary(p.id, text if ok else None,
                                       merge_tags(rule, raw.get("tags", []) if ok else [], support), basis,
                                       "llm" if ok else "rule", problems))
    return Batch(items, error, generator_model=getattr(llm, "model", None))


def summarize_region(class_counts: dict[str, int], places: list[SummaryPlace], llm: LLMClient | None) -> Batch:
    rule = region_rule_tags(class_counts)
    support = " ".join(f"{p.name} {p.description}" for p in places)
    result, error = _call(llm, REGION_SYSTEM, build_region_prompt(rule, places))
    if result is None:
        return Batch([RegionSummary(None, merge_tags(rule, [], support), "rule", [error])], error,
                     generator_model=getattr(llm, "model", None))
    text = " ".join(str(result.get("tagline", "")).split())
    problems = check_text(text, TAGLINE_MAX, places, [])
    if _names_in(text, [p.name for p in places]):
        problems.append("PLACE_NAME")
    ok = not problems
    summary = RegionSummary(text if ok else None, merge_tags(rule, result.get("tags", []) if ok else [], support),
                            "llm" if ok else "rule", problems)
    return Batch([summary], error, generator_model=getattr(llm, "model", None))

