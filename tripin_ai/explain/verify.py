"""생성 결과 검증. 하나라도 걸리면 그 부분은 규칙 문장으로 대체한다.

규칙 검사(LLM 없이 결정적)
  FORMAT          JSON 모양·id 누락·중복
  LENGTH          제목·소개·이유 길이
  BANNED_WORD     단정·과장 표현
  FOREIGN_PLACE   예시 코스에만 있는 장소 이름이 나옴 (예시의 사실을 옮김)
  UNKNOWN_NUMBER  근거에 없는 숫자 (연도·거리·높이 등을 지어냄)
  COPIED_EXAMPLE  예시 소개글과 긴 구절이 겹침 (문체가 아니라 문장을 베낌)
"""
import re

from .course_index import CourseExample
from .prompts import INTRO_MAX, REASON_MAX, TITLE_MAX, Place

BANNED = ("최고", "1위", "유일한", "유일무이", "반드시", "꼭 가야", "맛집", "무조건", "완벽한")
COPY_NGRAM = 12   # 공백 제외 12글자 이상 똑같이 겹치면 베낀 것으로 본다


def _chars(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _ngrams(text: str, n: int) -> set[str]:
    t = _chars(text)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def check_text(text: str, max_len: int, places: list[Place], examples: list[CourseExample], extra_facts: str = "") -> list[str]:
    problems = []
    if not text or len(text) > max_len:
        problems.append("LENGTH")
    if any(word in text for word in BANNED):
        problems.append("BANNED_WORD")
    ours = {p.name for p in places}
    foreign = {name for ex in examples for name in ex.stop_names if name and len(name) >= 2 and name not in ours}
    if any(name in text for name in foreign if not any(name in o for o in ours)):
        problems.append("FOREIGN_PLACE")
    facts = " ".join(f"{p.name} {p.description} {p.region_name}" for p in places) + " " + extra_facts
    if any(num not in facts for num in re.findall(r"\d+", text)):
        problems.append("UNKNOWN_NUMBER")
    grams = _ngrams(text, COPY_NGRAM)
    if any(grams & _ngrams(ex.overview, COPY_NGRAM) for ex in examples):
        problems.append("COPIED_EXAMPLE")
    return problems


def verify(result: dict, places: list[Place], examples: list[CourseExample], extra_facts: str = "") -> dict:
    """extra_facts: 장소 밖에서 써도 되는 사실(예: 코스 일수 "2일").
    반환: {"title": [...문제], "intro": [...], "reasons": {id: [...]}, "format": [...]}"""
    report = {"format": [], "title": [], "intro": [], "reasons": {}}
    if not isinstance(result, dict) or not isinstance(result.get("reasons"), list):
        report["format"].append("FORMAT")
        return report
    report["title"] = check_text(str(result.get("title", "")), TITLE_MAX, places, examples, extra_facts)
    report["intro"] = check_text(str(result.get("intro", "")), INTRO_MAX, places, examples, extra_facts)
    ids = [str(r.get("id")) for r in result["reasons"] if isinstance(r, dict)]
    if sorted(ids) != sorted(p.id for p in places) or len(set(ids)) != len(ids):
        report["format"].append("FORMAT")
    by_id = {str(r.get("id")): str(r.get("reason", "")) for r in result["reasons"] if isinstance(r, dict)}
    for p in places:
        report["reasons"][p.id] = check_text(by_id.get(p.id, ""), REASON_MAX, places, examples, extra_facts)
    return report


CLAIM_SYSTEM = """너는 여행 소개글의 사실 검증자다. [근거]에 없는 구체적 사실 주장을 찾는다.
- 구체적 사실: 지명·강·산·섬 이름, 시설·먹거리·체험 이름, 역사·연도·수치, "~가 있다/~할 수 있다" 같은 구체적 주장
- 허용: 바람·빛·파도·분위기·느낌 같은 감각·감정 표현, 근거에 있는 사실을 비유한 표현(예: 케이블카 → 하늘길)
JSON으로만 답한다: {"unsupported": ["근거에 없는 구절", ...]}  없으면 빈 목록."""


def claim_check(llm, text: str, places: list[Place]) -> list[str]:
    """LLM으로 근거에 없는 사실 주장을 찾는다(선택 기능, 호출이 한 번 더 든다). 실패하면 빈 목록(규칙 검사만 믿는다)."""
    from .llm import LLMUnavailable

    facts = "\n".join(f"- {p.name}: {' '.join(p.description.split())[:300]}" for p in places)
    try:
        result = llm.generate_json(CLAIM_SYSTEM, f"[근거]\n{facts}\n\n[소개글]\n{text}")
    except LLMUnavailable:
        return []
    unsupported = result.get("unsupported", []) if isinstance(result, dict) else []
    return [str(u) for u in unsupported if str(u).strip()]
