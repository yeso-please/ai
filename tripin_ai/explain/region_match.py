"""성향 맞춤 지역 문장 (backend 7-6, 디자인 B4 "이 지역이 나와 맞는 이유").

입력: 여행 성향 이름·한 줄 설명·선호(성향 콘텐츠), 지역명, 승인된 지역 태그·태그라인, 대표 관광지(이름·한 줄 소개).
🔒 회원 설문 답·닉네임은 받지 않는다. 사실은 지역 근거에 있는 것만 쓴다. 검증에 걸리면 규칙 문장으로 대신한다.
"""
import json
from dataclasses import dataclass, field

from .fallback import short_region
from .llm import LLMClient, LLMUnavailable
from .verify import check_text

PROMPT_VERSION = "region-match-v1"
TEXT_MAX = 160


@dataclass
class Landmark:
    name: str
    one_line: str = ""
    category: str = ""


@dataclass
class RegionMatchInput:
    persona_name: str
    persona_summary: str
    persona_likes: list[str]
    region_name: str
    tagline: str = ""
    tags: list[str] = field(default_factory=list)
    landmarks: list[Landmark] = field(default_factory=list)


@dataclass
class RegionMatchResult:
    text: str
    source: str            # "llm" | "rule"
    problems: list[str] = field(default_factory=list)
    llm_error: str | None = None
    generator_model: str | None = None
    prompt_version: str = PROMPT_VERSION


SYSTEM = f"""당신은 여행 앱에서 "이 지역이 당신과 잘 맞는 이유"를 쓰는 여행 작가다.

[사실 규칙 — 반드시 지킨다]
1. 지역의 사실은 [지역 근거]에 있는 것만 쓴다. 장소 이름·숫자·연도·시설·먹거리를 새로 만들지 않는다.
2. "최고", "1위", "유일한", "반드시", "맛집" 같은 단정·과장 표현을 쓰지 않는다.
3. 여행자를 부르는 이름·닉네임을 쓰지 않는다. 앞에 화면이 "OO님은"을 붙인다.
[문체]
4. 2~3문장, {TEXT_MAX}자 이내. 여행자 성향과 지역 근거의 공통점을 감각적으로 잇는다. 권하는 말투를 한 번 쓴다.
5. "적합하다", "다채로운", "다양한 즐길 거리", "어우러진" 같은 상투적 표현을 쓰지 않는다.
6. JSON으로만 답한다: {{"text": "..."}}"""


def build_prompt(data: RegionMatchInput) -> str:
    region = short_region(data.region_name) or data.region_name
    evidence = {
        "region": region,
        "tagline": data.tagline,
        "tags": data.tags,
        "landmarks": [{"name": l.name, "oneLine": l.one_line} for l in data.landmarks],
    }
    persona = {"name": data.persona_name, "summary": data.persona_summary, "likes": data.persona_likes[:8]}
    return (f"[여행자 성향]\n{json.dumps(persona, ensure_ascii=False)}\n\n"
            f"[지역 근거]\n{json.dumps(evidence, ensure_ascii=False)}\n\n이 지역이 이 여행자와 잘 맞는 이유를 JSON으로 써라.")


def rule_text(data: RegionMatchInput) -> str:
    region = short_region(data.region_name) or data.region_name
    if data.landmarks:
        names = ", ".join(l.name for l in data.landmarks[:2])
        return f"{data.persona_name}에게 {region}의 {names}을(를) 추천해요."
    if data.tags:
        return f"{data.persona_name}에게 #{' #'.join(data.tags[:3])} 여행지 {region}을(를) 추천해요."
    return f"{data.persona_name}의 다음 여행지로 {region}은(는) 어떨까요?"


class _Fact:
    """check_text가 근거 문자열을 만들 때 쓰는 최소 장소 모양(name·description·region_name)."""

    def __init__(self, name: str, description: str, region_name: str):
        self.name, self.description, self.region_name = name, description, region_name


def match_region(data: RegionMatchInput, llm: LLMClient | None) -> RegionMatchResult:
    if llm is None:
        return RegionMatchResult(rule_text(data), "rule", llm_error="NO_LLM")
    try:
        result = llm.generate_json(SYSTEM, build_prompt(data))
    except LLMUnavailable as e:
        return RegionMatchResult(rule_text(data), "rule", llm_error=e.code, generator_model=getattr(llm, "model", None))
    text = " ".join(str(result.get("text", "")).split()) if isinstance(result, dict) else ""
    facts = [_Fact(l.name, l.one_line, data.region_name) for l in data.landmarks]
    facts.append(_Fact(data.tagline, " ".join(data.tags), data.region_name))
    problems = check_text(text, TEXT_MAX, facts, [], extra_facts=f"{data.persona_name} {data.persona_summary}")
    if not text:
        problems.append("FORMAT")
    if problems:
        return RegionMatchResult(rule_text(data), "rule", problems=problems, generator_model=getattr(llm, "model", None))
    return RegionMatchResult(text, "llm", generator_model=getattr(llm, "model", None))

