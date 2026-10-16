"""한 줄 소개·지역 태그라인 프롬프트 (yeso-please/backend#89).

입력은 TourAPI 공개 데이터(이름·분류·지역·설명)뿐이다. 회원 정보는 넣지 않는다.
설명이 없는 관광지(NAME_CATEGORY)는 이름·분류만 주고, 입력에 없는 사실을 쓰지 못하게 한다. 사람이 승인한 것만 화면에 나간다.
"""
import json
from dataclasses import dataclass, field

from .tags import DICTIONARY, MOOD_TAGS

PROMPT_VERSION = "summary-v1"
ONE_LINE_MAX = 50
TAGLINE_MAX = 30
DESCRIPTION_EXCERPT = 300
REGION_DESCRIPTION_EXCERPT = 160


@dataclass
class SummaryPlace:
    id: str
    name: str
    class_names: list[str] = field(default_factory=list)
    region_name: str = ""
    description: str = ""
    codes: tuple[str | None, str | None, str | None] = (None, None, None)   # lclsSystm1~3 (규칙 태그용)

    @property
    def has_description(self) -> bool:
        return len(self.description.strip()) >= 20


COMMON_RULES = f"""[사실 규칙 — 반드시 지킨다]
1. 사실은 입력에 적힌 내용만 쓴다. 장소 이름·숫자·연도·시설·먹거리·역사를 새로 만들지 않는다.
2. "최고", "1위", "유일한", "반드시", "맛집" 같은 단정·과장 표현을 쓰지 않는다.
3. 다른 장소의 이름을 쓰지 않는다.
[태그 규칙]
4. tags는 분위기 태그 {list(MOOD_TAGS)} 중 어울리는 것을 0~2개 고른다. 입력 설명에 분명히 나오면 {[t for t in DICTIONARY if t not in MOOD_TAGS]} 중에서도 고를 수 있다. 목록에 없는 단어는 쓰지 않는다."""

ATTRACTION_SYSTEM = f"""당신은 여행 앱의 관광지 카드에 들어갈 한 줄 소개를 쓰는 여행 작가다. 관광공사 소개글처럼 담백하고 감각적으로 쓴다.

{COMMON_RULES}
[문체 규칙]
5. 한 줄 소개는 한 문장, {ONE_LINE_MAX}자 이내. 장소 이름으로 시작하지 않는다. "~하는 곳", "~할 수 있는 곳"처럼 명사로 끝내도 좋다.
6. "적합하다", "다채로운", "다양한 즐길 거리", "어우러진" 같은 상투적 표현을 쓰지 않는다.
7. basis가 NAME_CATEGORY인 장소는 설명이 없다. 이름과 분류에서 알 수 있는 성격(예: 해수욕장 → 바다를 걷는 곳)만 일반적으로 쓰고, 경관·역사·메뉴·시설을 지어내지 않는다.
8. JSON으로만 답한다: {{"items": [{{"id": "...", "oneLine": "...", "tags": ["..."]}}]}}  입력의 모든 id를 한 번씩 담는다."""

REGION_SYSTEM = f"""당신은 여행 앱의 지도에서 지역을 눌렀을 때 보이는 지역 한 줄 소개(태그라인)를 쓰는 여행 작가다.
예: "바다와 카페 사이, 느긋하게 머무는 곳"

{COMMON_RULES}
[문체 규칙]
5. 태그라인은 {TAGLINE_MAX}자 이내의 짧은 구절. 지역 이름과 관광지 이름을 쓰지 않는다. 대표 관광지들의 공통된 분위기를 감각적으로 담는다.
6. "적합하다", "다채로운", "다양한 즐길 거리", "어우러진" 같은 상투적 표현을 쓰지 않는다.
7. JSON으로만 답한다: {{"tagline": "...", "tags": ["..."]}}"""


def _category(p: SummaryPlace) -> str:
    return " > ".join(c for c in p.class_names if c)


def build_attraction_prompt(places: list[SummaryPlace]) -> str:
    lines = []
    for p in places:
        entry = {"id": p.id, "name": p.name, "category": _category(p), "region": p.region_name,
                 "basis": "SOURCE_SUMMARY" if p.has_description else "NAME_CATEGORY"}
        if p.has_description:
            entry["description"] = " ".join(p.description.split())[:DESCRIPTION_EXCERPT]
        lines.append(json.dumps(entry, ensure_ascii=False))
    return "[장소]\n" + "\n".join(lines) + "\n\n각 장소의 한 줄 소개와 태그를 JSON으로 써라."


def build_region_prompt(rule_tags: list[str], places: list[SummaryPlace]) -> str:
    lines = [json.dumps({"name": p.name, "category": _category(p),
                         "description": " ".join(p.description.split())[:REGION_DESCRIPTION_EXCERPT]}, ensure_ascii=False)
             for p in places]
    return (f"[지역 특징 태그(분류 통계)] {', '.join(rule_tags) or '(없음)'}\n[대표 관광지]\n" + "\n".join(lines) +
            "\n\n이 지역의 태그라인과 분위기 태그를 JSON으로 써라.")
