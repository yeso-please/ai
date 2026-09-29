"""생성 프롬프트. 문체 예시(관광공사 코스)와 사실 근거(우리 코스 장소)를 분리해 넘긴다.

🔒 회원 설문 답·메모·좋았던 여행지는 넣지 않는다. 장소별 이유의 근거로는 "그 장소의 특징 중 요청자 취향과
   겹치는 것"(예: 바다, 산책)만 넣는다 — 장소를 설명하는 말이라 다른 참여자가 봐도 설문이 드러나지 않는다.
"""
import json
from dataclasses import dataclass, field

from .course_index import CourseExample

PROMPT_VERSION = "course-explain-v2"
TITLE_MAX = 40
INTRO_MAX = 300
REASON_MAX = 80
DESCRIPTION_EXCERPT = 180


@dataclass
class Place:
    id: str
    name: str
    class_names: list[str] = field(default_factory=list)
    region_name: str = ""
    description: str = ""
    day: int = 1
    order: int = 1
    matched_features: list[str] = field(default_factory=list)   # 장소 특징 ∩ 요청자 취향 (예: "바다", "산책")
    closest_liked_region: str = ""   # 🔒 LLM에 보내지 않는다. 본인에게만 보이는 규칙 문장에만 쓴다.


# v2: v1은 사실은 지켰지만 "…둘러보기에 적합하다", "경상남도 거제시에서"처럼 밋밋했다. 관광공사 소개글의
#     특징(풍경으로 여는 첫 문장, 권유형 말투, 장소 이름을 엮는 흐름, 짧은 지역명)을 문체 규칙으로 넣었다.
SYSTEM = f"""당신은 한국관광공사 추천여행코스의 소개글을 쓰는 여행 작가다. 읽는 사람이 당장 떠나고 싶어지게 쓴다.

[사실 규칙 — 반드시 지킨다]
1. 사실은 [코스 장소]에 적힌 내용만 쓴다. 장소 이름·숫자·연도·역사적 사실을 새로 만들지 않는다.
2. [문체 예시]는 말투와 분위기만 참고한다. 예시의 문장·표현을 그대로 옮기지 않고, 예시에 나온 장소나 사실을 쓰지 않는다.
3. "최고", "1위", "유일한", "반드시", "맛집" 같은 단정·과장 표현을 쓰지 않는다.
4. 여행자 개인에 대한 추측(나이·동반자·과거 여행)을 쓰지 않는다.

[문체 규칙 — 관광공사 소개글처럼]
5. 소개글 첫 문장은 코스의 풍경이나 분위기로 연다. "~에서 ~를 만날 수 있는 일정이다", "이 코스는" 같은 안내문 투로 시작하지 않는다.
6. 코스 장소 이름 2~4개를 여행 흐름에 맞게 자연스럽게 엮는다.
7. "~해 보자", "~는 어떨까"처럼 권하는 말투를 한 번 쓴다. 느낌표는 한 번까지.
8. 지역은 짧게 쓴다(예: "강원특별자치도 강릉시" → "강릉").
9. "적합하다", "다채로운", "다양한 즐길 거리", "어우러진" 같은 상투적 표현을 쓰지 않는다. 바람·파도·빛·길 같은 감각적인 표현을 쓴다.
10. 제목은 {TITLE_MAX}자 이내, 코스의 분위기가 드러나는 짧은 문구로 쓴다.
11. 소개글은 2~3문장 {INTRO_MAX}자 이내. 장소별 이유는 그 장소에서 할 수 있는 일이나 느낄 수 있는 것을 한 문장 {REASON_MAX}자 이내로 쓴다.
12. JSON으로만 답한다: {{"title": "...", "intro": "...", "reasons": [{{"id": "...", "reason": "..."}}]}}
    reasons는 [코스 장소]의 모든 id를 같은 순서로 한 번씩 담는다."""


def build_user_prompt(region_name: str, days: int, places: list[Place], examples: list[CourseExample]) -> str:
    example_blocks = []
    for n, ex in enumerate(examples, 1):
        example_blocks.append(f"예시 {n} ({ex.theme})\n제목: {ex.title}\n소개: {' '.join(ex.overview.split())[:400]}")
    place_blocks = []
    for p in sorted(places, key=lambda p: (p.day, p.order)):
        place_blocks.append(json.dumps({
            "id": p.id, "day": p.day, "name": p.name, "category": " > ".join(c for c in p.class_names if c),
            "region": p.region_name, "description": " ".join(p.description.split())[:DESCRIPTION_EXCERPT],
            "features_matching_taste": p.matched_features,
        }, ensure_ascii=False))
    from .fallback import short_region   # fallback이 이 모듈의 Place를 쓰므로 함수 안에서 가져온다
    region = short_region(region_name) or region_name
    return (f"[문체 예시]\n" + ("\n\n".join(example_blocks) or "(없음)") +
            f"\n\n[우리 코스] 지역: {region}, {days}일\n[코스 장소]\n" + "\n".join(place_blocks) +
            "\n\n위 코스의 제목, 소개글, 장소별 추천 이유를 JSON으로 써라.")
