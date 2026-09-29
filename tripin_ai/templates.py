"""회원·관광지 → 임베딩용 문장. 규칙이 바뀌면 버전 이름을 새로 붙인다(기존 벡터와 섞이지 않게).

- attraction_text: 관광지 템플릿. "v0"는 설명 없이 이름·분류·지역, "v1"은 설명 앞부분을 덧붙인다.
- traveler_text:  AI Hub 여행로그 설문 형식 회원 템플릿("aihub-v1"). 온보딩이 이 형식으로 바뀌면
                  (yeso-please/backend#67) 서비스의 회원 템플릿이 된다.
- profile_text:   현재 서비스 온보딩 문항 형식 회원 템플릿("service-v1").

서버 계약의 templateVersion(정수)은 회원·관광지 템플릿 묶음(TEMPLATE_SETS)을 가리킨다.

두 문장 모두 자연어 문장으로 쓴다. 다국어 문장 임베딩 모델은 "키: 값" 나열보다 문장에 익숙하다.
"""
from dataclasses import dataclass, field

ATTRACTION_TEMPLATES = ("v0", "v1", "v2")
TRAVELER_TEMPLATES = ("aihub-v1",)
PROFILE_TEMPLATES = ("service-v1",)
DESCRIPTION_MAX_CHARS = 300
NOTE_MAX_CHARS = 200

# 서버 계약 templateVersion → (회원 템플릿, 관광지 템플릿). 한 번 정한 번호의 규칙은 바꾸지 않는다.
#   1: 구형 온보딩. 관광지는 설명이 있으면 붙인다.
#   2: AI Hub 온보딩. 관광지는 설명과 TourAPI 콘텐츠 유형을 붙인다.
TEMPLATE_SETS = {
    1: ("service-v1", "v1"),
    2: ("aihub-v1", "v2"),
}

CONTENT_TYPE_LABELS = {
    12: "관광지",
    14: "문화시설",
    15: "축제·행사",
    28: "레포츠",
    32: "숙박",
    38: "쇼핑",
    39: "음식점",
}

# 여행 스타일 1~7 척도. 1~3은 왼쪽, 5~7은 오른쪽을 선호, 4는 중립(문장에 넣지 않음).
# 2(숙박↔당일), 4(숙소 가격), 7(계획↔즉흥)은 관광지 선택과 관련이 약해 넣지 않는다.
# 8(사진)은 문항 방향이 확인되지 않아 넣지 않는다. 문구 근거는 docs/data.md 참고.
STYLE_POLES = {
    1: ("자연", "도시"),
    3: ("새로운 지역", "익숙한 지역"),
    5: ("휴양과 휴식", "체험 활동"),
    6: ("잘 알려지지 않은 곳", "잘 알려진 명소"),
}
STRENGTH = {1: "매우", 2: "꽤", 3: "약간", 5: "약간", 6: "꽤", 7: "매우"}

# 여행 동기 코드(AI Hub 'TMT') → 짧은 표현. 10(기타)은 넣지 않는다.
MOTIVES = {
    1: "일상에서 벗어나기",
    2: "휴식과 재충전",
    3: "동반자와 추억 만들기",
    4: "나를 돌아보기",
    5: "SNS에 올릴 사진",
    6: "운동과 건강",
    7: "새로운 경험",
    8: "역사와 문화 탐방",
    9: "특별한 날 기념",
}


@dataclass
class Attraction:
    name: str
    class_names: list[str] = field(default_factory=list)   # 분류 1~3단계 이름
    region_name: str = ""
    description: str = ""
    content_type_id: int | None = None


@dataclass
class LikedTrip:
    region_name: str
    tags: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class ServiceProfile:
    """현재 온보딩 답. 여행 MBTI·일정 밀도·제외 조건은 받지만 문장에는 넣지 않는다.
    - 제외 조건: 임베딩은 "싫다"를 이해하지 못하고 단어 쪽으로 끌려간다(reports/sanity_*). 백엔드가 필터로 처리한다.
    - MBTI·일정 밀도: 관광지 문장에 없는 어휘라 비교에 도움이 적다(#1)."""
    experience_tags: list[str] = field(default_factory=list)
    liked_trips: list[LikedTrip] = field(default_factory=list)
    travel_mbti: str = ""
    schedule_density: str = ""
    exclude_tags: list[str] = field(default_factory=list)


@dataclass
class TravelerSurvey:
    styles: dict[int, int] = field(default_factory=dict)    # 문항 번호 → 1~7
    motives: list[int] = field(default_factory=list)        # 동기 코드, 중요한 순
    liked_regions: list[str] = field(default_factory=list)  # "강원 강릉시" 같은 이름


def attraction_text(item: Attraction, version: str = "v0") -> str:
    if version not in ATTRACTION_TEMPLATES:
        raise ValueError(f"알 수 없는 관광지 템플릿: {version}")
    classes = []
    for name in item.class_names:
        if name and name not in classes:
            classes.append(name)
    parts = [item.name]
    if version == "v2":
        content_type = CONTENT_TYPE_LABELS.get(item.content_type_id)
        if content_type:
            parts.append(content_type)
    if classes:
        parts.append(" > ".join(classes))
    if item.region_name:
        parts.append(item.region_name)
    text = ". ".join(parts) + "."
    if version in ("v1", "v2") and item.description:
        text += " " + " ".join(item.description.split())[:DESCRIPTION_MAX_CHARS]
    return text


def traveler_text(survey: TravelerSurvey, version: str = "aihub-v1") -> str:
    if version not in TRAVELER_TEMPLATES:
        raise ValueError(f"알 수 없는 회원 템플릿: {version}")
    prefs = []
    for number, (left, right) in STYLE_POLES.items():
        value = survey.styles.get(number)
        if value is None or value == 4 or value not in STRENGTH:
            continue
        pole = left if value < 4 else right
        prefs.append(f"{pole}을 {STRENGTH[value]} 선호" if _has_final_consonant(pole) else f"{pole}를 {STRENGTH[value]} 선호")
    sentences = []
    if prefs:
        sentences.append(", ".join(prefs) + "하는 여행자.")
    motives = [MOTIVES[m] for m in survey.motives if m in MOTIVES]
    if motives:
        sentences.append("여행에서 원하는 것은 " + ", ".join(motives) + ".")
    regions = [r for r in dict.fromkeys(survey.liked_regions) if r]
    if regions:
        sentences.append("좋아하는 여행지는 " + ", ".join(regions) + ".")
    return " ".join(sentences)


def profile_text(profile: ServiceProfile, version: str = "service-v1") -> str:
    """같은 답이면 순서와 관계없이 같은 문장이 되도록 정렬한다(같은 입력 → 같은 벡터)."""
    if version not in PROFILE_TEMPLATES:
        raise ValueError(f"알 수 없는 회원 템플릿: {version}")
    sentences = []
    tags = sorted({t.strip() for t in profile.experience_tags if t and t.strip()})
    if tags:
        sentences.append(", ".join(tags) + "을 좋아하는 여행자." if _has_final_consonant(tags[-1])
                         else ", ".join(tags) + "를 좋아하는 여행자.")
    trips = sorted((t for t in profile.liked_trips if t.region_name), key=lambda t: t.region_name)
    if trips:
        described = []
        for trip in trips:
            trip_tags = ", ".join(sorted({t for t in trip.tags if t}))
            described.append(f"{trip.region_name}({trip_tags})" if trip_tags else trip.region_name)
        sentences.append("좋았던 여행지는 " + ", ".join(described) + ".")
        notes = [f"{t.region_name}: {' '.join(t.note.split())}" for t in trips if t.note and t.note.strip()]
        if notes:
            sentences.append(" ".join(notes)[:NOTE_MAX_CHARS])
    return " ".join(sentences)


def region_from_address(address: str) -> str:
    """ "경기도 수원시 팔달구 창룡대로 …" → "경기도 수원시 팔달구", "서울특별시 종로구 …" → "서울특별시 종로구". """
    tokens = str(address or "").split()
    if not tokens:
        return ""
    if tokens[0].startswith("세종"):
        return tokens[0]
    region = tokens[:2]
    if len(tokens) > 2 and tokens[1].endswith("시") and tokens[2].endswith("구"):
        region.append(tokens[2])
    return " ".join(region)


def _has_final_consonant(word: str) -> bool:
    last = word[-1]
    return "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0
