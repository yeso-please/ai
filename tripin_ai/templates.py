"""회원·관광지 → 임베딩용 문장. 규칙이 바뀌면 버전 이름을 새로 붙인다(기존 벡터와 섞이지 않게).

- attraction_text: 관광지 템플릿. "v0"는 설명 없이 이름·분류·지역, "v1"은 설명 앞부분을 덧붙인다.
- traveler_text:  AI Hub 여행로그 설문 형식 회원 템플릿("aihub-v1"). 온보딩이 이 형식으로 바뀌면
                  (yeso-please/backend#67) 서비스의 회원 템플릿이 된다.

두 문장 모두 자연어 문장으로 쓴다. 다국어 문장 임베딩 모델은 "키: 값" 나열보다 문장에 익숙하다.
"""
from dataclasses import dataclass, field

ATTRACTION_TEMPLATES = ("v0", "v1")
TRAVELER_TEMPLATES = ("aihub-v1",)
DESCRIPTION_MAX_CHARS = 300

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
    if classes:
        parts.append(" > ".join(classes))
    if item.region_name:
        parts.append(item.region_name)
    text = ". ".join(parts) + "."
    if version == "v1" and item.description:
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
