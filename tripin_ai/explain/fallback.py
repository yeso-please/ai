"""LLM 없이 만드는 규칙 문장. 생성 실패·검증 실패·키 없음일 때 쓰고, 본인 전용 문장도 여기서 만든다."""
from collections import Counter

from ..templates import _has_final_consonant
from .prompts import Place


def _eul(word: str) -> str:
    return word + ("을" if _has_final_consonant(word) else "를")


def _ieyo(word: str) -> str:
    return word + ("이에요" if _has_final_consonant(word) else "예요")


THEME_BY_CLASS = {"자연관광": "자연", "역사관광": "역사", "문화관광": "문화", "체험관광": "체험", "레저스포츠": "레저"}


def short_region(region_name: str) -> str:
    """ "강원특별자치도 강릉시" → "강릉", "서울특별시 종로구" → "종로구"(두 글자 구는 그대로). """
    last = region_name.split()[-1] if region_name.split() else ""
    return last[:-1] if len(last) > 2 and last[-1] in "시군" else last


def rule_title(region_name: str, days: int, places: list[Place]) -> str:
    themes = Counter(THEME_BY_CLASS.get(p.class_names[0], "") for p in places if p.class_names)
    theme = next((t for t, _ in themes.most_common() if t), "")
    region = short_region(region_name) or region_name
    return f"{region}, {_eul(theme)} 따라 걷는 {days}일" if theme else f"{region}에서 보내는 {days}일"


def rule_intro(region_name: str, days: int, places: list[Place]) -> str:
    names = [p.name for p in sorted(places, key=lambda p: (p.day, p.order))][:3]
    return f"{short_region(region_name) or region_name}에서 {', '.join(names)} 등을 둘러보는 {days}일 코스예요."


def rule_reason(place: Place) -> str:
    kind = place.class_names[-1] if place.class_names else "명소"
    if place.matched_features:
        return f"{_eul('·'.join(place.matched_features[:2]))} 즐기기 좋은 {_ieyo(kind)}."
    return f"{short_region(place.region_name) or '이 지역'}의 {_ieyo(kind)}."


def personal_reason(place: Place) -> str | None:
    """🔒 취향 기준 회원 본인에게만 보여 준다(다른 참여자·공유 링크에는 숨김)."""
    if not place.closest_liked_region:
        return None
    return f"좋아하신 {place.closest_liked_region} 여행과 비슷한 분위기예요."
