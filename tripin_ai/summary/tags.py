"""지역·관광지 태그 (yeso-please/backend#89). 고정 사전에서만 고른다.

사전은 2026-10-10 팀 확정안이다. 바꾸면 백엔드 `SummaryTags`와 DB 제약(V21)도 함께 바꾼다.

정하는 순서
  1. 분류코드(lclsSystm3, 없으면 lclsSystm2) 규칙: 데이터로 정해지는 태그. LLM 없이 붙인다.
  2. 설명 키워드 규칙: 분류코드로 알 수 없는 카페·야경·꽃·시장·골목·산책.
  3. 분위기 태그(감성여행·힐링·가족여행·데이트)만 LLM이 사전 안에서 고른다(service.py).
"""
from collections import Counter

DICTIONARY = (
    "바다", "산", "숲", "호수·강", "섬", "꽃",
    "역사", "전통", "박물관", "사찰",
    "시장", "카페", "야경", "골목",
    "액티비티", "산책", "체험",
    "감성여행", "힐링", "가족여행", "데이트",
)
MOOD_TAGS = ("감성여행", "힐링", "가족여행", "데이트")
MAX_TAGS = 4

# 코드 접두어 → 태그. 긴 접두어(세분류)가 먼저 맞으면 그것만 쓴다. 코드표: data/reference/tourapi_lcls_codes.csv
CODE_TAGS: dict[str, tuple[str, ...]] = {
    # 자연
    "NA010100": ("산",), "NA010200": ("숲",), "NA010300": ("산",), "NA010400": ("산",), "NA010500": ("산",),
    "NA020100": ("호수·강",), "NA020200": ("호수·강",), "NA020300": ("호수·강",), "NA020400": ("호수·강",),
    "NA020500": ("섬", "바다"), "NA020600": ("바다",), "NA020700": ("바다",), "NA020800": ("바다",),
    "NA020900": ("바다",),
    "NA030400": ("산책",),
    "NA040100": ("산",), "NA040200": ("산",), "NA040300": ("산",),
    "NA040500": ("산책",), "NA040600": ("숲", "힐링"), "NA040700": ("꽃", "숲"),
    # 역사
    "HS01": ("역사",), "HS010400": ("전통", "역사"), "HS010500": ("역사", "전통"), "HS010600": ("전통", "역사"),
    "HS02": ("역사",), "HS020300": ("사찰", "역사"), "HS030100": ("사찰",), "HS04": ("역사",),
    # 문화
    "VE010200": ("야경",), "VE010800": ("바다",),
    "VE020100": ("액티비티", "가족여행"), "VE020200": ("액티비티", "가족여행"),
    "VE020300": ("가족여행",), "VE020400": ("가족여행", "바다"),
    "VE03": ("산책",), "VE040100": ("골목",), "VE040200": ("골목",), "VE040300": ("산책",),
    "VE070100": ("박물관",), "VE070200": ("박물관", "역사"), "VE070300": ("박물관",), "VE070500": ("박물관", "가족여행"),
    "VE070600": ("박물관",), "VE090400": ("전통",),
    # 체험·레저
    "EX01": ("전통", "체험"), "EX02": ("체험",), "EX03": ("체험", "가족여행"), "EX04": ("사찰", "체험"),
    "EX05": ("힐링",), "EX06": ("체험",), "EX070100": ("바다", "체험"), "EX070200": ("체험",),
    "LS01": ("액티비티",), "LS02": ("액티비티",), "LS020400": ("바다", "액티비티"), "LS020600": ("바다", "액티비티"),
    "LS020200": ("호수·강", "액티비티"), "LS020800": ("호수·강", "액티비티"), "LS03": ("액티비티",), "LS04": ("액티비티",),
    # 쇼핑
    "SH060100": ("시장",), "SH060200": ("시장",),
    # 음식
    "FD050100": ("카페",), "FD050200": ("카페", "전통"),
}

# 분류코드로 알 수 없는 것만 설명 키워드로 붙인다. 바다·산처럼 분류가 있는 태그는 설명에 단어가 나와도 붙이지 않는다
# (박물관 설명의 "바다가 보이는" 같은 문장으로 잘못 붙는 것을 막는다).
KEYWORD_TAGS: dict[str, tuple[str, ...]] = {
    "카페": ("카페",),
    "야경": ("야경",),
    "꽃": ("벚꽃", "유채꽃", "수국", "철쭉", "진달래", "튤립", "코스모스", "꽃길", "꽃축제"),
    "시장": ("전통시장", "재래시장"),
    "골목": ("골목",),
    "산책": ("산책로", "둘레길", "산책하기"),
}

# 지역 태그: 지역 관광지 분류 중 이 비율 이상이고 이 수 이상일 때 규칙으로 붙인다.
REGION_MIN_SHARE = 0.08
REGION_MIN_COUNT = 3
REGION_MAX_RULE_TAGS = 3


def code_tags(*codes: str | None) -> list[str]:
    """가장 세분된 코드부터 접두어를 줄여 가며 처음 맞는 규칙을 쓴다."""
    for code in reversed([c for c in codes if c]):
        for length in (8, 6, 4):
            if len(code) >= length and code[:length] in CODE_TAGS:
                return list(CODE_TAGS[code[:length]])
    return []


def keyword_tags(text: str) -> list[str]:
    return [tag for tag, words in KEYWORD_TAGS.items() if any(w in text for w in words)]


def keyword_supported(tag: str, text: str) -> bool:
    return any(w in text for w in KEYWORD_TAGS.get(tag, ()))


def attraction_rule_tags(lcls1: str | None, lcls2: str | None, lcls3: str | None, name: str, description: str) -> list[str]:
    tags = code_tags(lcls1, lcls2, lcls3)
    for tag in keyword_tags(f"{name} {description}"):
        if tag not in tags:
            tags.append(tag)
    return tags[:MAX_TAGS]


def region_rule_tags(class_counts: dict[str, int]) -> list[str]:
    """지역 관광지 분류 분포 → 규칙 태그. 쇼핑(SH)은 면세점·마트가 많아 분모에서 빼고, 시장만 따로 센다."""
    base = sum(n for code, n in class_counts.items() if not code.startswith("SH"))
    scores: Counter[str] = Counter()
    for code, n in class_counts.items():
        for tag in code_tags(code):
            if tag in MOOD_TAGS:
                continue
            scores[tag] += n
    picked = []
    for tag, n in scores.most_common():
        if tag == "시장":
            ok = n >= REGION_MIN_COUNT
        else:
            ok = n >= REGION_MIN_COUNT and base > 0 and n / base >= REGION_MIN_SHARE
        if ok:
            picked.append(tag)
    return picked[:REGION_MAX_RULE_TAGS]


def merge_tags(rule: list[str], llm: list, support_text: str) -> list[str]:
    """규칙 태그 뒤에 LLM이 고른 태그를 붙인다. LLM 태그는 사전 안의 분위기 태그이거나, 근거 글에 키워드가 있을 때만 받는다."""
    rule = [t for t in dict.fromkeys(rule) if t in DICTIONARY]
    tags = rule[:MAX_TAGS - 1]   # 분위기 태그 한 자리는 남겨 둔다
    for tag in llm if isinstance(llm, list) else []:
        tag = str(tag).strip().lstrip("#")
        if len(tags) >= MAX_TAGS or tag in tags or tag not in DICTIONARY:
            continue
        if tag in MOOD_TAGS or keyword_supported(tag, support_text):
            tags.append(tag)
    tags += [t for t in rule if t not in tags][:MAX_TAGS - len(tags)]
    return tags
