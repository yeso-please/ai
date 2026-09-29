"""TourAPI(KorService2) areaBasedList2로 지역별 관광 콘텐츠 목록을 받아 CSV로 저장한다.

AI Hub 여행로그 방문지와 매칭하기 위한 기준 목록이다. 결과는 data/raw/tourapi/ 아래(커밋 제외)에 둔다.

키는 로그·명령 인자에 남기지 않는다. 다음 순서로 읽는다.
  1) 환경변수 TOUR_API_KEY
  2) 환경변수 TOUR_API_KEY_FILE이 가리키는 YAML의 tour.api.key

사용 예:
  TOUR_API_KEY_FILE=../hidden-travel/config/application-secret.yaml python scripts/collect_tourapi.py --regions 11 28 41

지역은 법정동 시도코드(lDongRegnCd, 11 서울·28 인천·41 경기 …)로 조회한다. KorService2에서는 옛 areaCode가
비어 있는 콘텐츠가 많아(예: 경복궁) areaCode로 조회하면 절반 이상이 빠진다.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://apis.data.go.kr/B551011/KorService2/areaBasedList2"
# 관광지·문화시설·축제공연행사·레포츠·쇼핑. 숙박(32)·음식점(39)은 관광지 매칭 대상이 아니다.
DEFAULT_CONTENT_TYPES = [12, 14, 15, 28, 38]
FIELDS = ["contentid", "contenttypeid", "title", "addr1", "addr2", "mapx", "mapy",
          "areacode", "sigungucode", "lDongRegnCd", "lDongSignguCd", "cat1", "cat2", "cat3", "firstimage"]


def load_key() -> str:
    key = os.environ.get("TOUR_API_KEY")
    if not key and os.environ.get("TOUR_API_KEY_FILE"):
        text = Path(os.environ["TOUR_API_KEY_FILE"]).read_text(encoding="utf-8")
        # tour:\n  api:\n    key: ... 형태에서 tour 블록의 첫 key만 읽는다.
        match = re.search(r"^tour:\s*\n(?:[ \t]+.*\n)*?[ \t]+key:\s*['\"]?([^'\"\s]+)", text, re.M)
        key = match.group(1) if match else None
    if not key:
        sys.exit("TOUR_API_KEY 또는 TOUR_API_KEY_FILE이 필요합니다.")
    # 인코딩된 키(%2B 등)를 넣었어도 한 번 디코딩해 두면 urlencode가 올바르게 다시 인코딩한다.
    return urllib.parse.unquote(key)


def fetch_page(key: str, region: int, content_type: int, page: int, rows: int) -> dict:
    params = {"serviceKey": key, "MobileOS": "ETC", "MobileApp": "tripin-ai", "_type": "json",
              "arrange": "A", "lDongRegnCd": region, "contentTypeId": content_type,
              "pageNo": page, "numOfRows": rows}
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            return data["response"]["body"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                sys.exit("호출 한도 초과(429). 다음에 이어서 실행하세요.")
            err = f"HTTP {e.code}"
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            err = type(e).__name__
        time.sleep(2 * (attempt + 1))
    sys.exit(f"요청 실패: region={region} type={content_type} page={page} ({err})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=int, nargs="+", required=True, help="법정동 시도코드 lDongRegnCd (11 서울, 28 인천, 41 경기 …)")
    parser.add_argument("--types", type=int, nargs="+", default=DEFAULT_CONTENT_TYPES)
    parser.add_argument("--out", default="data/raw/tourapi")
    parser.add_argument("--rows", type=int, default=1000)
    args = parser.parse_args()

    key = load_key()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = 0
    for region in args.regions:
        items = []
        for content_type in args.types:
            page = 1
            while True:
                body = fetch_page(key, region, content_type, page, args.rows)
                calls += 1
                raw = body.get("items") or {}
                page_items = raw.get("item", []) if isinstance(raw, dict) else []
                if isinstance(page_items, dict):
                    page_items = [page_items]
                items.extend(page_items)
                if page * args.rows >= int(body.get("totalCount", 0)):
                    break
                page += 1
            print(f"region={region} type={content_type} 누적 {len(items)}건")
        path = out_dir / f"region_{region}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(items)
        print(f"저장: {path} ({len(items)}건)")
    print(f"총 호출 {calls}회")


if __name__ == "__main__":
    main()
