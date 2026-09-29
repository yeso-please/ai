"""TourAPI 추천코스(contentTypeId=25)와 코스별 지점·설명을 수집한다 (#6 코스 데이터 실험).

하루 호출 한도(개발계정 1,000회) 안에서 나눠 받도록 이어받기를 지원한다. 이미 받은 코스는 건너뛴다.
  1) 코스 목록        areaBasedList2 contentTypeId=25 (약 2회)
  2) 코스 지점        detailInfo2 (코스당 1회) → 지점 contentid(subcontentid), 순서, 이름, 지점 설명
  3) 코스 설명        detailCommon2 (코스당 1회) → overview

출력 (data/raw/tourapi/courses/, 커밋 제외)
  courses.csv  stops.csv  overviews.csv  (stops·overviews는 이어 붙인다)

키는 collect_tourapi.py와 같은 방식으로 읽고 출력하지 않는다. 한도 초과 응답을 받으면 즉시 멈춘다.

사용 예:
  TOUR_API_KEY_FILE=../hidden-travel/config/application-secret.yaml python scripts/collect_tourapi_courses.py --max-calls 700
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_tourapi import load_key  # noqa: E402

BASE = "https://apis.data.go.kr/B551011/KorService2/"
OUT = Path("data/raw/tourapi/courses")
COURSE_FIELDS = ["contentid", "title", "lclsSystm1", "lclsSystm2", "lclsSystm3", "lDongRegnCd", "lDongSignguCd",
                 "addr1", "mapx", "mapy", "cpyrhtDivCd", "modifiedtime"]
STOP_FIELDS = ["course_id", "subnum", "subcontentid", "subname", "subdetailoverview"]


class QuotaExceeded(Exception):
    pass


class Client:
    def __init__(self, key: str, max_calls: int):
        self.key, self.max_calls, self.calls = key, max_calls, 0

    def get(self, op: str, **params) -> dict:
        if self.calls >= self.max_calls:
            raise QuotaExceeded(f"이번 실행 호출 상한({self.max_calls})에 도달")
        query = {"serviceKey": self.key, "MobileOS": "ETC", "MobileApp": "tripin-ai", "_type": "json", **params}
        url = BASE + op + "?" + urllib.parse.urlencode(query)
        for attempt in range(3):
            self.calls += 1
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    text = resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    raise QuotaExceeded("HTTP 429")
                time.sleep(2 * (attempt + 1))
                continue
            except urllib.error.URLError:
                time.sleep(2 * (attempt + 1))
                continue
            # 한도 초과는 JSON이 아니라 XML 오류 문서로 올 때가 있다.
            if "LIMITED_NUMBER_OF_SERVICE_REQUESTS" in text or "<returnReasonCode>22" in text:
                raise QuotaExceeded("일일 호출 한도 초과 응답")
            try:
                return json.loads(text)["response"]["body"]
            except (json.JSONDecodeError, KeyError):
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"{op} 요청 실패 (응답 형식 오류)")


def items_of(body: dict) -> list[dict]:
    raw = body.get("items") or {}
    items = raw.get("item", []) if isinstance(raw, dict) else []
    return [items] if isinstance(items, dict) else items


def done_ids(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig") as f:
        return {row[column] for row in csv.DictReader(f)}


def append(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-calls", type=int, default=700, help="이번 실행에서 쓸 최대 호출 수 (하루 한도 1,000 안에서)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = Client(load_key(), args.max_calls)

    courses_path = OUT / "courses.csv"
    try:
        if not courses_path.exists():
            courses, page = [], 1
            while True:
                body = client.get("areaBasedList2", contentTypeId=25, arrange="A", numOfRows=1000, pageNo=page)
                courses += items_of(body)
                if page * 1000 >= int(body.get("totalCount", 0)):
                    break
                page += 1
            append(courses_path, COURSE_FIELDS, courses)
            print(f"코스 목록 {len(courses)}개")
        with courses_path.open(encoding="utf-8-sig") as f:
            course_ids = [row["contentid"] for row in csv.DictReader(f)]

        stops_path, stops_done_path = OUT / "stops.csv", OUT / "stops_done.csv"
        done = done_ids(stops_done_path, "course_id")
        for course_id in [c for c in course_ids if c not in done]:
            stops = items_of(client.get("detailInfo2", contentId=course_id, contentTypeId=25, numOfRows=100, pageNo=1))
            append(stops_path, STOP_FIELDS, [{**s, "course_id": course_id} for s in stops])
            append(stops_done_path, ["course_id", "n_stops"], [{"course_id": course_id, "n_stops": len(stops)}])

        overviews_path = OUT / "overviews.csv"
        done = done_ids(overviews_path, "course_id")
        for course_id in [c for c in course_ids if c not in done]:
            items = items_of(client.get("detailCommon2", contentId=course_id))
            overview = items[0].get("overview", "") if items else ""
            append(overviews_path, ["course_id", "overview"], [{"course_id": course_id, "overview": overview}])
        print("수집 완료")
    except QuotaExceeded as e:
        print(f"중단: {e}. 다음 실행에서 이어서 받는다.")
    finally:
        n_courses = len(done_ids(courses_path, "contentid")) if courses_path.exists() else 0
        n_stops = len(done_ids(OUT / "stops_done.csv", "course_id"))
        n_over = len(done_ids(OUT / "overviews.csv", "course_id"))
        print(f"이번 호출 {client.calls}회 | 코스 {n_courses}개 중 지점 수집 {n_stops}개, 설명 수집 {n_over}개")


if __name__ == "__main__":
    main()
