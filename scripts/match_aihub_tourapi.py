"""AI Hub 여행로그 방문지 ↔ TourAPI 관광 콘텐츠 매칭률을 잰다.

좌표 반경 안의 TourAPI 후보 중 이름이 가장 비슷한 것을 고른다. 결과 CSV는 data/interim/ 아래(커밋 제외)에 둔다.

사용 예:
  python scripts/match_aihub_tourapi.py \
    --visits data/raw/.../TL_csv/tn_visit_area_info_E.csv \
    --tourapi data/raw/tourapi/region_11.csv data/raw/tourapi/region_28.csv data/raw/tourapi/region_41.csv
"""
import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

# AI Hub 방문지 유형 중 관광지 성격인 것(식당·숙소·집·역 등 제외). tc_codeb 'VIS' 참고.
TOURIST_VISIT_TYPES = {1, 2, 3, 4, 5, 6, 7, 8, 13}
# 축제·행사(15)는 장소가 아니라 기간 콘텐츠라 매칭 대상에서 뺀다.
EXCLUDED_CONTENT_TYPES = {15}
# 매장 지점·행사·부속 프로그램 이름. "한복남 경복궁점"이 경복궁으로 붙는 것을 막는다.
NOISE_TITLE = re.compile(r"(점|지점|야간개장|축제|페스티벌|별빛야행|체험|투어|공연|전시회)$")


def normalize(name: str) -> str:
    name = str(name).lower()
    name = re.sub(r"\(.*?\)|\[.*?\]", "", name)
    return re.sub(r"[\s·\-_.,&'\"]", "", name)


def similarity(visit: str, title: str, raw_title: str) -> float:
    if not visit or not title:
        return 0.0
    if visit == title:
        return 1.0
    if visit in title or title in visit:
        if NOISE_TITLE.search(raw_title.strip()) and visit not in raw_title:
            return 0.0
        short, long_ = sorted((len(visit), len(title)))
        return 0.7 + 0.3 * short / long_
    return SequenceMatcher(None, visit, title).ratio()


def haversine_m(lat, lng, lats, lngs):
    lat, lng, lats, lngs = map(np.radians, (lat, lng, lats, lngs))
    h = np.sin((lats - lat) / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin((lngs - lng) / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(h))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visits", required=True)
    parser.add_argument("--tourapi", nargs="+", required=True)
    parser.add_argument("--radius", type=float, default=1000, help="후보 탐색 반경(m)")
    parser.add_argument("--out", default="data/interim/aihub_tourapi_match.csv")
    args = parser.parse_args()

    tour = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in args.tourapi], ignore_index=True)
    tour = tour[~tour.contenttypeid.isin(EXCLUDED_CONTENT_TYPES)].dropna(subset=["mapx", "mapy"])
    tour = tour.drop_duplicates("contentid").reset_index(drop=True)
    tour["n"] = tour.title.map(normalize)

    visits = pd.read_csv(args.visits, encoding="utf-8-sig", low_memory=False)
    visits = visits[visits.VISIT_AREA_TYPE_CD.isin(TOURIST_VISIT_TYPES)].dropna(subset=["X_COORD", "Y_COORD"]).copy()
    visits["n"] = visits.VISIT_AREA_NM.map(normalize)
    counts = visits.n.value_counts()
    places = visits.drop_duplicates("n")

    rows = []
    for p in places.itertuples():
        dist = haversine_m(p.Y_COORD, p.X_COORD, tour.mapy.values, tour.mapx.values)
        best = (None, 0.0, None)
        for i in np.where(dist <= args.radius)[0]:
            sim = similarity(p.n, tour.n.iat[i], tour.title.iat[i])
            # 같은 유사도면 더 가까운 후보를 고른다.
            if sim > best[1] or (sim == best[1] and best[0] is not None and dist[i] < best[2]):
                best = (i, sim, dist[i])
        i, sim, d = best
        rows.append({
            "visit_name": p.VISIT_AREA_NM, "visit_type": p.VISIT_AREA_TYPE_CD, "visits": counts[p.n],
            "contentid": None if i is None else tour.contentid.iat[i],
            "tour_title": None if i is None else tour.title.iat[i],
            "contenttypeid": None if i is None else tour.contenttypeid.iat[i],
            "sim": round(sim, 3), "dist_m": None if d is None else round(d),
        })
    result = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"TourAPI 후보 {len(tour)}건, 방문 {len(visits)}건, 서로 다른 방문지 {len(result)}곳")
    for radius in (200, 500, 1000):
        for threshold in (0.8, 0.9):
            ok = (result.sim >= threshold) & (result.dist_m <= radius)
            print(f"반경 {radius}m, 유사도 {threshold} 이상: 장소 {ok.mean():.1%}, 방문 {result.visits[ok].sum() / result.visits.sum():.1%}")


if __name__ == "__main__":
    main()
