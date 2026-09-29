"""매칭 결과로 평가·학습용 데이터셋을 만든다 (#3).

입력
  data/raw/aihub/<권역>/<TL|VL>/  (extract_aihub.py)
  data/raw/tourapi/region_*.csv   (collect_tourapi.py)
  data/reference/tourapi_lcls_codes.csv
  data/interim/match_<권역>_<TL|VL>.csv (match_aihub_tourapi.py)

출력 (data/interim/eval/, 커밋 제외)
  attractions.csv  TourAPI 관광지. 분류 코드와 이름, 법정동 시도·시군구 코드
  travelers.csv    여행 1건 = 여행자 1명. 설문(여행 스타일·동기·선호 시군구), 동반 형태, 여행 미션
  visits.csv       TourAPI로 매칭된 관광지 방문과 만족도·재방문·추천 의향

선택 출력
  --review N  매칭 검수용 표본 CSV (data/interim/review/match_review.csv)
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

from match_aihub_tourapi import MATCH_MAX_DIST_M, MATCH_MIN_SIM, TOURIST_VISIT_TYPES, normalize

REGIONS = ["capital", "east", "west", "jeju"]
SPLITS = ["TL", "VL"]


def read_aihub(region: str, split: str, table: str) -> pd.DataFrame:
    # 여행 테이블은 권역 접미어(_E 등)가 붙고, 코드표(tc_*)는 붙지 않는다.
    path = (glob.glob(f"data/raw/aihub/{region}/{split}/{table}_?.csv") + [f"data/raw/aihub/{region}/{split}/{table}.csv"])[0]
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def sgg_names(region: str) -> dict[str, str]:
    """AI Hub tc_sgg에서 5자리 시군구 코드 → "시도 시군구" 이름."""
    sgg = read_aihub(region, "TL", "tc_sgg")
    rows = sgg[sgg.SGG_CD2.notna() & sgg.SGG_CD3.isna()]
    return {f"{int(r.SGG_CD1)}{int(r.SGG_CD2):03d}": f"{r.SIDO_NM} {r.SGG_NM}" for r in rows.itertuples()}


def build_attractions() -> pd.DataFrame:
    tour = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype=str) for p in glob.glob("data/raw/tourapi/region_*.csv")])
    tour = tour.drop_duplicates("contentid")
    codes = pd.read_csv("data/reference/tourapi_lcls_codes.csv", dtype=str)
    names = {**dict(zip(codes.lclsSystm1Cd, codes.lclsSystm1Nm)), **dict(zip(codes.lclsSystm2Cd, codes.lclsSystm2Nm)),
             **dict(zip(codes.lclsSystm3Cd, codes.lclsSystm3Nm))}
    for level in (1, 2, 3):
        tour[f"lcls{level}_name"] = tour[f"lclsSystm{level}"].map(names)
    cols = ["contentid", "title", "contenttypeid", "lclsSystm1", "lclsSystm2", "lclsSystm3",
            "lcls1_name", "lcls2_name", "lcls3_name", "lDongRegnCd", "lDongSignguCd", "addr1",
            "mapx", "mapy", "cpyrhtDivCd", "firstimage"]
    return tour[cols]


def build_travelers(region: str, split: str, names: dict[str, str]) -> pd.DataFrame:
    master = read_aihub(region, split, "tn_traveller_master")
    travel = read_aihub(region, split, "tn_travel")
    df = travel.merge(master, on="TRAVELER_ID")
    out = pd.DataFrame({
        "travel_id": df.TRAVEL_ID, "traveler_id": df.TRAVELER_ID, "region": region, "split": split,
        "gender": df.GENDER, "age_grp": df.AGE_GRP, "companion": df.TRAVEL_STATUS_ACCOMPANY,
        "travel_days": (pd.to_datetime(df.TRAVEL_END_YMD) - pd.to_datetime(df.TRAVEL_START_YMD)).dt.days + 1,
        "mission": df.TRAVEL_MISSION_CHECK,
    })
    for i in range(1, 9):
        out[f"style_{i}"] = df[f"TRAVEL_STYL_{i}"]
    for i in range(1, 4):
        out[f"motive_{i}"] = df[f"TRAVEL_MOTIVE_{i}"].astype("Int64")
        code = df[f"TRAVEL_LIKE_SGG_{i}"].astype("Int64").astype(str)
        out[f"like_sgg_{i}"] = code.where(code != "<NA>")
        out[f"like_sgg_{i}_name"] = out[f"like_sgg_{i}"].map(names)
    return out


def build_visits(region: str, split: str, match: pd.DataFrame) -> pd.DataFrame:
    visits = read_aihub(region, split, "tn_visit_area_info")
    visits = visits[visits.VISIT_AREA_TYPE_CD.isin(TOURIST_VISIT_TYPES)].dropna(subset=["X_COORD", "Y_COORD"]).copy()
    visits["n"] = visits.VISIT_AREA_NM.map(normalize)
    ok = match[(match.sim >= MATCH_MIN_SIM) & (match.dist_m <= MATCH_MAX_DIST_M)]
    visits = visits.merge(ok[["n", "contentid", "sim", "dist_m"]], on="n")
    return pd.DataFrame({
        "travel_id": visits.TRAVEL_ID, "visit_area_id": visits.VISIT_AREA_ID, "visit_order": visits.VISIT_ORDER,
        "visit_name": visits.VISIT_AREA_NM, "visit_type": visits.VISIT_AREA_TYPE_CD,
        "contentid": visits.contentid.astype("Int64").astype(str),
        "satisfaction": visits.DGSTFN, "revisit_intention": visits.REVISIT_INTENTION,
        "recommend_intention": visits.RCMDTN_INTENTION, "choice_reason": visits.VISIT_CHC_REASON_CD,
        "match_sim": visits.sim, "match_dist_m": visits.dist_m,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/interim/eval")
    parser.add_argument("--review", type=int, default=0, help="매칭 검수용 표본 수 (0이면 만들지 않음)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    attractions = build_attractions()
    travelers, visits, matches = [], [], []
    for region in REGIONS:
        names = sgg_names(region)
        for split in SPLITS:
            match = pd.read_csv(f"data/interim/match_{region}_{split}.csv", encoding="utf-8-sig")
            travelers.append(build_travelers(region, split, names))
            visits.append(build_visits(region, split, match))
            matches.append(match.assign(region=region, split=split))
    travelers, visits = pd.concat(travelers), pd.concat(visits)
    visits = visits.merge(attractions[["contentid", "contenttypeid", "lDongRegnCd", "lDongSignguCd"]], on="contentid")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    attractions.to_csv(out / "attractions.csv", index=False, encoding="utf-8-sig")
    travelers.to_csv(out / "travelers.csv", index=False, encoding="utf-8-sig")
    visits.to_csv(out / "visits.csv", index=False, encoding="utf-8-sig")

    positives = visits[(visits.satisfaction >= 4) & (visits.contenttypeid != "38")]
    per_travel = positives.groupby("travel_id").contentid.nunique()
    print(f"관광지 {len(attractions)}건, 여행 {len(travelers)}건, 매칭 방문 {len(visits)}건")
    print(travelers.assign(has=travelers.travel_id.isin(per_travel.index))
          .groupby("split").has.agg(["size", "sum"]).rename(columns={"size": "여행", "sum": "정답 1곳 이상(쇼핑 제외)"}))

    if args.review:
        match = pd.concat(matches).drop_duplicates(["region", "n"])
        ok = match[(match.sim >= MATCH_MIN_SIM) & (match.dist_m <= MATCH_MAX_DIST_M)]
        # 권역마다 같은 수를 뽑는다.
        sample = ok.groupby("region").sample(n=args.review // len(REGIONS), random_state=args.seed)
        review = sample[["region", "visit_name", "visit_addr", "tour_title", "tour_addr", "dist_m", "sim", "visits"]].copy()
        review["verdict"] = ""   # O: 같은 장소, P: 큰 시설 안 매장·부속 시설, X: 다른 장소
        review["note"] = ""
        review_dir = Path("data/interim/review")
        review_dir.mkdir(parents=True, exist_ok=True)
        review.sample(frac=1, random_state=args.seed).to_csv(review_dir / "match_review.csv", index=False, encoding="utf-8-sig")
        print(f"검수용 표본 {len(review)}건: {review_dir / 'match_review.csv'}")


if __name__ == "__main__":
    main()
