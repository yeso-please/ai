"""평가 데이터(#3)의 관광지·여행자를 오프라인으로 임베딩한다 (#4).

서비스와 같은 tripin_ai.templates / tripin_ai.encoder를 쓴다.

출력 (data/interim/emb/<모델 버전>/, 커밋 제외)
  attractions_<템플릿>.npy / .csv   관광지 벡터와 contentid·문장
  travelers_<템플릿>.npy / .csv     여행자 벡터와 travel_id·문장

사용 예:
  .venv/Scripts/python scripts/embed_offline.py --attraction-template v0 --traveler-template aihub-v1
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tripin_ai.encoder import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION, Encoder  # noqa: E402
from tripin_ai.templates import Attraction, TravelerSurvey, attraction_text, region_from_address, traveler_text  # noqa: E402

EVAL_DIR = Path("data/interim/eval")
# 축제·행사(15)는 장소가 아니라 후보에서 뺀다. 쇼핑(38)은 평가 옵션이라 임베딩은 해 둔다.
EXCLUDED_CONTENT_TYPES = {"15"}


def attraction_texts(version: str) -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "attractions.csv", dtype=str)
    df = df[~df.contenttypeid.isin(EXCLUDED_CONTENT_TYPES)].copy()
    df["text"] = [
        attraction_text(Attraction(name=r.title, class_names=[r.lcls1_name, r.lcls2_name, r.lcls3_name],
                                   region_name=region_from_address(r.addr1),
                                   description=getattr(r, "overview", "") or ""), version)
        for r in df.itertuples()
    ]
    return df[["contentid", "text"]]


def traveler_texts(version: str) -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "travelers.csv", dtype=str)
    texts = []
    for r in df.itertuples():
        styles = {i: int(getattr(r, f"style_{i}")) for i in range(1, 9) if pd.notna(getattr(r, f"style_{i}"))}
        motives = [int(getattr(r, f"motive_{i}")) for i in range(1, 4) if pd.notna(getattr(r, f"motive_{i}"))]
        regions = [getattr(r, f"like_sgg_{i}_name") for i in range(1, 4) if pd.notna(getattr(r, f"like_sgg_{i}_name"))]
        texts.append(traveler_text(TravelerSurvey(styles=styles, motives=motives, liked_regions=regions), version))
    df["text"] = texts
    return df[["travel_id", "text"]]


def save(encoder: Encoder, frame: pd.DataFrame, path: Path) -> None:
    start = time.perf_counter()
    vectors = encoder.encode(frame.text.tolist(), show_progress=True)
    np.save(path.with_suffix(".npy"), vectors)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"{path.name}: {len(frame)}건, {vectors.shape[1]}차원, {time.perf_counter() - start:.0f}초")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--attraction-template", default="v0")
    parser.add_argument("--traveler-template", default="aihub-v1")
    args = parser.parse_args()

    out = Path("data/interim/emb") / args.model_version
    out.mkdir(parents=True, exist_ok=True)
    encoder = Encoder(args.model_name, args.model_version)
    save(encoder, attraction_texts(args.attraction_template), out / f"attractions_{args.attraction_template}")
    save(encoder, traveler_texts(args.traveler_template), out / f"travelers_{args.traveler_template}")


if __name__ == "__main__":
    main()
