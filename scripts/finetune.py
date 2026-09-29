"""취향 임베딩 파인튜닝 (#6). 여행자 단위 k겹마다 평가 묶음을 빼고 학습한 뒤, 그 묶음을 평가할 벡터를 만든다.

사용 예:
  .venv/Scripts/python scripts/finetune.py --folds 0 --epochs 1          # 파일럿: 1겹
  .venv/Scripts/python scripts/finetune.py --folds 0 1 2 3 4 --epochs 1  # 전체
  .venv/Scripts/python scripts/finetune.py --folds 0 --max-steps 20      # 속도 측정
  python scripts/finetune.py --folds 0 1 2 3 4 --device cuda             # Colab GPU (docs/colab.md)
  python scripts/finetune.py --folds 0 1 2 3 4 --course-pairs real       # 코스 쌍 보조 학습 (통제군: shuffled)
  python scripts/finetune.py --final --device cuda                        # 서비스용 최종 모델 (전체 여행자로 학습)

출력 (커밋 제외)
  data/interim/models/<버전>/fold<k>/                    학습한 모델
  data/interim/emb/<버전>/fold<k>_attractions.npy         후보 관광지 벡터 (평가 데이터 관광지 순서, 안 쓴 곳은 0)
  data/interim/emb/<버전>/fold<k>_travelers.npy / .csv    평가 묶음 여행자 벡터
  data/interim/emb/<버전>/fold<k>_train.json              학습 설정·데이터 수·시간
  data/interim/models/<버전>/final/                       --final일 때: 서비스용 모델 (+ final_train.json)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tripin_ai.encoder import DEFAULT_MODEL_NAME  # noqa: E402
from tripin_ai.evaluation import data as eval_data  # noqa: E402
from tripin_ai.training.course_pairs import build_course_triplets  # noqa: E402
from tripin_ai.training.triplets import build_triplets  # noqa: E402


def class_names() -> dict[str, str]:
    codes = pd.read_csv("data/reference/tourapi_lcls_codes.csv", dtype=str)
    return dict(zip(codes.lclsSystm3Cd, codes.lclsSystm3Nm))


def train_fold(args, data, fold: int | None, attraction_texts: list[str], traveler_texts: dict[str, str]) -> None:
    """fold가 None이면 서비스용 최종 모델: 전체 여행자로 학습하고 평가용 임베딩은 만들지 않는다."""
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    label = "final" if fold is None else f"fold{fold}"
    train_ids = {t for t, f in data.fold_of.items() if f != fold}
    triplets = build_triplets(data, train_ids, traveler_texts, attraction_texts, seed=args.seed + (fold or 0))
    negative_sources = triplets.attrs["negative_sources"]
    if args.max_examples:
        triplets = triplets.head(args.max_examples)
    n_course = 0
    if args.course_pairs != "none":
        course = build_course_triplets(data, attraction_texts, mode=args.course_pairs, course_dir=Path(args.course_dir),
                                       class_names=class_names(), seed=args.seed + (fold or 0))
        n_course = len(course)
        triplets = pd.concat([triplets, course]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"[{label}] 학습 예시 {len(triplets)}건 (코스 {args.course_pairs} {n_course}건), 음성 출처 {negative_sources}")

    model = SentenceTransformer(args.base_model, device=args.device)
    # 설명을 붙인 관광지 문장(템플릿 v1)은 평균 약 190토큰이라 기본 128에서 대부분 잘린다. 저장한 모델에 함께 기록된다.
    model.max_seq_length = args.max_seq_length
    model_dir = Path("data/interim/models") / args.version / label
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(model_dir / "checkpoints"), num_train_epochs=args.epochs, max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr, warmup_steps=0.1,  # 비율(10%)
        # 같은 여행자 문장이 한 배치에 두 번 들어가면 그 여행자의 다른 만족한 곳이 음성이 돼 버린다.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50, save_strategy="no", report_to="none", seed=args.seed, use_cpu=args.device == "cpu",
        fp16=args.device == "cuda",
        dataloader_drop_last=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args,
                                         train_dataset=Dataset.from_pandas(triplets[["anchor", "positive", "negative"]]),
                                         loss=losses.MultipleNegativesRankingLoss(model))
    start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start
    model.save(str(model_dir))
    info = {"fold": fold, "base_model": args.base_model, "examples": len(triplets), "course_pairs": args.course_pairs,
            "course_examples": n_course, "negative_sources": negative_sources, "epochs": args.epochs,
            "max_steps": args.max_steps, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
            "train_seconds": round(train_seconds), "attraction_template": args.attraction_template,
            "traveler_template": args.traveler_template, "max_seq_length": args.max_seq_length}
    if fold is None:
        (model_dir / "final_train.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[final] 학습 {train_seconds:.0f}초 → {model_dir}")
        return

    # 평가에 필요한 것만 임베딩한다: 후보 관광지 전체, 이 묶음 여행자.
    needed = np.unique(np.concatenate(list(data.candidates.values())))
    start = time.perf_counter()
    vectors = model.encode([attraction_texts[i] for i in needed], batch_size=64, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype("<f4")
    a_full = np.zeros((len(data.attractions), vectors.shape[1]), dtype="<f4")
    a_full[needed] = vectors
    eval_ids = sorted(t for t, f in data.fold_of.items() if f == fold and t in traveler_texts)
    t_vec = model.encode([traveler_texts[t] for t in eval_ids], batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype("<f4")
    embed_seconds = time.perf_counter() - start

    out = Path("data/interim/emb") / args.version
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"fold{fold}_attractions.npy", a_full)
    np.save(out / f"fold{fold}_travelers.npy", t_vec)
    pd.DataFrame({"travel_id": eval_ids}).to_csv(out / f"fold{fold}_travelers.csv", index=False)
    info["embed_seconds"] = round(embed_seconds)
    (out / f"fold{fold}_train.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[fold {fold}] 학습 {train_seconds:.0f}초, 임베딩 {embed_seconds:.0f}초 → {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="mminilm-l12-ft-v1")
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-version", default="mminilm-l12-v1", help="문장을 가져올 기성 임베딩 폴더")
    parser.add_argument("--attraction-template", default="v0")
    parser.add_argument("--traveler-template", default="aihub-v1")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Colab GPU에서는 cuda")
    parser.add_argument("--course-pairs", default="none", choices=["none", "real", "shuffled"],
                        help="TourAPI 추천코스 쌍을 보조로 섞는다. shuffled는 지점을 같은 시군구 무작위로 바꾼 통제군")
    parser.add_argument("--course-dir", default="data/raw/tourapi/courses")
    parser.add_argument("--final", action="store_true", help="서비스용: 전체 여행자로 학습 (평가용 겹 무시)")
    args = parser.parse_args()

    base = Path("data/interim/emb") / args.base_version
    attractions = pd.read_csv(base / f"attractions_{args.attraction_template}.csv", dtype=str)
    travelers = pd.read_csv(base / f"travelers_{args.traveler_template}.csv", dtype=str)
    # 평가와 같은 분할(시드 42)을 써야 평가 묶음 여행자가 학습에 섞이지 않는다.
    data = eval_data.load(attractions.contentid.tolist(), seed=42)
    traveler_texts = dict(zip(travelers.travel_id, travelers.text))
    if args.final:
        train_fold(args, data, None, attractions.text.tolist(), traveler_texts)
        return
    for fold in args.folds:
        train_fold(args, data, fold, attractions.text.tolist(), traveler_texts)


if __name__ == "__main__":
    main()
