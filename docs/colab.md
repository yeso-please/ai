# Colab GPU로 파인튜닝하기 (#6)

CPU에서는 1겹 1에폭에 약 1시간(5겹 약 5시간)이 걸린다. Colab 무료 GPU(T4)를 쓰면 수십 분 안에 끝난다.
학습만 Colab에서 하고, **평가는 로컬에서** 한다(기준선 임베딩과 평가 코드가 로컬에 있다).

> ⚠️ 묶음에는 **AI Hub 데이터**가 들어 있다. 본인 Colab 세션(또는 본인 Google Drive)에만 올리고, 노트북·폴더를 공유하거나 공개 레포에 올리지 않는다(AI Hub 이용정책: 재배포 금지). 세션이 끝나면 Colab의 파일은 사라진다.

## 1. 로컬: 묶음 만들기

```bash
python scripts/make_colab_bundle.py
```

→ `data/interim/colab/tripin_colab.zip` (코드 + 평가 데이터, 약 5MB)

## 2. Colab: GPU 켜기

1. https://colab.research.google.com → **새 노트북**
2. 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** → 저장

## 3. Colab: 셀 실행

**셀 1. 묶음 업로드** (파일 선택 창에서 `tripin_colab.zip`)
```python
from google.colab import files
files.upload()
!unzip -q -o tripin_colab.zip -d tripin && ls tripin
```

**셀 2. 패키지와 GPU 확인** (Colab에는 PyTorch가 이미 있다)
```python
%cd /content/tripin
!pip install -q "sentence-transformers>=3" datasets accelerate
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

**셀 3. 학습** (5겹 전체. 겹마다 학습 → 평가용 임베딩까지)
```python
!python scripts/finetune.py --folds 0 1 2 3 4 --epochs 1 --device cuda --batch-size 64 --version mminilm-l12-ft-b64
```
- `--version`은 결과 폴더 이름이다. 설정이 다르면 이름도 바꿔서, 로컬 파일럿(`mminilm-l12-ft-v1`, 배치 32)을 덮어쓰지 않게 한다.
- 로컬 파일럿과 같은 조건으로 비교하려면 `--batch-size 32`. GPU에서는 64가 더 빠르고, 배치 안 음성이 많아져 보통 더 잘 배운다.
- 끊겨도 겹마다 결과가 저장된다. 남은 겹만 `--folds 3 4`처럼 다시 돌리면 된다.

**셀 4. 결과 내려받기** (평가에 필요한 벡터만. 모델 파일은 크고 평가에 필요 없다)
```python
!cd /content/tripin && zip -q -r emb_ft.zip data/interim/emb/mminilm-l12-ft-b64
files.download("/content/tripin/emb_ft.zip")
```

## 4. 로컬: 결과 넣고 평가

`emb_ft.zip`을 `ai/` 폴더에 풀어 `data/interim/emb/mminilm-l12-ft-b64/fold*_*.npy`가 생기게 한 뒤:

```bash
.venv/Scripts/python scripts/evaluate.py --finetuned mminilm-l12-ft-b64
```

## 모델 파일이 필요할 때

서비스에 쓸 최종 모델은 **전체 여행자로 한 번 더 학습**해서 만든다(겹별 모델은 평가용). 그때는 셀 4에서 `data/interim/models/<버전>`도 함께 zip으로 받는다(겹당 약 470MB).

## 다른 실험 (셀 3만 바꾼다)

| 실험 | 셀 3 명령 | 결과 폴더 |
|---|---|---|
| 관광지 설명 포함 (템플릿 v1) | `!python scripts/finetune.py --folds 0 1 2 3 4 --device cuda --batch-size 64 --attraction-template v1 --max-seq-length 256 --version mminilm-l12-ft-b64-v1` | `mminilm-l12-ft-b64-v1` |
| **서비스 템플릿 2 (설명 + 콘텐츠 유형, 템플릿 v2)** | `!python scripts/finetune.py --folds 0 1 2 3 4 --device cuda --batch-size 64 --attraction-template v2 --max-seq-length 256 --version mminilm-l12-ft-b64-v2` | `mminilm-l12-ft-b64-v2` |
| **취향 대조 음성 (개인화 몫 키우기)** | `!python scripts/finetune.py --folds 0 1 2 3 4 --device cuda --batch-size 64 --attraction-template v2 --max-seq-length 256 --negatives taste --version mminilm-l12-ft-b64-v2-taste` | `mminilm-l12-ft-b64-v2-taste` |
| 코스 쌍 보조 학습 | `!python scripts/finetune.py --folds 0 1 2 3 4 --device cuda --batch-size 64 --course-pairs real --version mminilm-l12-ft-b64-course` | `mminilm-l12-ft-b64-course` |
| 코스 쌍 통제군 (지점을 무작위로 섞음) | `!python scripts/finetune.py --folds 0 1 2 3 4 --device cuda --batch-size 64 --course-pairs shuffled --version mminilm-l12-ft-b64-shuf` | `mminilm-l12-ft-b64-shuf` |
| 서비스용 최종 모델 | `!python scripts/finetune.py --final --device cuda --batch-size 64 --version <채택한 설정 이름>` | `data/interim/models/<이름>/final` |

- 셀 4의 zip 경로를 결과 폴더에 맞게 바꾼다. 여러 실험을 한 세션에서 돌렸다면 `data/interim/emb`를 통째로 묶어도 된다.
- 코스 쌍 실험은 코스 수집(`scripts/collect_tourapi_courses.py`)이 끝난 뒤 묶음을 다시 만들어야 전체 코스가 들어간다. 지금 묶음에는 수집된 만큼만 들어 있다.
- 로컬 평가: `.venv/Scripts/python scripts/evaluate.py --finetuned mminilm-l12-ft-b64 mminilm-l12-ft-b64-course mminilm-l12-ft-b64-shuf`
- 설명 실험의 공정 비교: `... evaluate.py --finetuned mminilm-l12-ft-b64 mminilm-l12-ft-b64-v1 --restrict-to data/interim/described_contentids.csv`
- 취향 대조 음성(`--negatives taste`)은 음성을 "설문이 가장 다른 여행자가 같은 시군구에서 만족한 곳"으로 바꿔, 장소 매력이 아니라 취향 차이를 배우게 한다. 개인화 몫은 설문 섞기 대조군으로 잰다: `... evaluate.py --attraction-template v2 --finetuned mminilm-l12-ft-b64-v2 mminilm-l12-ft-b64-v2-taste --shuffled-control --baseline 무작위` (설명 있는 곳만은 `--restrict-to data/interim/described_contentids.csv`)
- 템플릿 v2 실험은 같은 템플릿 기성 모델과 비교한다: `... evaluate.py --attraction-template v2 --finetuned mminilm-l12-ft-b64-v2` (설명 있는 곳만: `--restrict-to data/interim/described_contentids.csv` 추가)
- 최종 모델은 서비스 배포용이라 모델 파일(약 470MB)을 받아야 한다: `!cd /content/tripin && zip -q -r model_final.zip data/interim/models/<이름>/final`
