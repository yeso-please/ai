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
