# 여행기 피드백을 다음 추천에 반영하기 (2026-10-30)

`scripts/diary_feedback_eval.py`, 결과 `data/interim/eval_results/diary-feedback-v1/`(비공개). CPU.
서비스 구현: backend `DiaryFeedback`·`TasteRanker.applyDiaryFeedback`(코스 생성), 설계 backend `docs/design/recommendation.md` §5.

## 1. 백엔드가 실제로 가진 신호

여행기를 **발행**하고 `includeInTasteProfile`(기본 켬)이 켜져 있으면 `diary_taste_signals`에 한 행이 생긴다.

| 필드 | 단위 | 비고 |
|---|---|---|
| `satisfaction` | **여행 단위** 1~5, 선택 | 장소별 만족도는 없다 |
| `experience_tags` | 여행 단위, 사전 12개 중 최대 5개 | AI Hub에 대응 정보가 없어 검증 불가 |
| `region_sig_cd` | 여행 지역 | |
| (조인) 여행의 최종 코스 `course_items` | 장소 목록 | 편집이 반영된 최종 코스 = 다녀온 장소의 대리값 |

제품 원칙(product.md): 본문·사진·EXIF·열람 수는 쓰지 않는다. 낮은 가중치의 취향 신호로만 쓴다. 사용자가 끌 수 있다.

## 2. 설계 원칙과 근거 논문

| 원칙 | 근거 |
|---|---|
| **여행 단위 만족도를 장소마다 복사하지 않는다.** 묶음(bag) 단위 라벨은 개별 항목의 라벨이 아니다 | Dietterich, Lathrop & Lozano-Pérez (1997), *Solving the multiple instance problem with axis-parallel rectangles*, Artificial Intelligence 89 · Kück & de Freitas (2005), *Learning about individuals from group statistics*, UAI · Quadrianto et al. (2009), *Estimating labels from label proportions*, JMLR 10 |
| 만족도는 **여행 전체의 신뢰도 가중치**로만 쓴다. 다녀왔다는 사실은 약한 긍정, 만족도는 그 확신의 크기 | Hu, Koren & Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets*, ICDM — 선호(preference)와 신뢰도(confidence)를 분리 |
| 지난 여행 장소의 **의미 벡터 중심**으로 다음 후보를 끌어올린다 | Rocchio (1971), *Relevance feedback in information retrieval*, SMART System · Salton & Buckley (1990), *Improving retrieval performance by relevance feedback*, JASIS 41(4) |
| 기록이 적을 때 개인 값이 튀지 않도록 **전체 분포로 수축**한다(경험적 베이즈) | Efron & Morris (1975), *Data analysis using Stein's estimator and its generalizations*, JASA 70 · Agarwal & Chen (2009), *Regression-based latent factor models*, KDD — 특성 기반 사전분포 + 개인 편차 |
| 불만족한 여행이라고 그 장소들을 ‘싫어함’으로 밀어내지 않는다(**비대칭**). 평가는 무작위로 결측되지 않는다 | Marlin & Zemel (2009), *Collaborative prediction and ranking with non-random missing data*, RecSys · Joachims et al. (2005), *Accurately interpreting clickthrough data as implicit feedback*, SIGIR |
| 오프라인에서 확신할 수 없으면 **무작위 배정과 노출 기록**으로 서비스에서 검증한다 | Li, Chu, Langford & Wang (2011), *Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms*, WSDM · Schnabel et al. (2016), *Recommendations as treatments*, ICML · Joachims, Swaminathan & Schnabel (2017), *Unbiased learning-to-rank with biased feedback*, WSDM |

## 3. 반영 식 (서비스와 실험이 같다)

```text
점수 = z(순위 모델) + λ · w̄ · ( z(Rocchio) + z(분류 lift) ) / 2        (같은 지역 후보 안에서 표준화)
  Rocchio   = 후보 E5 벡터 · 지난 여행 장소 벡터 중심(여행별 중심을 w로 가중 합 → 정규화)
  분류 lift = log(p_u / p_0),  p_u = (지난 여행 장소의 그 중분류 수 + k·p_0) / (지난 여행 장소 수 + k),  k = 5
              p_0 = 추천 가능 관광지 전체(카탈로그)의 중분류(lclsSystm2) 비율
  w(s)      = clip(0.5 + 0.25·(s − 3), 0, 1),  만족도 없음 → s = 3      (s=1 → 0, 3 → 0.5, 5 → 1)
```

## 4. 오프라인 실험

**대리 설계.** AI Hub는 응답자당 여행이 하나다. 그래서 두 개 이상의 시군구를 방문한 여행자에서 **가장 먼저 방문한 시군구 = 지난 여행(여행기)**, 나머지 시군구 = 다음 여행으로 둔다. 여행기 신호는 서비스와 같은 정보만 쓴다(그 지역 방문지 + 방문지 만족도 평균을 반올림한 여행 단위 만족도). 순위 모델은 TL로만 학습, λ는 TL 안쪽 검증 여행자(935명)에서 고르고, 평가는 TL에 없는 VL 여행자 **556명(다음 여행 질의 678개)**. 여행자 단위 쌍 부트스트랩 95% 구간.

**λ 선택 (안쪽 검증, NDCG@10, base 0.4970)**

| λ | both | category | rocchio | oracle(장소별 만족도) |
|---:|---:|---:|---:|---:|
| −0.3 (진단) | 0.4863 | 0.4832 | 0.4781 | 0.4709 |
| −0.1 (진단) | 0.4958 | 0.4964 | 0.4951 | 0.4939 |
| 0.1 | 0.4968 | 0.4950 | 0.4977 | 0.4983 |
| 0.3 | 0.4909 | 0.4778 | 0.4876 | 0.4850 |
| 0.8 | 0.4402 | 0.3762 | 0.4454 | 0.4294 |

**검증 결과 (VL 556명, λ = 0.1, 비대칭 가중)**

| 방법 | NDCG@10 | 차이 [95% CI] | Hit@10 차이 [95% CI] |
|---|---:|---:|---:|
| base (현재 서비스) | 0.5075 | — | — |
| both (서비스 적용안) | 0.5081 | +0.0006 [−0.0043, +0.0059] | +0.67%p [−0.48, +1.83] |
| category | 0.5076 | +0.0001 [−0.0047, +0.0054] | +0.51%p [−0.71, +1.73] |
| rocchio | 0.5057 | −0.0018 [−0.0075, +0.0039] | +0.34%p [−0.77, +1.49] |
| 남의 여행기 (대조) | 0.5064 | −0.0010 [−0.0060, +0.0039] | −0.32%p [−1.32, +0.67] |
| 장소별 만족도 (서비스에 없음, 참고) | 0.5067 | −0.0008 [−0.0068, +0.0050] | +0.27%p [−0.82, +1.43] |

지난 여행 만족도 ≤ 3인 여행자(40명): 대칭 가중(불만족 여행 장소를 밀어냄) both −0.0060 [−0.0165, 0.0000] vs **비대칭 −0.0003 [−0.0024, +0.0014]**.
지난 여행 만족도 분포: 1점 5 · 2점 8 · 3점 28 · 4점 270 · 5점 247.

## 5. 해석

1. **이 대리 실험에서는 여행기 신호가 다음 추천을 개선하지 못했다.** 모든 방법의 차이가 신뢰구간 안에서 0을 포함하고, λ를 키우면 양·음 방향 모두 나빠진다. 장소별 만족도(서비스에 없는 더 풍부한 정보)를 써도 같다. 순위 모델이 이미 설문과 장소 특성으로 대부분을 설명하고 있어, 같은 사람의 다른 지역 방문지가 추가 정보를 거의 주지 않는다.
2. **비대칭 원칙은 근거가 있다.** 불만족 여행을 대칭으로 반영하면 그 여행자들의 순위가 더 떨어졌다(표본 40명이라 경계적).
3. **이 실험이 답하지 못하는 것.** 한 여행 안의 다른 지역은 서비스의 ‘몇 달 뒤 다음 여행’과 다르다(같은 동행·같은 날씨·여행 중 다양성 추구). 실제 사용자가 직접 남긴 만족도·경험 태그도 AI Hub에는 없다. 그래서 오프라인 결과만으로는 “반영하면 좋아진다”도 “반영하면 안 된다”도 말할 수 없다.

## 6. 서비스 반영 결정

- **반영 경로를 서비스에 붙이되, 회원을 무작위로 절반 나눠 적용한다**(A/B). λ = 0.1(오프라인에서 해가 측정되지 않은 가장 작은 값), 비대칭 가중, 최근 여행기 10개.
- 배정은 회원 ID 해시로 고정(같은 회원은 항상 같은 쪽), 실험 버전마다 해시가 달라 다음 실험과 겹치지 않는다.
- 노출 기록 `ranker_version`에 `lgbm-ranker-v1+diary-feedback-v1(n=…)`(적용) / `…-control(n=…)`(대조)로 남긴다. 코스 생성은 이미 단계별 뽑힐 확률을 기록하므로 다른 λ도 역성향 가중(IPS)으로 오프라인 재평가할 수 있다(Li et al. 2011).
- 설정: `ranker.diary-feedback.weight`(0이면 끔), `ranker.diary-feedback.treatment-share`.
- 지역 뽑기 점수(지역 간 비교)에는 아직 적용하지 않는다. 실험은 지역 안 장소 순위만 검증했다.
- 경험 태그는 AI Hub로 검증할 수 없어 점수에 쓰지 않는다. 저장은 계속되므로 A/B 데이터로 따로 평가한다.

**판정 기준(사전 등록).** 대조군 대비 (a) 생성한 코스에서 사용자가 지우지 않고 남긴 추천 장소 비율, (b) 다음 여행기 만족도, (c) 코스 재생성률을 회원 단위로 비교한다. 남긴 비율이 약 70%일 때 +5%p 차이를 α = 0.05, 검정력 0.8로 보려면 군당 약 1,250명이 필요하다(두 비율 차이 검정). 그 전에는 효과를 주장하지 않고, (a)가 유의하게 좋아지면 λ를 올리고 나빠지면 끈다.

## 7. 한계

- 대리 실험(한 여행 안의 지역)이며, 지난 여행 만족도 3점 이하가 41명뿐이다.
- 최종 코스는 함께 간 친구의 편집이 섞인 장소 목록이고, 실제 방문 여부는 모른다.
- 노출 기록만으로 평가가 공정해지지는 않는다. 코스 생성의 확률 기록과 무작위 배정이 있어야 한다.
