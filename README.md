# TriPin AI

회원 취향과 관광지를 같은 벡터 공간에 올리는 **임베딩 서비스**와, 우리 데이터로 하는 **취향 모델 학습**을 둔다.

## 백엔드와의 계약

백엔드(`yeso-please/backend`)는 이 서비스를 이렇게 부른다. 백엔드 쪽은 이미 구현돼 있고, 이 서비스만 없다.

```
POST /embeddings
```

| 항목 | 내용 |
|---|---|
| 호출 주체 | 백엔드 `HttpEmbeddingClient` (설정 `EMBEDDING_SERVICE_BASE_URL`) |
| 응답 | `{"embeddingBase64": "...", "dimension": 384}` |
| 벡터 형식 | **float32, 리틀엔디언** 바이트를 base64로. PyTorch/NumPy의 `tensor.numpy().astype('float32').tobytes()` 그대로 |
| 차원 | 백엔드 설정 `embedding.expected-dimension`(현재 384)과 같아야 한다. 모델을 바꾸면 백엔드 설정도 함께 바꾼다 |
| 버전 | 백엔드가 `model_version`·`template_version`을 함께 저장한다. 모델·입력 텍스트 규칙이 바뀌면 버전을 올린다 |

요청 본문 모양은 백엔드 `EmbeddingRequest`를 따른다.

## 쓰는 곳

- 회원 취향 벡터: 온보딩 제출 시 (`user_taste_vectors`)
- 관광지 벡터: 사전 배치 (`attraction_embeddings`, backend #54)
- 코스 생성의 취향 점수 (backend #49), 지역 추첨의 "내 취향" 조건 (backend #41)

## 계획

1. 사전학습 한국어 문장 임베딩 모델(PyTorch, sentence-transformers)로 `/embeddings`를 띄운다. 학습 없이 시작한다
2. 우리 데이터로 파인튜닝한다: 설문의 좋았던 여행지, 코스에서 남긴 곳·교체한 곳, 관광지 설명
3. 코스 점수(취향·다양성·거리) 가중치를 사용자 반응으로 조정한다
