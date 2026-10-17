# TriPin 임베딩 서버 (#1). CPU 전용.
#   docker build -t tripin-ai .
#   docker run -p 8000:8000 -v <E5-final-folder>:/models/final:ro tripin-ai
# 파인튜닝한 모델을 쓰려면 폴더를 마운트하고 MODEL_NAME·MODEL_VERSION을 바꾼다:
#   docker run -p 8000:8000 -v $(pwd)/models/final:/models/final -e MODEL_NAME=/models/final -e MODEL_VERSION=<버전> tripin-ai
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/hf
WORKDIR /app

# GPU용 torch(수 GB) 대신 CPU용을 받는다.
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
RUN pip install --no-cache-dir "sentence-transformers==5.7.0" "transformers==4.57.3" "fastapi>=0.110" "uvicorn[standard]>=0.29" numpy pandas

# 개발 RDS와 같은 학습 E5를 런타임에 /models/final로 마운트한다.
# 빌드 시 MiniLM을 내려받지 않는다. 가중치는 Git/이미지에 포함하지 않는다.
COPY tripin_ai ./tripin_ai
COPY app ./app
COPY data/reference ./data/reference

ENV MODEL_NAME=/models/final MODEL_VERSION=e5-base-rating-2022-2023-t4-all-v2 MODEL_INPUT_FORMAT=e5 TEMPLATE_VERSIONS=3 MAX_BATCH_ITEMS=8
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
