# TriPin 임베딩 서버 (#1). CPU 전용.
#   docker build -t tripin-ai .
#   docker run -p 8000:8000 tripin-ai
# 파인튜닝한 모델을 쓰려면 폴더를 마운트하고 MODEL_NAME·MODEL_VERSION을 바꾼다:
#   docker run -p 8000:8000 -v $(pwd)/models/final:/models/final -e MODEL_NAME=/models/final -e MODEL_VERSION=<버전> tripin-ai
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/hf
WORKDIR /app

# GPU용 torch(수 GB) 대신 CPU용을 받는다.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
RUN pip install --no-cache-dir "sentence-transformers>=3" "fastapi>=0.110" "uvicorn[standard]>=0.29" numpy pandas

# 기본 모델을 이미지에 미리 받아 첫 요청 지연을 없앤다.
ARG MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

COPY tripin_ai ./tripin_ai
COPY app ./app
COPY data/reference ./data/reference

ENV MODEL_NAME=${MODEL_NAME} MODEL_VERSION=mminilm-l12-v1 TEMPLATE_VERSION=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
