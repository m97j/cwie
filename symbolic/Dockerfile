# ----------- 베이스 이미지 -----------
FROM python:3.10-slim

# ----------- 작업 디렉토리 -----------
WORKDIR /app

# ----------- 시스템 패키지 설치 -----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ----------- 의존성 설치 -----------
# requirements 먼저 복사 → 캐시 활용 가능
COPY requirements.txt /app/requirements.txt

# PyTorch CPU 버전 설치 (GPU 필요 없음)
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip install --no-cache-dir -r /app/requirements.txt

# ----------- 코드 복사 -----------
# 의존성 설치 후 코드 복사 → requirements 변경 없는 한 캐시 재사용 가능
COPY . /app/

# ----------- 권한 설정 및 초기화 -----------
# 권한 설정
RUN chmod -R 777 /app

# 디렉토리 생성
RUN mkdir -p /app/models/fallback-npc-model \
    && mkdir -p /app/models/sentence-embedder \
    && mkdir -p /app/rag/chroma_DB \
    && mkdir -p /app/.cache/huggingface/hub \
    mkdir -p /app/.cache/sentence_transformers && chmod -R 777 /app/.cache

# 환경변수 설정
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

ENV FALLBACK_MODEL_NAME=skt/ko-gpt-trinity-1.2B-v0.5
ENV EMBEDDER_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


# ----------- 포트 설정 -----------
EXPOSE 7860

# ----------- 실행 명령 -----------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

