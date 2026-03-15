import os
import torch
from dotenv import load_dotenv

# .env 파일 로드 (로컬 개발 시)
load_dotenv()

# 모델 경로 (환경변수 없으면 기본값 사용)
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTERS = os.getenv("ADAPTER_MODEL", "m97j/npc_LoRA-fps")

# 장치 설정
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저/모델 공통
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 1024))
NUM_FLAGS = int(os.getenv("NUM_FLAGS", 7))  # flags.json 길이와 일치

# 생성 파라미터
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", 400))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", 0.7))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", 0.9))

# Hugging Face Token (Private 모델 접근용)
HF_TOKEN = os.getenv("HF_TOKEN")
