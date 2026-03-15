import httpx
from typing import Any, Dict
from config import HF_SERVE_URL, HF_TIMEOUT

async def _post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hugging Face Spaces에 POST 요청을 보내는 내부 함수.
    endpoint는 '/predict_main' 같은 상대 경로.
    """
    url = f"{HF_SERVE_URL.rstrip('/')}{endpoint}"
    async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

async def call_main(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    메인 모델 추론 호출 함수.
    """
    return await _post("/predict_main", payload)

'''
----------- 아래 내용은 ai-server내부적으로 구현 [추후 수정 가능]--------------     

async def call_preprocess(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await _post("/predict_preprocess", payload)


async def call_postprocess(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await _post("/predict_postprocess", payload)

    
async def call_rag(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RAG 기반 추론 호출 함수 (예: 문서 검색 + 생성).
    """
    return await _post("/hf-serve/predict_rag", payload)

async def call_adapter_test(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter 테스트용 엔드포인트 호출 함수.
    """
    return await _post("/hf-serve/test_adapter", payload)


-------------------------------------------------------------------------
'''