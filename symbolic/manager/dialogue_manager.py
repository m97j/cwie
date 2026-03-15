from fastapi import Request
from pipeline.preprocess import preprocess_input
from pipeline.generator import generate_response
from pipeline.postprocess import postprocess_fallback, postprocess_main
from models.fallback_model import generate_fallback_response
from .prompt_builder import build_main_prompt, build_fallback_prompt

async def handle_dialogue(
    request: Request,
    session_id: str,
    npc_id: str,
    user_input: str,
    context: dict,
) -> dict:
    """
    전체 대화 처리 파이프라인:
      1) preprocess_input() → pre 데이터 생성
      2) main 경로: main prompt → main model → postprocess_pipeline()
      3) fallback 경로: fallback prompt → fallback model → fallback_final_check()
    """
    # 1. Preprocess
    pre = await preprocess_input(request, session_id, npc_id, user_input, context)

    # 2. Fallback 경로
    if not pre.get("is_valid", True):
        # fallback prompt 구성 (내부에서 additional_trigger 기반 분기)
        fb_prompt = build_fallback_prompt(pre, session_id, npc_id)

        # fallback model 호출
        fb_raw = await generate_fallback_response(request, fb_prompt)

        return_payload_fb = postprocess_fallback(request, pre, fb_raw)

        return return_payload_fb

    # 3. Main 경로
    main_prompt = build_main_prompt(pre, session_id, npc_id)

    # main model 호출
    result = await generate_response(session_id, npc_id, main_prompt, max_tokens=200)

    # postprocess_pipeline에서 최종 payload 생성
    return_payload_main = await postprocess_main(
        request=request,
        pre_data=pre,           # preprocess 결과 전체 전달
        model_payload=result,   # main model 출력
    )

    return return_payload_main
