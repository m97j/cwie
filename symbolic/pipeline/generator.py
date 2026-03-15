from utils.hf_client import call_main

'''
async def generate_response(session_id: str, npc_id: str, prompt: str, max_tokens: int = 200,
                             temperature: float = 0.7, top_p: float = 0.9,
                             do_sample: bool = True, repetition_penalty: float = 1.05) -> dict:
    payload = {
        "session_id": session_id,
        "npc_id": npc_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty
    }
    return await call_main(payload)


'''
async def generate_response(session_id: str, npc_id: str, prompt: str, max_tokens: int = 200) -> dict:
    payload = {
        "session_id": session_id,
        "npc_id": npc_id,
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    return await call_main(payload)  # {"text":..., "delta":..., "flag":...}
#'''