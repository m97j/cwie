import json, torch
from fastapi import Request
from manager.agent_manager import agent_manager
from models.fallback_model import generate_fallback_response
from utils.context_parser import ContextParser
from sentence_transformers import util

def _short_history(context: dict, max_turns: int = 3) -> list:
    short_history = []
    for h in context.get("dialogue_history", [])[-max_turns:]:
        if "player" in h and "npc" in h:
            short_history.append({"role": "player", "text": h["player"]})
            short_history.append({"role": "npc", "text": h["npc"]})
    return short_history

# def _load_forbidden_trigger_data(npc_id: str) -> dict:
#     docs = retrieve(f"{npc_id}:forbidden_trigger_list", filters={"npc_id": npc_id}, top_k=1)
#     if not docs:
#         return {}
#     try:
#         return json.loads(docs[0]) if isinstance(docs[0], str) else docs[0]
#     except Exception:
#         return {}

def _semantic_match_embedder(embedder, user_input: str, trigger_texts: list, threshold: float = 0.75):
    if not trigger_texts:
        return (False, 0.0, None)
    inp_emb = embedder.encode(user_input, convert_to_tensor=True)
    trg_embs = embedder.encode(trigger_texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(inp_emb, trg_embs).squeeze(0)
    max_score, idx = torch.max(cos_scores, dim=0)
    score_val = float(max_score.item())
    matched_text = trigger_texts[int(idx.item())]
    return (score_val >= threshold, score_val, matched_text)

async def extract_emotion_via_fallback(request: Request, user_input: str) -> str:
    prompt = (
        "다음 문장의 화자 감정을 한 단어 또는 짧은 문장으로 설명하시오.\n\n"
        f"[문장]\n{user_input}\n\n"
        "지시:\n- 감정을 직접적으로 표현하지 않아도 문맥을 통해 추론하시오.\n"
        "- 가능한 경우 감정의 강도나 뉘앙스도 반영하시오.\n"
        "- 예: 분노, 슬픔, 혼란, 기대, 무관심, 초조함 등\n"
        "- 단어 하나 또는 짧은 문장으로만 출력하시오.\n\n"
        "정답:"
    )
    response = await generate_fallback_response(request, prompt)
    return response.strip()

async def _llm_trigger_check(request: Request, user_input: str, label_list: list) -> bool:
    if not label_list:
        return False
    criteria_block = "\n".join(f"- {c}" for c in label_list)
    prompt = (
        "다음은 의미 비교를 위한 판단 기준과 검사 대상입니다.\n\n"
        "[CRITERIA]\n"
        f"{criteria_block}\n"
        "[/CRITERIA]\n\n"
        "[INPUT]\n"
        f"{user_input}\n"
        "[/INPUT]\n\n"
        "지시:\n"
        "- [INPUT] 내용이 [CRITERIA] 항목 중 하나와 의미가 같거나 유사하면 YES, 그렇지 않으면 NO만 출력하시오.\n"
        "- 단어 그대로 포함되지 않아도 의미가 유사하면 YES로 간주하시오.\n"
        "- 확신이 없거나 판단이 애매하면 NO를 출력하시오.\n\n"
        "정답:"
    )
    txt = await generate_fallback_response(request, prompt)
    ans = txt.strip().upper()
    normalized = ans.replace(".", "").replace("!", "").strip()
    return (
        normalized == "YES" or
        normalized == "Y" or
        normalized.startswith("YES") or
        normalized.startswith("Y") or
        normalized.startswith("예") or
        normalized.startswith("네")
    )

async def preprocess_input(
    request: Request,
    session_id: str,
    npc_id: str,
    user_input: str,
    context: dict
) -> dict:
    parser = ContextParser(context)
    emotion = await extract_emotion_via_fallback(request, user_input)

    require_items = context.get("require", {}).get("items", [])
    require_actions = context.get("require", {}).get("actions", [])
    require_game_state = context.get("require", {}).get("game_state", [])
    require_delta = context.get("require", {}).get("delta", {})

    quest_stage = parser.game.get("quest_stage", "default")
    location = parser.game.get("location", context.get("location", "unknown"))

    # --- RAG bundle 로드 ---
    agent = agent_manager.get_agent(npc_id)
    bundle = agent.load_rag_bundle(quest_stage, location)

    # === 1차 검사: trigger_def 기반 ===
    td_docs = bundle.get("trigger_def", [])
    if td_docs:
        td = td_docs[0]
        trig = td.get("trigger", {})

        text_ok = not trig.get("required_text") or any(t in user_input for t in trig["required_text"])
        items_ok = not trig.get("required_items", {}).get("mandatory") or set(trig["required_items"]["mandatory"]).issubset(set(require_items))
        actions_ok = not trig.get("required_actions", {}).get("mandatory") or set(trig["required_actions"]["mandatory"]).issubset(set(require_actions))
        gs_ok = not trig.get("required_game_state", {}).get("mandatory") or set(trig["required_game_state"]["mandatory"]).issubset(set(require_game_state))
        delta_ok = all(require_delta.get(k, 0) >= v for k, v in trig.get("required_delta", {}).get("mandatory", {}).items())

        if text_ok and items_ok and actions_ok and gs_ok and delta_ok:
            return {
                "session_id": session_id,
                "player_utterance": user_input,
                "npc_id": npc_id,
                "tags": parser.npc,
                "player_state": parser.player,
                "game_state": parser.game,
                "context": _short_history(context),
                "emotion": emotion,
                "triggers": trig,
                "is_valid": True,
                "additional_trigger": None,
                "rag_main_docs": (
                    td_docs
                    + bundle.get("lore", [])
                    + bundle.get("description", [])
                    + bundle.get("npc_persona", [])
                    + bundle.get("dialogue_turn", [])
                    + bundle.get("flag_def", [])
                    + bundle.get("main_res_validate", [])
                ),
                "rag_fallback_docs": bundle.get("fallback", []) + bundle.get("npc_persona", []),
                "trigger_meta": {}
            }

    # === 2차 검사: forbidden-trigger 기반 ===
    forbidden_data = bundle.get("forbidden_trigger_list", [{}])[0]
    keywords = forbidden_data.get("triggers", {}).get("keywords", [])
    trigger_texts = forbidden_data.get("triggers", {}).get("text", [])

    embedder = request.app.state.embedder
    matched_key = None
    confidence = 0.0
    kw_match = None
    txt_match = None

    # 1. keyword 유사도 검사
    kw_hit, kw_score, kw_match = _semantic_match_embedder(embedder, user_input, keywords, threshold=0.75)

    # 2. text 유사도 검사
    txt_hit, txt_score, txt_match = _semantic_match_embedder(embedder, user_input, trigger_texts, threshold=0.75)

    # 3. 유사도 높은 쪽 선택
    if kw_hit and (kw_score >= txt_score):
        matched_key = "keyword_match"
        confidence = kw_score
    elif txt_hit:
        matched_key = "text_match"
        confidence = txt_score
    elif max(kw_score, txt_score) >= 0.65:
        # 가장 가까운 keyword와 text만 label 후보로 전달
        label_candidates = []
        if kw_match:
            label_candidates.append(kw_match)
        if txt_match:
            label_candidates.append(txt_match)

        if await _llm_trigger_check(request, user_input, label_candidates):
            matched_key = "semantic_match_llm"
            confidence = max(kw_score, txt_score)

    # === trigger_meta 매칭 보정 ===
    actual_trigger = None
    if matched_key:
        # kw_match나 txt_match 값이 실제 trigger_meta.trigger 값과 일치하는지 확인
        for tm in bundle.get("trigger_meta", []):
            if tm.get("trigger") in (kw_match, txt_match):
                actual_trigger = tm.get("trigger")
                break

    trigger_meta = {}
    if actual_trigger:
        trigger_meta = next((tm for tm in bundle.get("trigger_meta", []) if tm.get("trigger") == actual_trigger), {})
        trigger_meta["confidence"] = confidence

    additional_trigger = bool(actual_trigger)

    return {
        "session_id": session_id,
        "player_utterance": user_input,
        "npc_id": npc_id,
        "tags": parser.npc,
        "player_state": parser.player,
        "game_state": parser.game,
        "context": _short_history(context),
        "emotion": emotion,
        "triggers": [],
        "is_valid": False,
        "additional_trigger": additional_trigger,
        "rag_main_docs": (
            bundle.get("lore", [])
            + bundle.get("description", [])
            + bundle.get("npc_persona", [])
            + bundle.get("dialogue_turn", [])
            + bundle.get("flag_def", [])
            + bundle.get("main_res_validate", [])
        ),
        "rag_fallback_docs": bundle.get("fallback", []) + bundle.get("npc_persona", []),
        "trigger_meta": trigger_meta
    }
