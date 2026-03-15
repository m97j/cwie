import torch, random, re
from typing import Dict, Any, List, Optional, Tuple
from fastapi import Request
from sentence_transformers import util
from models.fallback_model import generate_fallback_response

ALPHA_THR = 0.58
DELTA_CLAMP = (-1.0, 1.0)

# ----------------------------
# Utilities
# ----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _adjust_delta_with_rag(delta: Dict[str, float]) -> Dict[str, float]:
    trust = _clamp(float(delta.get("trust", 0.0)), *DELTA_CLAMP)
    rel   = _clamp(float(delta.get("relationship", 0.0)), *DELTA_CLAMP)
    return {"trust": trust, "relationship": rel}

def _embedding_similarity(embedder, text: str, examples: List[str]) -> float:
    if not examples:
        return 0.0
    inp_emb = embedder.encode(text, convert_to_tensor=True)
    ex_embs = embedder.encode(examples, convert_to_tensor=True)
    cos_scores = util.cos_sim(inp_emb, ex_embs)
    return float(torch.mean(cos_scores).item())

def _doc_type(doc: Dict[str, Any]) -> Optional[str]:
    if "type" in doc:
        return doc.get("type")
    return doc.get("metadata", {}).get("type")

def _get_flag_doc(rag_docs: List[Dict[str, Any]], flag_name: str) -> Dict[str, Any]:
    for doc in rag_docs:
        if _doc_type(doc) == "flag_def" and doc.get("flag_name") == flag_name:
            return doc
    return {}

def _get_turn_doc(rag_docs: List[Dict[str, Any]], npc_id: str, quest_stage: str) -> Dict[str, Any]:
    # 동일 npc_id/quest_stage인 가장 최신(turn_index 최대) 문서를 우선 반환
    candidates = [
        d for d in rag_docs
        if _doc_type(d) == "dialogue_turn"
        and d.get("npc_id") == npc_id
        and d.get("quest_stage") == quest_stage
    ]
    if not candidates:
        return {}
    return sorted(candidates, key=lambda d: d.get("turn_index", -1))[-1]

def _short_ctx_from_pre(pre_data: dict) -> str:
    pairs = pre_data.get("context", []) or []
    return "\n".join(f"{m.get('role', 'user')}: {m.get('text', '')}" for m in pairs)

async def fetch_response_policy_from_pre(pre_data: dict) -> str:
    for doc in pre_data.get("rag_main_docs", []):
        if _doc_type(doc) == "main_res_validate":
            return doc.get("text", "") or doc.get("chunk", "")
    return (
        "응답이 NPC persona와 현재 상태(delta, flags)에 부합하는지 검증하시오. "
        "부적절한 표현은 완화하고, 세계관을 유지하시오."
    )

# ----------------------------
# RAG helpers
# ----------------------------

def _extract_expected_delta(rag_docs: List[Dict[str, Any]]) -> Dict[str, float]:
    # trigger_def.delta_expected 우선, 없으면 dialogue_turn.delta 평균(선택)
    expected = {}
    for doc in rag_docs:
        if _doc_type(doc) == "trigger_def" and doc.get("delta_expected"):
            expected.update(doc["delta_expected"])
    return expected

def _collect_value_contexts(rag_docs: List[Dict[str, Any]], value: str) -> List[str]:
    contexts = []
    for doc in rag_docs:
        # description/content/text 필드에서 value가 언급된 문장 수집
        for key in ("content", "text", "npc", "player"):
            if value and isinstance(doc.get(key), str) and value in doc[key]:
                contexts.append(doc[key])
    return contexts

def _weight_by_doc_type(t: str) -> float:
    # 등장 위치 가중치(상황에 맞게 조정)
    return {
        "dialogue_turn": 1.2,
        "trigger_def": 1.0,
        "description": 1.0,
        "npc_persona": 0.9,
        "lore": 0.7,
        "flag_def": 0.8,
        "main_res_validate": 0.8,
    }.get(t, 1.0)

def _collect_positive_negative_texts(rag_docs: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    pos, neg = [], []
    for doc in rag_docs:
        t = _doc_type(doc)
        w = _weight_by_doc_type(t)
        if isinstance(doc.get("examples_positive"), list):
            pos.extend([f"[{t}] {s}" for s in doc["examples_positive"]] * int(max(1, round(w))))
        if isinstance(doc.get("examples_good"), list):
            pos.extend([f"[{t}] {s}" for s in doc["examples_good"]] * int(max(1, round(w))))
        if isinstance(doc.get("examples_negative"), list):
            neg.extend([f"[{t}] {s}" for s in doc["examples_negative"]] * int(max(1, round(w))))
        if isinstance(doc.get("examples_bad"), list):
            neg.extend([f"[{t}] {s}" for s in doc["examples_bad"]] * int(max(1, round(w))))
    return pos, neg

# ----------------------------
# Delta 검증/보정
# ----------------------------

def _adjust_delta_with_rag_and_embedding(
    delta: Dict[str, float],
    rag_docs: List[Dict[str, Any]],
    embedder,
    player_utt: str,
    npc_text: str,
    flags_yes: List[str],
    sim_threshold: float = 0.72,
    diff_threshold: float = 0.18,
    blend: float = 0.6  # expected에 끌어당기는 비율
) -> Dict[str, float]:
    trust = _clamp(float(delta.get("trust", 0.0)), *DELTA_CLAMP)
    rel   = _clamp(float(delta.get("relationship", 0.0)), *DELTA_CLAMP)

    expected = _extract_expected_delta(rag_docs)
    pos, neg = _collect_positive_negative_texts(rag_docs)

    context_text = f"PLAYER: {player_utt}\nNPC: {npc_text}\nFLAGS: {', '.join(flags_yes) if flags_yes else 'none'}"
    pos_sim = _embedding_similarity(embedder, context_text, pos) if pos else 0.0
    neg_sim = _embedding_similarity(embedder, context_text, neg) if neg else 0.0
    # 맥락이 ‘긍정’에 가깝고 기대와 차이가 크면 기대 쪽으로 보정
    def _pull(val, key):
        if key in expected:
            exp = float(expected[key])
            if abs(val - exp) > diff_threshold and pos_sim - neg_sim >= (sim_threshold - 0.1):
                return _clamp(blend * exp + (1 - blend) * val, *DELTA_CLAMP)
        return val

    trust = _pull(trust, "trust")
    rel   = _pull(rel, "relationship")
    return {"trust": trust, "relationship": rel}

# ----------------------------
# Flag 보정 로직
# ----------------------------

def adjust_flags_with_rag_and_embedding(
    flags_prob: Dict[str, float],
    flags_thr: Dict[str, float],
    rag_flags_score: Dict[str, float],
    rag_flags_pred: Dict[str, int],
    embedder,
    npc_text: str,
    rag_positive_examples: Dict[str, List[str]],
    deltas_final: Dict[str, float],  # ← delta 보정 결과 반영
    rag_docs: List[Dict[str, Any]],
    alpha_model: float = 0.6,
    margin: float = 0.05,
    sim_threshold: float = 0.8,
    random_jitter: float = 0.05
) -> Dict[str, int]:
    # 전체 패턴 유사도
    model_vector = [flags_prob.get(name, 0.0) for name in rag_flags_score.keys()]
    rag_vector = [rag_flags_score.get(name, 0.0) for name in rag_flags_score.keys()]
    sim = float(
        embedder.encode([model_vector], convert_to_tensor=True)
        @ embedder.encode([rag_vector], convert_to_tensor=True).T
    )

    expected = _extract_expected_delta(rag_docs)

    final_preds = {}
    for name in rag_flags_score.keys():
        prob_model = float(flags_prob.get(name, 0.0))
        thr_model  = float(flags_thr.get(name, 0.5))
        score_rag  = float(rag_flags_score.get(name, 0.0))
        _ = int(rag_flags_pred.get(name, 0))

        emb_score = _embedding_similarity(embedder, npc_text, rag_positive_examples.get(name, []))

        # delta 일관성 보정(해당 flag가 예상될 때 delta와의 불일치 패널티)
        delta_penalty = 0.0
        if expected:
            # 신호가 양의 변화인데 모델 delta가 큰 음수인 경우 등
            if "trust" in expected and deltas_final.get("trust", 0.0) * expected["trust"] < 0:
                delta_penalty += 0.08
            if "relationship" in expected and deltas_final.get("relationship", 0.0) * expected["relationship"] < 0:
                delta_penalty += 0.06

        # 혼합 점수 + 임베딩 + 델타 정합
        blended_score = (
            alpha_model * prob_model
            + (1 - alpha_model) * score_rag
            + 0.2 * emb_score
            - delta_penalty
        )
        thr_blend = alpha_model * thr_model + (1 - alpha_model) * 0.5

        if abs(blended_score - thr_blend) <= margin:
            adjusted_score = score_rag if sim < sim_threshold else blended_score
        else:
            adjusted_score = blended_score

        if adjusted_score != score_rag:
            adjusted_score += random.uniform(-random_jitter, random_jitter)
            adjusted_score = max(0.0, min(1.0, adjusted_score))

        final_preds[name] = int(adjusted_score >= thr_blend)

    return final_preds

# ----------------------------
# Validators / Rewriters
# ----------------------------

async def validate_or_rewrite_response(
    request: Request,
    response_text: str,
    description_text: str,
    ctx_text: str,
    player_utt: str,
    deltas: Dict[str, float],
    flags_yes: List[str],
    flags_values: Dict[str, str],         # ← 추가
    value_contexts: Dict[str, List[str]], # ← 추가
) -> str:
    flag_value_info = "\n".join(f"- {k}: {v}" for k, v in flags_values.items()) if flags_values else "none"
    value_ctx_lines = []
    for k, arr in value_contexts.items():
        if arr:
            # 너무 길어지는 것을 방지하여 상위 1~2개만
            value_ctx_lines.append(f"- {k}: {arr[0]}")
            if len(arr) > 1:
                value_ctx_lines.append(f"  (more: {min(2, len(arr)-1)} refs)")
    value_ctx_info = "\n".join(value_ctx_lines) if value_ctx_lines else "none"

    prompt = (
        "다음은 게임 내 NPC 응답입니다.\n"
        f"[RESPONSE]\n{response_text}\n[/RESPONSE]\n\n"
        "아래의 검증 기준을 만족하는지 판단하고, 만족하지 않으면 기준에 맞게 자연스럽게 재작성하세요.\n"
        f"[FINAL_CHECK_DESCRIPTION]\n{description_text}\n[/FINAL_CHECK_DESCRIPTION]\n\n"
        "상태 정보:\n"
        f"- DELTA: trust={deltas.get('trust',0.0):.3f}, relationship={deltas.get('relationship',0.0):.3f}\n"
        f"- FLAGS(YES): {', '.join(flags_yes) if flags_yes else 'none'}\n"
        f"- FLAG_VALUES:\n{flag_value_info}\n"
        f"- VALUE_CONTEXTS:\n{value_ctx_info}\n\n"
        "맥락:\n"
        f"[CTX]\n{ctx_text}\n[/CTX]\n"
        f"[PLAYER]\n{player_utt}\n[/PLAYER]\n\n"
        "요구사항:\n"
        "- 기준을 만족하면 응답을 그대로 출력하되 민감한 표현은 완화하세요.\n"
        "- 기준을 만족하지 않으면 기준을 충족하도록 응답을 자연스럽게 재작성하세요.\n"
        "- 출력은 NPC의 최종 대사만 한 줄로 제공하세요."
    )
    fb_raw = await generate_fallback_response(request, prompt)
    return fb_raw.strip()

# ----------------------------
# Main path postprocess
# ----------------------------

async def postprocess_main(
    request: Request,
    pre_data: dict,
    model_payload: dict
) -> dict:
    embedder = request.app.state.embedder
    npc_id = pre_data["npc_id"]
    quest_stage = pre_data["game_state"].get("quest_stage", "default")
    location = pre_data["game_state"].get("location", "unknown")

    rag_docs = pre_data.get("rag_main_docs", [])
    npc_text_in = (model_payload.get("npc_output_text") or "").strip()
    player_utt = pre_data.get("player_utterance", "")

    # 1) Delta 검증/보정(의미 기반 + 기대값)
    deltas_in = model_payload.get("deltas", {}) or {}
    deltas_adj = _adjust_delta_with_rag_and_embedding(
        delta=deltas_in,
        rag_docs=rag_docs,
        embedder=embedder,
        player_utt=player_utt,
        npc_text=npc_text_in,
        flags_yes=[],
    )

    # 2) Flag 보정(임베딩/기대 델타 반영)
    flags_binary = adjust_flags_with_rag_and_embedding(
        flags_prob=model_payload.get("flags_prob", {}),
        flags_thr=model_payload.get("flags_thr", {}),
        rag_flags_score={doc["flag_name"]: doc.get("score_rag", 0.0) for doc in rag_docs if _doc_type(doc) == "flag_def"},
        rag_flags_pred={doc["flag_name"]: doc.get("pred_rag", 0) for doc in rag_docs if _doc_type(doc) == "flag_def"},
        embedder=embedder,
        npc_text=npc_text_in,
        rag_positive_examples={doc["flag_name"]: doc.get("examples_positive", []) for doc in rag_docs if _doc_type(doc) == "flag_def"},
        deltas_final=deltas_adj,
        rag_docs=rag_docs,
    )

    # 상세 정보 기록 + yes 리스트
    flags_detail = {}
    flags_yes_list: List[str] = []
    for name, pred in flags_binary.items():
        flag_doc = _get_flag_doc(rag_docs, name)
        score_model = float(model_payload.get("flags_prob", {}).get(name, 0.0))
        thr_model = float(model_payload.get("flags_thr", {}).get(name, 0.5))
        rag_thr = float(flag_doc.get("threshold", 0.5)) if flag_doc else 0.5
        examples_pos = flag_doc.get("examples_positive", []) if flag_doc else []
        emb_score = _embedding_similarity(embedder, npc_text_in, examples_pos) if examples_pos else 0.0
        thr_blend = ALPHA_THR * thr_model + (1.0 - ALPHA_THR) * rag_thr

        flags_detail[name] = {
            "score_model": score_model,
            "thr_model": thr_model,
            "thr_rag": rag_thr,
            "thr_blend": thr_blend,
            "emb_score": emb_score,
            "pred": pred
        }
        if pred == 1:
            flags_yes_list.append(name)

    # 3) Flag value 추출(대화 턴 실제 값 우선) + value 맥락 수집
    flags_values: Dict[str, str] = {}
    value_contexts: Dict[str, List[str]] = {}
    turn_doc = _get_turn_doc(rag_docs, npc_id, quest_stage)

    def _turn_flag_value(doc: Dict[str, Any], fname: str) -> Optional[str]:
        if not doc:
            return None
        # 리스트 구조 전제
        flags = doc.get("flags")
        if isinstance(flags, list):
            for f in flags:
                if f.get("flag_name") == fname:
                    return f.get("flag_value")
        # 하위호환: dict인 경우 yes(1)/no(0)만 제공됨
        if isinstance(flags, dict) and fname in flags:
            return "yes" if flags.get(fname) else "no"
        return None

    for name in flags_yes_list:
        if name in ["give_item", "npc_action", "change_player_state", "change_game_state"]:
            val = _turn_flag_value(turn_doc, name)
            if val:
                flags_values[name] = val
                value_contexts[name] = _collect_value_contexts(rag_docs, val)

    # 3-1) value 일치성 임베딩 검증(응답과 value 맥락의 유사도)
    # 유사도가 낮으면 response 재작성에서 보정되도록 힌트 제공
    # (여기서 바로 값을 바꾸지는 않고, 검증 프롬프트에 context로 전달)
    # 필요 시 하드 트리거를 추가할 수 있음

    # 4) 응답 검증/재작성(최종 delta/flags/value 기준)
    desc_text = await fetch_response_policy_from_pre(pre_data)
    ctx_text = _short_ctx_from_pre(pre_data)

    npc_text_out = await validate_or_rewrite_response(
        request=request,
        response_text=npc_text_in,
        description_text=desc_text,
        ctx_text=ctx_text,
        player_utt=player_utt,
        deltas=deltas_adj,
        flags_yes=flags_yes_list,
        flags_values=flags_values,
        value_contexts=value_contexts,
    )

    return {
        "session_id": model_payload.get("session_id"),
        "npc_output_text": npc_text_out,
        "deltas": deltas_adj,  # 보정 완료 델타
        "flags": {k: 1 if k in flags_yes_list else 0 for k in flags_binary.keys()},
        "valid": True,
        "meta": {
            "npc_id": npc_id,
            "quest_stage": quest_stage,
            "location": location,
            "additional_trigger": pre_data.get("additional_trigger", False),
            "trigger_meta": pre_data.get("trigger_meta", {}),
            "flags_detail": flags_detail,
            "flags_values": flags_values,
            "value_contexts": value_contexts,
        }
    }


# ----------------------------
# Fallback path postprocess
# ----------------------------

async def fallback_final_check(
    request: Request,
    fb_response: str,
    player_utt: str,
    npc_config: dict,
    action_delta: dict
) -> str:
    """
    fallback 응답의 최종 보정:
      1) npc_action / npc_emotion / delta와 의미적 일치
      2) 세계관 및 안전성(표현 완화)
    """
    checks = []
    npc_action = action_delta.get("npc_action")
    npc_emotion = action_delta.get("npc_emotion")
    delta = action_delta.get("delta", {}) or {}

    if npc_action:
        checks.append(f"NPC는 '{npc_action}' 행동을 반영해야 함")
    if npc_emotion:
        checks.append(f"NPC는 '{npc_emotion}' 감정을 표현해야 함")
    for name, value in delta.items():
        direction = "긍정적" if value > 0.5 else "부정적" if value < -0.5 else "중립적"
        checks.append(f"{name} 값({value:.2f})은 {direction} 방향이며, 이에 맞는 반응이어야 함")

    checks.append("응답이 NPC persona와 세계관에 부합해야 함")
    checks.append("민감한 표현은 완화해야 함")

    delta_desc = ", ".join([f"{k}={v:.2f}(-1.0~1.0)" for k, v in delta.items()]) or "없음"

    prompt = (
        "다음은 게임 내 NPC의 응답입니다.\n"
        f"[RESPONSE]\n{fb_response}\n[/RESPONSE]\n\n"
        "검증 기준:\n" + "\n".join(f"- {c}" for c in checks) + "\n\n"
        f"플레이어 발화: {player_utt}\n"
        "요구사항:\n"
        "- 기준을 만족하면 응답을 그대로 출력하세요.\n"
        "- 기준을 만족하지 않으면 기준에 부합하도록 자연스럽게 수정하세요.\n"
        "- 출력은 NPC의 최종 대사만 한 줄로 제공하세요.\n\n"
        "NPC 상태 요약:\n"
        f"- ACTION: {npc_action or '없음'}\n"
        f"- EMOTION: {npc_emotion or '없음'}\n"
        f"- DELTA: {delta_desc}\n"
    )

    fb_checked = await generate_fallback_response(request, prompt)
    return fb_checked.strip()


async def postprocess_fallback(
    request: Request,
    pre_data: dict,
    fb_raw_text: str
) -> dict:
    """
    Fallback 모델 출력에 대해:
      - 특수 fallback이면 action/delta 반영하여 최종 보정
      - deltas는 pre_data.trigger_meta.delta를 이번 턴 변화량으로 사용
      - flags는 기본적으로 비어있음(필요 시 pre에서 확정 가능)
    """
    npc_id = pre_data["npc_id"]
    quest_stage = pre_data["game_state"].get("quest_stage", "default")
    location = pre_data["game_state"].get("location", "unknown")

    trigger_meta = pre_data.get("trigger_meta", {}) or {}
    action_delta = {
        "npc_action": trigger_meta.get("npc_action"),
        "npc_emotion": trigger_meta.get("npc_emotion"),
        "delta": trigger_meta.get("delta", {}) or {}
    }

    # 이번 턴 변화량(특수 fallback의 경우 trigger_meta.delta가 기준)
    deltas_adj = _adjust_delta_with_rag(action_delta.get("delta", {}))

    # 특수 fallback 보정
    player_utt = pre_data.get("player_utterance", "")
    npc_config = pre_data.get("tags", {}) or {}

    if pre_data.get("additional_trigger", False):
        fb_checked = await fallback_final_check(
            request=request,
            fb_response=fb_raw_text,
            player_utt=player_utt,
            npc_config=npc_config,
            action_delta={"npc_action": action_delta.get("npc_action"),
                          "npc_emotion": action_delta.get("npc_emotion"),
                          "delta": deltas_adj}
        )
    else:
        fb_checked = fb_raw_text.strip()

    return {
        "session_id": pre_data.get("session_id"),
        "npc_output_text": fb_checked,
        "deltas": deltas_adj,          # 이번 턴 변화량
        "flags": {},                   # 기본 비어 있음(필요 시 pre 단계에서 확정 가능)
        "valid": False,
        "meta": {
            "npc_id": npc_id,
            "quest_stage": quest_stage,
            "location": location,
            "additional_trigger": pre_data.get("additional_trigger", False),
            "trigger_meta": trigger_meta
        }
    }
