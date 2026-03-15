from typing import Dict, Any

def build_webtest_prompt(npc_id: str, npc_location: str, player_utt: str) -> str:
    """
    Web Test 전용: 최소 입력값(NPC ID, Location, Player 발화)으로
    모델 학습 포맷에 맞는 prompt 문자열을 생성.
    """
    pre = {
        "npc_id": npc_id,
        "npc_location": npc_location,
        "tags": {
            "quest_stage": "",
            "relationship": "",
            "trust": "",
            "npc_mood": "",
            "player_reputation": "",
            "style": ""
        },
        "player_state": {
            "items": [],
            "actions": [],
            "position": ""
        },
        "rag_main_docs": [],
        "context": [],
        "player_utterance": player_utt
    }
    return _assemble_prompt_for_model(pre)

def _assemble_prompt_for_model(pre: Dict[str, Any]) -> str:
    """
    Web Test 전용 내부 함수:
    pre dict → 모델 입력 포맷 문자열(<SYS>~<NPC>)
    """

    tags = pre.get("tags", {})
    ps = pre.get("player_state", {})
    rag_docs = pre.get("rag_main_docs", [])

    # RAG 문서 분리
    lore_text = ""
    desc_text = ""
    for doc in rag_docs:
        if "LORE:" in doc:
            lore_text += doc + "\n"
        elif "DESCRIPTION:" in doc:
            desc_text += doc + "\n"
        else:
            if "lore" in doc.lower():
                lore_text += doc + "\n"
            elif "description" in doc.lower():
                desc_text += doc + "\n"

    prompt = [
        "<SYS>",
        f"NPC_ID={pre.get('npc_id','')}",
        f"NPC_LOCATION={pre.get('npc_location','')}",
        "TAGS:",
        f" quest_stage={tags.get('quest_stage','')}",
        f" relationship={tags.get('relationship','')}",
        f" trust={tags.get('trust','')}",
        f" npc_mood={tags.get('npc_mood','')}",
        f" player_reputation={tags.get('player_reputation','')}",
        f" style={tags.get('style','')}",
        "</SYS>",
        "<RAG>",
        f"LORE: {lore_text.strip() or '(없음)'}",
        f"DESCRIPTION: {desc_text.strip() or '(없음)'}",
        "</RAG>",
        "<PLAYER_STATE>",
        f"items={','.join(ps.get('items', []))}" if ps.get("items") else "items=",
        f"actions={','.join(ps.get('actions', []))}" if ps.get("actions") else "actions=",
        f"position={ps.get('position','')}",
        "</PLAYER_STATE>",
        "<CTX>"
    ]

    for h in pre.get("context", []):
        prompt.append(f"{h['role']}: {h['text']}")
    prompt.append("</CTX>")

    prompt.append(f"<PLAYER>{pre.get('player_utterance','').rstrip()}")
    prompt.append("<STATE>")
    prompt.append("<NPC>")

    return "\n".join(prompt)
