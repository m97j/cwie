from typing import Dict, Any

def build_main_prompt(pre: Dict[str, Any], session_id: str, npc_id: str) -> str:
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
            # fallback: type 기반 분리 가능
            if "lore" in doc.lower():
                lore_text += doc + "\n"
            elif "description" in doc.lower():
                desc_text += doc + "\n"

    prompt = [
        "<SYS>",
        f"NPC_ID={tags.get('npc_id','')}",
        f"NPC_LOCATION={tags.get('location','')}",
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
        "<PLAYER_STATE>"
    ]

    if ps.get("items"):
        prompt.append(f"items={','.join(ps['items'])}")
    if ps.get("actions"):
        prompt.append(f"actions={','.join(ps['actions'])}")
    if ps.get("position"):
        prompt.append(f"position={ps['position']}")
    prompt.append("</PLAYER_STATE>")

    prompt.append("<CTX>")
    for h in pre.get("context", []):
        prompt.append(f"{h['role']}: {h['text']}")
    prompt.append("</CTX>")

    prompt.append(f"<PLAYER>{pre.get('player_utterance','').rstrip()}")
    prompt.append("<STATE>")
    prompt.append("<NPC>")

    return "\n".join(prompt)



def build_fallback_prompt(pre: Dict[str, Any], session_id: str, npc_id: str) -> str:
    """
    additional_trigger 값에 따라 일반 fallback / 특수 fallback 프롬프트를 한 함수에서 처리
    """
    tags = pre.get("tags", {})
    ps = pre.get("player_state", {})
    gs = pre.get("game_state", {})
    rag_text = "\n".join(f"- {doc}" for doc in pre.get("rag_fallback_docs", []))
    fb_style = pre.get("fallback_style") or {}
    trigger_meta = pre.get("trigger_meta", {}) or {}

    items = ",".join(ps.get("items", []))
    actions = ",".join(ps.get("actions", []))
    location = gs.get("location") or ps.get("location", "unknown")
    quest_stage = gs.get("quest_stage", "unknown")

    # 기본 안내문
    instr = (
        "당신은 NPC persona를 가진 캐릭터입니다. "
        "플레이어 발화에 자연스럽고 맥락에 맞는 대사를 생성하세요. "
        "스토리 진행 조건은 충족되지 않았습니다."
    )

    # additional_trigger=True → 특수 fallback
    if pre.get("additional_trigger"):
        # trigger_meta 기반 구체화
        s = fb_style.get("style") or trigger_meta.get("npc_style")
        a = fb_style.get("npc_action") or trigger_meta.get("npc_action")
        e = fb_style.get("npc_emotion") or trigger_meta.get("npc_emotion")
        more = []
        if s: more.append(f"대화 스타일={s}")
        if a: more.append(f"NPC 행동={a}")
        if e: more.append(f"NPC 감정={e}")
        if more:
            instr += " " + "; ".join(more) + "."
        # 특수 fallback임을 명시
        instr += " 이 반응은 플레이어의 특정 발화(금지 트리거)에 의해 유발된 것입니다."

    return f"""
<FALLBACK>
NPC_ID={npc_id}
SESSION_ID={session_id}
LOCATION={location}
QUEST_STAGE={quest_stage}
MOOD={tags.get("npc_mood","neutral")}
STYLE={tags.get("style","neutral")}
ITEMS={items}
ACTIONS={actions}
EMOTION_SUMMARY={', '.join([f"{k}:{round(v,2)}" for k,v in pre.get('emotion',{}).items()])}
INPUT="{pre['player_utterance']}"

RAG_CONTEXT:
{rag_text or "(none)"}

INSTRUCTION:
{instr}
</FALLBACK>
""".strip()






'''
def build_fallback_prompt(pre: dict, session_id: str, npc_id: str) -> str:
    tags = pre.get("tags", {})
    ps = pre.get("player_state", {})
    gs = pre.get("game_state", {})
    rag_text = "\n".join(f"- {doc}" for doc in pre.get("rag_fallback_docs", []))
    fb = pre.get("fallback_style") or {}

    items = ",".join(ps.get("items", []))
    actions = ",".join(ps.get("actions", []))
    location = gs.get("location") or ps.get("location", "unknown")
    quest_stage = gs.get("quest_stage", "unknown")

    instr = "조건 불충족. 스토리 진행은 하지 않고, 캐릭터 일관성을 유지하며 자연스럽게 응답하라."
    if fb:
        # 선택적 구체화
        s = fb.get("style"); a = fb.get("npc_action"); e = fb.get("npc_emotion")
        more = []
        if s: more.append(f"대화 스타일={s}")
        if a: more.append(f"NPC 행동={a}")
        if e: more.append(f"NPC 감정={e}")
        if more:
            instr += " " + "; ".join(more) + "."

    return f"""
<FALLBACK>
NPC_ID={npc_id}
SESSION_ID={session_id}
LOCATION={location}
QUEST_STAGE={quest_stage}
MOOD={tags.get("npc_mood","neutral")}
STYLE={tags.get("style","neutral")}
ITEMS={items}
ACTIONS={actions}
EMOTION_SUMMARY={', '.join([f"{k}:{round(v,2)}" for k,v in pre.get('emotion',{}).items()])}
INPUT="{pre['player_utterance']}"

RAG:
{rag_text or "(none)"}

INSTRUCTION:
{instr}
</FALLBACK>
""".strip()
'''