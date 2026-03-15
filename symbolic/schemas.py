from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class NPCConfig(BaseModel):
    id: Optional[str] = Field(None, description="NPC 고유 ID (설계 기준)")
    name: Optional[str] = Field(None, description="NPC 표시 이름")
    persona_name: Optional[str] = Field(None, description="NPC 페르소나 이름")
    dialogue_style: Optional[str] = Field(None, description="대화 스타일")
    relationship: Optional[float] = Field(None, description="기본 관계 수치 (-1.0~1.0)")
    npc_mood: Optional[str] = Field(None, description="기본 감정 상태")
    trigger_values: Optional[Dict[str, List[str]]] = Field(None, description="트리거 값 목록")
    trigger_definitions: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="트리거 정의")

class DialogueTurn(BaseModel):
    player: str
    npc: str

class Context(BaseModel):
    require: Optional[Dict[str, Any]] = Field(None, description="pre 1차 조건 판단용 필수/선택 요소[rag 문서에 작성됨]")
    player_state: Dict[str, Any] = Field(..., description="플레이어 현재 상태")
    game_state: Dict[str, Any] = Field(..., description="게임 전역 상태")
    npc_state: Dict[str, Any] = Field(..., description="DB 최신 NPC 상태")
    npc_config: Optional[NPCConfig] = Field(None, description="RAG 기반 설계 정보")
    dialogue_history: Optional[List[DialogueTurn]] = Field(default_factory=list, description="최근 대화 히스토리")


class AskReq(BaseModel):
    session_id: str = Field(..., description="세션 고유 ID")
    npc_id: str = Field(..., description="NPC 고유 ID")
    user_input: str = Field(..., description="플레이어 입력 문장")
    context: Context = Field(..., description="게임 및 NPC 상태 정보")

class AskRes(BaseModel):
    session_id: str
    npc_output_text: str
    deltas: Dict[str, float] = Field(default_factory=dict, description="이번 턴 변화량")
    flags: Dict[str, int] = Field(default_factory=dict, description="플래그 이진값 {flag_name: 0|1}")
    valid: bool
    meta: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터 (npc_id, quest_stage, location 등)")
