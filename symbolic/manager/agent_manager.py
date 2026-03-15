from typing import Dict, List
from rag.rag_manager import retrieve

class NPCAgent:
    def __init__(self, npc_id: str):
        self.npc_id = npc_id
        self.cache: Dict[str, Dict[str, List[dict]]] = {}  # quest_stage:location별 캐시

    def load_rag_bundle(self, quest_stage: str, location: str) -> Dict[str, List[dict]]:
        """
        해당 NPC/퀘스트 스테이지/위치의 모든 문서를 한 번에 로드하고 type별로 분류.
        quest_stage/location이 'any'인 문서도 병합.
        """
        cache_key = f"{quest_stage}:{location}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        filters_base = {"npc_id": self.npc_id}

        # 1. 정확히 일치
        docs_exact = retrieve(f"{self.npc_id}:bundle", filters={**filters_base, "quest_stage": quest_stage, "location": location}, top_k=50) or []
        # 2. quest_stage=any
        docs_any_stage = retrieve(f"{self.npc_id}:bundle", filters={**filters_base, "quest_stage": "any", "location": location}, top_k=50) or []
        # 3. location=any
        docs_any_loc = retrieve(f"{self.npc_id}:bundle", filters={**filters_base, "quest_stage": quest_stage, "location": "any"}, top_k=50) or []
        # 4. quest_stage=any, location=any
        docs_global = retrieve(f"{self.npc_id}:bundle", filters={**filters_base, "quest_stage": "any", "location": "any"}, top_k=50) or []

        all_docs = docs_exact + docs_any_stage + docs_any_loc + docs_global

        # type별 분류
        bundle: Dict[str, List[dict]] = {}
        for doc in all_docs:
            t = doc.get("type", "unknown")
            bundle.setdefault(t, []).append(doc)

        self.cache[cache_key] = bundle
        return bundle


class AgentManager:
    def __init__(self):
        self.agents: Dict[str, NPCAgent] = {}

    def get_agent(self, npc_id: str) -> NPCAgent:
        if npc_id not in self.agents:
            self.agents[npc_id] = NPCAgent(npc_id)
        return self.agents[npc_id]


# 전역 인스턴스
agent_manager = AgentManager()
