import os, json
from typing import List, Dict, Any, Optional
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from config import CHROMA_DIR

# === 초기화 ===
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
print(f"📂 ChromaDB 경로: {CHROMA_DIR.resolve()}")

_client = PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(name="game_docs")
_embedder: Optional[EmbeddingFunction] = None


# === Embedder 설정 ===
def set_embedder(embedder: Any):
    global _embedder
    _embedder = embedder


def chroma_initialized() -> bool:
    return os.path.exists(str(CHROMA_DIR)) and len(os.listdir(str(CHROMA_DIR))) > 0


# === type별 content 추출 ===
def extract_content(doc: Dict[str, Any]) -> str:
    """문서 type에 따라 content 필드를 생성"""
    if "content" in doc and isinstance(doc["content"], str):
        return doc["content"]

    t = doc.get("type", "").lower()
    if t in ["description", "lore", "fallback", "main_res_validate", "npc_persona"]:
        return doc.get("description", "") or doc.get("content", "")
    elif t == "trigger_def":
        return doc.get("description", json.dumps(doc.get("trigger", {}), ensure_ascii=False))
    elif t == "dialogue_turn":
        # player + npc 대사를 합쳐서 저장
        return f"PLAYER: {doc.get('player', '')}\nNPC: {doc.get('npc', '')}".strip()
    elif t == "flag_def":
        return "\n".join(doc.get("examples_positive", []))
    elif t == "trigger_meta":
        return doc.get("trigger", "")
    else:
        # 알 수 없는 type이면 가능한 모든 텍스트 필드 합침
        text_parts = []
        for k, v in doc.items():
            if isinstance(v, str):
                text_parts.append(v)
        return "\n".join(text_parts)


# === 디스크에서 문서 로드 ===
def load_game_docs_from_disk(path: str) -> List[Dict[str, Any]]:
    docs = []
    for filename in os.listdir(path):
        full = os.path.join(path, filename)
        if filename.endswith(".json"):
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for i, doc in enumerate(data):
                        if "id" not in doc:
                            doc["id"] = f"{filename}_{i}"
                        doc["content"] = extract_content(doc)
                        docs.append(doc)
                elif isinstance(data, dict):
                    if "id" not in data:
                        data["id"] = filename
                    data["content"] = extract_content(data)
                    docs.append(data)
        elif filename.endswith(".txt"):
            with open(full, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append({
                    "id": filename,
                    "type": "text",
                    "content": content,
                    "metadata": {}
                })
    return docs


# === 문서 추가 ===
def add_docs(docs: List[Dict[str, Any]], batch_size: int = 32):
    assert _embedder is not None, "Embedder not initialized"
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        ids, contents, embeddings, metadatas = [], [], [], []
        for doc in batch:
            # id는 필수, content는 없으면 빈 문자열
            doc_id = doc.get("id", f"doc_{i}")
            content = doc.get("content", "")
            ids.append(doc_id)
            contents.append(content)
            metadatas.append(doc)  # 원본 전체 저장
            emb = _embedder.encode(content).tolist() if content else []
            embeddings.append(emb)
        _collection.add(
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )


# === 문서 검색 ===
def retrieve(query: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    assert _embedder is not None, "Embedder not initialized"

    if query:
        q_emb = _embedder.encode(query).tolist()
        res = _collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=filters or {}
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
    else:
        res = _collection.get(
            where=filters or {},
            limit=top_k
        )
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])

    # 원본 구조 복원
    results = []
    for d, m in zip(docs, metas):
        if isinstance(m, dict):
            results.append({
                "id": m.get("id", ""),
                "type": m.get("type", "unknown"),
                "npc_id": m.get("npc_id", ""),
                "quest_stage": m.get("quest_stage", ""),
                "location": m.get("location", ""),
                "content": d,
                "metadata": m
            })
        else:
            results.append({
                "id": "",
                "type": "unknown",
                "npc_id": "",
                "quest_stage": "",
                "location": "",
                "content": d,
                "metadata": {}
            })
    return results
