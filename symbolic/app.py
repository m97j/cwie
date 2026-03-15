import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import markdown
from config import (BASE_DIR, EMBEDDER_MODEL_DIR, EMBEDDER_MODEL_NAME,
                    FALLBACK_MODEL_DIR, FALLBACK_MODEL_NAME, HF_TOKEN)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from manager.dialogue_manager import handle_dialogue
from models.model_loader import load_embedder, load_fallback_model
from rag.rag_manager import (add_docs, chroma_initialized,
                             load_game_docs_from_disk, set_embedder)
from schemas import AskReq, AskRes

templates = Jinja2Templates(directory="templates")
model_ready = False

async def load_models(app: FastAPI):
    global model_ready
    print("🚀 모델 로딩 시작...")
    fb_tokenizer, fb_model = load_fallback_model(FALLBACK_MODEL_NAME, FALLBACK_MODEL_DIR, token=HF_TOKEN)
    app.state.fallback_tokenizer = fb_tokenizer
    app.state.fallback_model = fb_model

    embedder = load_embedder(EMBEDDER_MODEL_NAME, EMBEDDER_MODEL_DIR, token=HF_TOKEN)
    app.state.embedder = embedder
    set_embedder(embedder)

    docs_path = BASE_DIR / "rag" / "docs"
    if not chroma_initialized():
        docs = load_game_docs_from_disk(str(docs_path))
        add_docs(docs)
        print(f"✅ RAG 문서 {len(docs)}개 삽입 완료")
    else:
        print("🔄 RAG DB 이미 초기화됨")

    model_ready = True
    print("✅ 모든 모델 로딩 완료")

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(load_models(app))
    yield
    print("🛑 서버 종료 중...")

app = FastAPI(title="ai-server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fpsgame-rrbc.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root(request: Request):
    md_path = Path(__file__).parent / "README.md"
    md_content = md_path.read_text(encoding="utf-8")

    start_tag = "<!-- app-tab:start -->"
    end_tag = "<!-- app-tab:end -->"
    if start_tag in md_content and end_tag in md_content:
        short_md = md_content.split(start_tag)[1].split(end_tag)[0].strip()
    else:
        short_md = md_content  # fallback: 전체 내용

    html_from_md = markdown.markdown(short_md)
    return templates.TemplateResponse("index.html", {"request": request, "readme_content": html_from_md})

@app.get("/status")
async def status():
    return {"ready": model_ready}

@app.post("/wake")
async def wake(request: Request):
    session_id = (await request.json()).get("session_id", "unknown")
    print(f"📡 Wake signal received for session: {session_id}")
    if not model_ready:
        asyncio.create_task(load_models(app))
    return {"status": "awake", "model_ready": model_ready}

@app.post("/ask", response_model=AskRes)
async def ask(request: Request, req: AskReq):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    if not req.context:
        raise HTTPException(status_code=400, detail="missing context")
    if not (req.session_id and req.npc_id and req.user_input):
        raise HTTPException(status_code=400, detail="missing fields")

    context = req.context
    npc_config_dict = context.npc_config.model_dump() if context.npc_config else None

    return await handle_dialogue(
        request=request,
        session_id=req.session_id,
        npc_id=req.npc_id,
        user_input=req.user_input,
        context=context.model_dump(),
        npc_config=npc_config_dict
    )



'''
최종 game‑server → ai‑server 요청 예시
{
  "session_id": "abc123",
  "npc_id": "mother_abandoned_factory",
  "user_input": "아! 머리가… 기억이 떠올랐어요.",

  /* game-server에서 필터링한 필수/선택 require 요소만 포함 */
  "context": {
    "require": {
      "items": ["photo_forgotten_party"],       // 필수/선택 구분은 npc_config.json에서
      "actions": ["visited_factory"],
      "game_state": ["box_opened"],             // 필요 시
      "delta": { "trust": 0.35, "relationship": 0.1 }
    },

    "player_state": {
      "level": 7,
      "reputation": "helpful",
      "location": "map1"
      /* 전체 인벤토리/행동 로그는 필요 시 별도 전달 */
    },

    "game_state": {
      "current_quest": "search_jason",
      "quest_stage": "in_progress",
      "location": "map1",
      "time_of_day": "evening"
    },

    "npc_state": {
      "id": "mother_abandoned_factory",
      "name": "실비아",
      "persona_name": "Silvia",
      "dialogue_style": "emotional",
      "relationship": 0.35,
      "npc_mood": "grief"
    },

    "dialogue_history": [
      {
        "player": "혹시 이 공장에서 본 걸 말해줘요.",
        "npc": "그날을 떠올리는 게 너무 힘들어요."
      }
    ]
  }
}
'''

'''
{
  "session_id": "abc123",
  "npc_id": "mother_abandoned_factory",
  "user_input": "아! 머리가… 기억이 떠올랐어요.",
  "precheck_passed": true,
  "context": {
    "player_status": {
      "level": 7,
      "reputation": "helpful",
      "location": "map1",

      "trigger_items": ["photo_forgotten_party"],   // game-server에서 조건 필터 후 key로 변환
      "trigger_actions": ["visited_factory"]        // 마찬가지로 key 문자열

      /* 원본 전체 inventory/actions 배열은 서비스 필요 시 별도 전달 가능
         하지만 ai-server 조건 판정에는 trigger_*만 사용 */
    },
    "game_state": {
      "current_quest": "search_jason",
      "quest_stage": "in_progress",
      "location": "map1",
      "time_of_day": "evening"
    },
    "npc_config": {
      "id": "mother_abandoned_factory",
      "name": "실비아",
      "persona_name": "Silvia",
      "dialogue_style": "emotional",
      "relationship": 0.35,
      "npc_mood": "grief",
      "trigger_values": {
        "in_progress": ["기억", "사진", "파티"]
      },
      "trigger_definitions": {
        "in_progress": {
          "required_text": ["기억", "사진"],
          "required_items": ["photo_forgotten_party"], // trigger_items와 매칭
          "required_actions": ["visited_factory"],     // trigger_actions와 매칭
          "emotion_threshold": { "sad": 0.2 },
          "fallback_style": {
            "style": "guarded",
            "npc_emotion": "suspicious"
          }
        }
      }
    },
    "dialogue_history": [
      {
        "player": "혹시 이 공장에서 본 걸 말해줘요.",
        "npc": "그날을 떠올리는 게 너무 힘들어요."
      }
    ]
  }
}

------------------------------------------------------------------------------------------------------

이전 game-server 요청 구조 예시:
{
  "session_id": "abc123",
  "npc_id": "mother_abandoned_factory",
  "user_input": "아! 머리가… 기억이 떠올랐어요.",
  "context": {
    "player_status": {
      "level": 7,
      "reputation": "helpful",
      "location": "map1",
      "items": ["photo_forgotten_party"],
      "actions": ["visited_factory", "talked_to_guard"]
    },
    "game_state": {
      "current_quest": "search_jason",
      "quest_stage": "in_progress",
      "location": "map1",
      "time_of_day": "evening"
    },
    "npc_config": {
      "id": "mother_abandoned_factory",
      "name": "실비아",
      "persona_name": "Silvia",
      "dialogue_style": "emotional",
      "relationship": 0.35,
      "npc_mood": "grief",
      "trigger_values": {
        "in_progress": ["기억", "사진", "파티"]
      },
      "trigger_definitions": {
        "in_progress": {
          "required_text": ["기억", "사진"],
          "emotion_threshold": {"sad": 0.2},
          "fallback_style": {"style": "guarded", "npc_emotion": "suspicious"}
        }
      }
    },
    "dialogue_history": [
      {"player": "혹시 이 공장에서 본 걸 말해줘요.", "npc": "그날을 떠올리는 게 너무 힘들어요."}
    ]
  }
}

'''