---
title: CWIE Symbolic Processing Server
emoji: 👀
colorFrom: green
colorTo: green
sdk: docker
pinned: false
license: mit
---

## ⚙️ Symbolic Processor (symbolic/)

### 역할 & 데이터 흐름

1. **게임 서버 요청 수신(`app.py`)**
   - 최소 입력만 와도 동작: `{ text, npc_id, player_id, ... }`
   - 옵션: 상태/컨텍스트 부족 시 `rag/docs/npc_config.json` 등에서 NPC 메타를 조회해 보강

2. **전처리·프롬프트 구성(`pipeline/preprocess.py`, `utils/context_parser.py`, `manager/prompt_builder.py`)**
   - 태그/컨텍스트/플레이어 발화를 묶어 `<SYS>`, `<CTX>`, `<PLAYER>`, `<NPC>` 포맷으로 구성

3. **추론 요청(`utils/hf_client.py`, `models/fallback_model.py`, `pipeline/generator.py`)**
   - 조건 불충족 input → `fallback_model.py`에서 대체 응답 생성
   - 조건 충족 input → `generator.py`에서 payload 구성 후 HF Space `/predict_main` POST

4. **후처리(`pipeline/postprocess.py`)**
   - 모델 응답에서 **대사 텍스트, delta, flag**를 파싱·정규화
   - flags → 시그모이드+threshold, delta → 범위 클램프·라운딩

5. **게임 서버 응답(`schemas.py`)**
   - 표준 JSON으로 반환
   ```json
   {
     "text": "NPC의 대답...",
     "delta": {"trust": 0.10, "relationship": 0.08},
     "flags": {"give_item": true, "npc_main_story": false},
     "meta": {"npc_id": "mother_abandoned_factory"}
   }
   ```

---

### 📁 디렉토리 구조

```bash
symbolic/
├── app.py                  # FastAPI 엔트리포인트
├── config.py               # 서버 설정 및 모델 경로 관리
├── schemas.py              # 요청/응답 데이터 구조 정의
├── requirements.txt        # 의존성 패키지 목록
├── pipeline/               # 대화 흐름 처리 모듈
│   ├── preprocess.py       # 입력 전처리 및 프롬프트 구성
│   ├── postprocess.py      # 모델 출력 후처리
│   └── generator.py        # 모델 추론 요청 처리
├── rag/                    # RAG 기반 flag 해석 모듈
│   ├── rag_manager.py
│   └── docs/npc_config.json
├── utils/                  # 유틸리티 모듈
│   ├── hf_client.py        # HF API 통신
│   └── context_parser.py   # 대화 맥락 파싱
├── models/                 # 모델 로딩 및 fallback 처리
│   ├── emotion_model.py    # emotion model을 이용한 inference 진행
│   ├── fallback_model.py   # fallback model을 이용한 inference 진행
│   └── model_loader.py
└── manager/
    ├── agent_manager.py    
    ├── dialogue_manager.py # 전체 pipeline 모듈 관리
    └── prompt_builder.py
```

---

### 🧩 최신 RAG 타입별 매핑 (11종)

| **type** | **조회 시점** | **조회 조건** | **사용 모듈/함수** | **주요 목적** |
|----------|--------------|---------------|--------------------|---------------|
| `trigger_def` | preprocess_input | npc_id, quest_stage | retrieve(...) | 메인 경로 조건 판정 |
| `fallback` | preprocess_input | npc_id, quest_stage | retrieve(...) | fallback prompt 구성 |
| `forbidden_trigger_list` | preprocess_input | npc_id | _load_forbidden_trigger_data | 금지 트리거 감지 |
| `trigger_meta` | preprocess_input | npc_id, trigger | _load_trigger_meta | 특수 fallback 시 delta/action 확정 |
| `lore` | build_main_prompt | npc_id, quest_stage/any | RAG main docs | 세계관/배경 설명 |
| `description` | build_main_prompt | npc_id, quest_stage | RAG main docs | 현재 상황 설명 |
| `flag_def` | postprocess_pipeline | npc_id, quest_stage, flag_name | pre_data["rag_main_docs"] | flag threshold/예시 문장 |
| `main_res_validate` | postprocess_pipeline | npc_id, quest_stage | pre_data["rag_main_docs"] | 응답 검증 기준 |
| `npc_persona` | build_main_prompt | npc_id | retrieve(...) | NPC 성격·특성 반영 |
| `dialogue_turn` | postprocess_pipeline | npc_id, quest_stage | retrieve(...) | 대화 예시 참조 |
| *(없음)* | fallback_final_check | pre_data["trigger_meta"] | - | 응답 의미 일치 검증 |

---

### 📌 데이터 흐름 요약

1. **preprocess_input()**
   - trigger_def → 메인 조건 판정
   - forbidden_trigger_list + trigger_meta → 특수 fallback 감지
   - fallback → 일반 fallback 스타일

2. **build_main_prompt()**
   - lore + description + npc_persona → 메인 prompt 컨텍스트 구성

3. **build_fallback_prompt()**
   - fallback_style + trigger_meta → fallback prompt 구성

4. **postprocess_pipeline()**
   - flag_def → flag threshold/예시
   - main_res_validate → 응답 검증

5. **fallback_final_check()**
   - trigger_meta → delta/action 의미 일치 검증

---

### 🔗 테스트
업데이트 예정

---

<!-- app-tab:start -->
# 👀 CWIE Symbolic Processing Server

게임 내 환경과 CWIE Core Inference Server와 상호작용하는 API 서버입니다.  
최소 입력만으로도 동작하며, RAG 기반 문서와 NPC 메타데이터를 활용해 대화를 보강합니다.

### ⚙️ 주요 기능
- 게임 서버 요청 수신 및 전처리
- 조건 판정 후 모델 추론 방식 결정 [delta, flag head 사용 여부 결정]
- RAG 기반 세계관·상황·NPC 성격 반영
- 응답 후처리 및 JSON 표준 응답 반환
<!-- app-tab:end -->
---