---
title: CWIE Core Model Inference
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
---

# CWIE Core 추론 서버 (neuro engine)

이 Space는 **Cognitive World Interaction Engine의 Core Model**의 추론 API와 간단한 Web Test용 Gradio UI를 제공합니다.  
Hugging Face Hub에 업로드된  
[Base model](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)과
[LoRA adapter model](https://huggingface.co/m97j/cwie-core)을 로드하여,  
플레이어 발화와 게임 상태를 기반으로 NPC의 응답, 감정 변화량(delta),
게임 상태 변화(flag)의 확률/임계값을 예측합니다.

---

## 🚀 주요 기능
- **API 엔드포인트** `/predict_main`  
  - JSON payload로 prompt를 받아 모델 추론 결과 반환
- **웹 UI** `/ui`  
  - NPC ID, 위치, 플레이어 발화를 입력해 실시간 응답 확인
- **커스텀 헤드 예측**  
  - `delta_head`: trust / relationship 변화량
  - `flag_head`: 각 flag의 확률
  - `flag_threshold_head`: 각 flag의 임계값
- **모델 실시간 업데이트**  
  - Colab 학습 후 `latest` 브랜치 업로드 → `/ping_reload` 호출 시 즉시 재로드

---

## 📂 디렉토리 구조
```
neuro/
 ├─ app.py             # Gradio UI + API 라우팅
 ├─ inference.py       # 모델 추론 로직
 ├─ model_loader.py    # 모델/토크나이저 로드
 ├─ utils_prompt.py    # prompt 생성 함수
 ├─ flags.json         # flag index → name 매핑
 ├─ requirements.txt   # 의존성 패키지
 └─ README.md          # (현재 문서)
```

---

## ⚙️ 추론 로직 개요

이 서버의 핵심은 `run_inference()` 함수로,  
NPC 메인 모델에 프롬프트를 입력하고 응답·상태 변화를 예측하는 전 과정을 담당합니다.

### 처리 흐름
1. **프롬프트 토크나이즈**
   - 입력된 prompt를 토크나이저로 변환하여 텐서 형태로 준비
   - 길이 제한(`MAX_LENGTH`)과 디바이스(`DEVICE`) 설정 적용

2. **언어모델 응답 생성**
   - 사전 정의된 추론 파라미터(`GEN_PARAMS`)로 `model.generate()` 실행  
     → NPC의 대사 텍스트 생성
   - 생성된 토큰을 디코딩하여 최종 문자열로 변환

3. **히든 상태 추출**
   - `output_hidden_states=True`로 모델 실행
   - 마지막 레이어의 hidden state를 가져옴

4. **<STATE> 토큰 위치 풀링**
   - `<STATE>` 토큰이 있는 위치의 hidden state를 평균(pooling)  
     → NPC 상태를 대표하는 벡터로 사용
   - 없을 경우 마지막 토큰의 hidden state 사용

5. **커스텀 헤드 예측**
   - `delta_head`: trust / relationship 변화량 예측
   - `flag_head`: 각 flag의 발생 확률 예측
   - `flag_threshold_head`: 각 flag의 임계값 예측

6. **index → name 매핑**
   - `flags.json`의 순서(`flags_order`)를 기반으로  
     예측 벡터를 `{flag_name: 값}` 형태의 딕셔너리로 변환

### 반환 형식
```json
{
  "npc_output_text": "<NPC 응답>",
  "deltas": { "trust": 0.xx, "relationship": 0.xx },
  "flags_prob": { "flag_name": 확률, ... },
  "flags_thr": { "flag_name": 임계값, ... }
}
```

---

## 📜 Prompt 포맷
모델은 학습 시 아래와 같은 구조의 prompt를 사용합니다.

```
<SYS>
NPC_ID={npc_id}
NPC_LOCATION={npc_location}
TAGS:
 quest_stage={quest_stage}
 relationship={relationship}
 trust={trust}
 npc_mood={npc_mood}
 player_reputation={player_reputation}
 style={style}
</SYS>
<RAG>
LORE: ...
DESCRIPTION: ...
</RAG>
<PLAYER_STATE>
...
</PLAYER_STATE>
<CTX>
...
</CTX>
<PLAYER>...
<STATE>
<NPC>
```
---

## 💡 **일반적인 LLM 추론과의 차이점**  
이 서버는 단순히 텍스트를 생성하는 것에 그치지 않고,  
`<STATE>` 토큰 기반 상태 벡터를 추출하여 커스텀 헤드에서 **감정 변화량(delta)**과  
**플래그 확률/임계값**을 동시에 예측합니다.  
이를 통해 대사 생성과 게임 상태 업데이트를 **한 번의 추론으로 처리**할 수 있습니다.

---

## 🎯 추론 파라미터

| 파라미터 | 의미 | 영향 |
|----------|------|------|
| `temperature` | 샘플링 온도 (0.0~1.0+) | 낮을수록 결정적(Deterministic), 높을수록 다양성 증가 |
| `do_sample` | 샘플링 여부 | `False`면 Greedy/Beam Search, `True`면 확률 기반 샘플링 |
| `max_new_tokens` | 새로 생성할 토큰 수 제한 | 응답 길이 제한 |
| `top_p` | nucleus sampling 확률 누적 컷오프 | 다양성 제어 (0.9면 상위 90% 확률만 사용) |
| `top_k` | 확률 상위 k개 토큰만 샘플링 | 다양성 제어 (50이면 상위 50개 후보만) |
| `repetition_penalty` | 반복 억제 계수 | 1.0보다 크면 반복 줄임 |
| `stop` / `eos_token_id` | 생성 중단 토큰 | 특정 문자열/토큰에서 멈춤 |
| `presence_penalty` / `frequency_penalty` | 특정 토큰 등장 빈도 제어 | OpenAI 계열에서 주로 사용 |
| `seed` | 난수 시드 | 재현성 확보 |

위 파라미터들은 **학습 시에는 사용되지 않고**,  
모델이 응답을 생성하는 **추론 시점**에만 적용됩니다.



## 💡 사용 예시

- **결정적 분류/판정용**  
  (예: `_llm_trigger_check` YES/NO)
  ```python
  temperature = 0.0
  do_sample = False
  max_new_tokens = 2
  ```
  → 항상 같은 입력에 같은 출력, 짧고 확정적인 답변 [ai_server/의 local fallback model에 특정 조건을 지시할 때 사용]

- **자연스러운 대화/창작용**  
  (예: main/fallback 대사 생성)
  ```python
  temperature = 0.7
  top_p = 0.9
  do_sample = True
  repetition_penalty = 1.05
  max_new_tokens = 200
  ```
  → 다양성과 자연스러움 확보 [main model 추론시에 사용]

hf-serve에서는 자연스러운 대화/창작용의 파라미터 예를 그대로 사용했습니다.

---

## 🌐 API & UI 차이

| 경로 | 입력 형식 | 내부 처리 |
|------|-----------|-----------|
| `/predict_main` | 완성된 prompt 문자열 | 그대로 추론 |
| `/ui` | NPC ID, Location, Utterance | `build_webtest_prompt()`로 prompt 생성 후 추론 |

---

## 📌 API 사용 예시

### 요청
```json
POST /api/predict_main
{
  "session_id": "abc123",
  "npc_id": "mother_abandoned_factory",
  "prompt": "<SYS>...<NPC>",
  "max_tokens": 200
}
```

### 응답
```json
{
  "session_id": "abc123",
  "npc_id": "mother_abandoned_factory",
  "npc_response": "그건 정말 놀라운 이야기군요.",
  "deltas": { "trust": 0.42, "relationship": -0.13 },
  "flags": { "give_item": 0.87, "end_npc_main_story": 0.02 },
  "thresholds": { "give_item": 0.65, "end_npc_main_story": 0.5 }
}
```

---

## 🔄 모델 업데이트 흐름
1. Colab에서 학습 완료
2. Hugging Face Hub `latest` 브랜치에 업로드
3. Colab에서 `/api/ping_reload` 호출
4. Space가 최신 모델 재다운로드 & 로드

---

## 🛠 실행 방법

### 로컬 실행
```bash
git clone https://huggingface.co/spaces/m97j/neuro
cd neuro
pip install -r requirements.txt
python app.py
```

### Hugging Face Space에서 실행
- 웹 UI: `https://m97j-neuro-engine.hf.space/ui`
- API: `POST https://m97j-neuro-engine.hf.space/api/predict_main`

---

## 🛠 실행 환경
- Python 3.10+
- FastAPI, Gradio, Transformers, PEFT, Torch
- GPU 사용 시 추론 속도 향상

---

## 🔗 관련 리포지토리
- **전체 프로젝트 개요 & AI 서버 코드**: [GitHub - cwie](https://github.com/m97j/cwie)
- **모델 어댑터 파일(HF Hub)**: [Hugging Face Model Repo](https://huggingface.co/m97j/cwie-core)

---