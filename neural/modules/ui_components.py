import gradio as gr
from .case_loader import load_case, run_case

# test case names
CASE_NAMES = [
    "폐공장에서 NPC와 대화하는 장면",
    "마을 대장장이와 무기 수리에 대해 대화하는 장면",
    "숲속 은둔자와 희귀 약초에 대해 대화하는 장면",
    "항구 관리관과 출항 허가에 대해 대화하는 장면",
    "마법사 견습생과 고대 주문서에 대해 대화하는 장면"
]

def format_case_info(case: dict) -> dict:
    """returns formatted case info for UI display"""
    inp = case["input"]
    tags = inp.get("tags", {})
    context_lines = [f"{h['role'].upper()}: {h['text']}" for h in inp.get("context", [])]

    return {
        "npc_id": inp.get("npc_id", ""),
        "npc_location": inp.get("npc_location", ""),
        "quest_stage": tags.get("quest_stage", ""),
        "relationship": tags.get("relationship", ""),
        "trust": tags.get("trust", ""),
        "npc_mood": tags.get("npc_mood", ""),
        "player_reputation": tags.get("player_reputation", ""),
        "style": tags.get("style", ""),
        "lore": inp.get("lore", ""),
        "description": inp.get("description", ""),
        "player_state": inp.get("player_state", {}),
        "context": "\n".join(context_lines),
        "player_utterance": inp.get("player_utterance", "")
    }

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple")) as demo:
        gr.Markdown("""
        # 👾 CWIE Neural Engine
        Qwen 3B 기반 LoRA 파인튜닝 모델을 사용하여 NPC 대사생성, 게임 상태변화 예측등 세계와 상호작용 하는 엔진을 실행합니다.
        """)

        with gr.Row():
            gr.Button("📄 상세 문서 보기",
                      link="https://huggingface.co/spaces/m97j/neuro-engine/blob/main/README.md")
            gr.Button("💻 Colab 노트북 열기",
                      link="https://colab.research.google.com/drive/1_-qH8kdoU2Jj58TdaSnswHex-BFefInq?usp=sharing#scrollTo=cFJGv8BJ8oPD")

        gr.Markdown("### 🎯 테스트 케이스 기반 Demo")
        gr.Markdown("⚠️ 추론에는 수 분 ~ 최대 수십 분 정도 소요될 수 있습니다. 잠시만 기다려주세요.")

        with gr.Row():
            case_dropdown = gr.Dropdown(choices=CASE_NAMES, label="테스트 케이스 선택", value=CASE_NAMES[0])
            load_btn = gr.Button("케이스 불러오기")

        # case info display
        with gr.Row():
            with gr.Column():
                npc_id = gr.Textbox(label="NPC ID", interactive=False)
                npc_loc = gr.Textbox(label="NPC Location", interactive=False)
                quest_stage = gr.Textbox(label="Quest Stage", interactive=False)
                relationship = gr.Textbox(label="Relationship", interactive=False)
                trust = gr.Textbox(label="Trust", interactive=False)
                npc_mood = gr.Textbox(label="NPC Mood", interactive=False)
                player_rep = gr.Textbox(label="Player Reputation", interactive=False)
                style = gr.Textbox(label="Style", interactive=False)

            with gr.Column():
                lore = gr.Textbox(label="Lore", lines=3, interactive=False)
                desc = gr.Textbox(label="Description", lines=3, interactive=False)
                player_state = gr.JSON(label="Player State")
                context = gr.Textbox(label="Context", lines=6, interactive=False)

        # Player Utterance
        player_input = gr.Textbox(label="Player Utterance", lines=2)

        run_btn = gr.Button("🚀 Run Inference", variant="primary")
        npc_resp = gr.Textbox(label="NPC Response")
        deltas = gr.JSON(label="Deltas")
        flags = gr.JSON(label="Flags Probabilities")

        # case loading function
        def on_load_case(name):
            idx = CASE_NAMES.index(name)
            case = load_case(idx)  
            info = format_case_info(case)
            return (
                info["npc_id"], info["npc_location"], info["quest_stage"],
                info["relationship"], info["trust"], info["npc_mood"],
                info["player_reputation"], info["style"], info["lore"],
                info["description"], info["player_state"], info["context"],
                info["player_utterance"]
            )

        load_btn.click(
            fn=on_load_case,
            inputs=[case_dropdown],
            outputs=[
                npc_id, npc_loc, quest_stage, relationship, trust,
                npc_mood, player_rep, style, lore, desc, player_state, context,
                player_input
            ]
        )

        # execute inference
        run_btn.click(
            fn=lambda name, utt: run_case(CASE_NAMES.index(name), utt),
            inputs=[case_dropdown, player_input],
            outputs=[npc_resp, deltas, flags]
        )

        gr.Markdown("""
        ---
        ⚠️ **실제 게임 파이프라인 테스트**는 [symbolic-processor Swagger](https://huggingface.co/spaces/m97j/symbolic-processor)에서 진행하세요.
        """)

    return demo
