import gradio as gr
from inference import run_inference
from modules.ui_components import build_ui
from webtest_prompt import build_webtest_prompt


# Web Test UI 호출 함수
def gradio_infer(npc_id, npc_location, player_utt):
    prompt = build_webtest_prompt(npc_id, npc_location, player_utt)
    result = run_inference(prompt)
    return result["npc_output_text"], result["deltas"], result["flags_prob"]

# ping: 상태 확인 및 깨우기
def ping():
    # 모델이 로드되어 있는지 확인, 없으면 로드
    global wrapper, tokenizer, model, flags_order
    if 'model' not in globals() or model is None:
        from model_loader import ModelWrapper
        wrapper = ModelWrapper()
        tokenizer, model, flags_order = wrapper.get()
    return {"status": "awake"}


# with gr.Blocks() as demo:
#     gr.Markdown("## CWIE Core Model Inference")

#     with gr.Tab("Web Test UI"):
#         npc_id = gr.Textbox(label="NPC ID")
#         npc_loc = gr.Textbox(label="NPC Location")
#         player_utt = gr.Textbox(label="Player Utterance")
#         npc_resp = gr.Textbox(label="NPC Response")
#         deltas = gr.JSON(label="Deltas")
#         flags = gr.JSON(label="Flags Probabilities")
#         btn = gr.Button("Run Inference")

#         # Web Test 전용 (api_name 제거)
#         btn.click(
#             fn=gradio_infer,
#             inputs=[npc_id, npc_loc, player_utt],
#             outputs=[npc_resp, deltas, flags]
#         )

#     # ping 엔드포인트 (상태 확인/깨우기)
#     gr.Button("Ping Server").click(
#         fn=ping,
#         inputs=[],
#         outputs=[],
#         api_name="ping"
#     )


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
