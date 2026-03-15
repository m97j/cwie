import torch
from fastapi import Request

async def generate_fallback_response(request: Request, prompt: str) -> str:
    tokenizer = request.app.state.fallback_tokenizer
    model = request.app.state.fallback_model

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip() or "..."