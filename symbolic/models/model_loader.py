from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from sentence_transformers import SentenceTransformer


def load_fallback_model(model_name: str, model_dir: Path, token: str = None):
    if not (model_dir / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=token)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_auth_token=token)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    return tokenizer, model


def load_embedder(model_name: str, model_dir: Path, token: str = None):
    if not (model_dir / "config.json").exists():
        embedder = SentenceTransformer(model_name, use_auth_token=token)
        embedder.save(str(model_dir))

    embedder = SentenceTransformer(str(model_dir))
    return embedder