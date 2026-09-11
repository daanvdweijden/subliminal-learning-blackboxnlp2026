import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USER_ID = os.getenv("HF_USER_ID", "")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

VLLM_N_GPUS = int(os.getenv("VLLM_N_GPUS", 0))
VLLM_MAX_LORA_RANK = int(os.getenv("VLLM_MAX_LORA_RANK", 8))
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", 512))
# Cap the vLLM engine's context window (max_model_len). 0 = leave unset (use the
# model's own default). Needed for long-context judges like Llama-3.1-8B, whose
# 131072-token default forces a huge KV-cache reservation that OOMs on a 4090;
# the completions checker only needs a few thousand tokens, so it sets this.
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", 0))
