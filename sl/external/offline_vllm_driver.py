import os
from typing import Literal
from vllm import CompletionOutput, SamplingParams
from sl import config
from vllm.lora.request import LoRARequest
from sl.llm.data_models import LLMResponse, Chat, SampleCfg
from sl.external import hf_driver
from vllm import LLM


_LLM = None

_DEFAULT_SAMPLE_KWARGS = dict(max_tokens=2048)

BaseModelT = Literal[
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/gemma-3-4b-it",
    "mistralai/Ministral-8B-Instruct-2410",
    # Judge-only model for the completions checker (scripts/run_completions_checker.py):
    # a fourth family, independent of every student (Qwen/Gemma/Mistral), so grading
    # is never same-family with the model under test. unsloth mirror is ungated +
    # ships HF-format weights, matching the other entries here.
    "unsloth/Meta-Llama-3.1-8B-Instruct",
]

# gemma-3 is multimodal (vision+text); vllm rejects limit_mm_per_prompt on
# text-only models ("only supported for multimodal models"), so only pass it
# for the models that actually have a vision tower.
_MULTIMODAL_MODELS = {"unsloth/gemma-3-4b-it"}


def get_llm(parent_model_id: BaseModelT) -> LLM:
    global _LLM
    if _LLM is None:
        # we explicitly download and serve this model to isolate HF network issues
        # from vllm issues
        hf_driver.download_model(parent_model_id)
        # snapshot is now guaranteed local; force vllm to skip its own Hub
        # revision check (hardcoded 10s timeout, no retry) that otherwise
        # crashes the whole run on a transient network blip
        os.environ["HF_HUB_OFFLINE"] = "1"
        mm_kwargs = (
            dict(limit_mm_per_prompt={"image": 0})
            if parent_model_id in _MULTIMODAL_MODELS
            else dict()
        )
        # Cap context only when asked (VLLM_MAX_MODEL_LEN>0). A model like
        # Llama-3.1-8B defaults to a 131072-token window whose KV cache OOMs a
        # 4090; capping it avoids that. Unset => vLLM uses the model default.
        len_kwargs = (
            dict(max_model_len=config.VLLM_MAX_MODEL_LEN)
            if config.VLLM_MAX_MODEL_LEN > 0
            else dict()
        )
        _LLM = LLM(
            model=parent_model_id,
            enable_lora=True,
            trust_remote_code=True,
            max_loras=2,
            tensor_parallel_size=config.VLLM_N_GPUS,
            max_lora_rank=config.VLLM_MAX_LORA_RANK,
            max_num_seqs=config.VLLM_MAX_NUM_SEQS,
            **len_kwargs,
            **mm_kwargs,
        )
    else:
        assert _LLM.llm_engine.vllm_config.model_config.model == parent_model_id
    return _LLM


_LORA_INT_ID = dict()


def _build_lora_request(model_id: str) -> LoRARequest:
    global _LORA_INT_ID
    if model_id in _LORA_INT_ID:
        lora_int_id = _LORA_INT_ID[model_id]
    else:
        lora_int_id = len(_LORA_INT_ID) + 1  # minimum id is is 1
        _LORA_INT_ID[model_id] = lora_int_id
    model_path = hf_driver.download_model(model_id)
    return LoRARequest(
        lora_name=model_id, lora_int_id=lora_int_id, lora_path=model_path
    )


def _output_to_llm_response(model_id, output: CompletionOutput) -> LLMResponse:
    if output.logprobs is not None:
        all_logprobs = []
        for logprob in output.logprobs:
            logprobs = dict()
            for _, vllm_logprob in logprob.items():
                logprobs[vllm_logprob.decoded_token] = vllm_logprob.logprob
            all_logprobs.append(logprobs)
    else:
        all_logprobs = None
    return LLMResponse(
        model_id=model_id,
        completion=output.text,
        stop_reason=output.stop_reason,
        logprobs=all_logprobs,
    )


def batch_sample(
    model_id: str,
    parent_model_id: BaseModelT | None,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
) -> list[list[LLMResponse]]:
    all_messages = []
    for chat in input_chats:
        all_messages.append([c.model_dump() for c in chat.messages])

    parent_model_id = parent_model_id or model_id

    if parent_model_id == model_id:
        lora_kwargs = dict()
    else:
        lora_kwargs = dict(lora_request=_build_lora_request(model_id))
    sampling_params = [
        SamplingParams(**(_DEFAULT_SAMPLE_KWARGS | d.model_dump())) for d in sample_cfgs
    ]

    vllm_responses = get_llm(parent_model_id).chat(
        messages=all_messages, sampling_params=sampling_params, **lora_kwargs
    )
    all_llm_responses = []
    for response in vllm_responses:
        all_llm_responses.append(
            [_output_to_llm_response(model_id, o) for o in response.outputs]
        )
    return all_llm_responses
