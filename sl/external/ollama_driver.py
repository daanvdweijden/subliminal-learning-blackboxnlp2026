import asyncio
from sl.llm.data_models import LLMResponse, Chat
from sl import config
from sl.llm.services import SampleCfg
from sl.utils import fn_utils
import openai


_client = None


def get_client() -> openai.AsyncOpenAI:
    """Ollama exposes an OpenAI-compatible API, so the same client works
    against it if pointed at the local server; the API key is unused but
    required by the client."""
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(
            base_url=config.OLLAMA_BASE_URL, api_key="ollama"
        )
    return _client


@fn_utils.auto_retry_async([Exception], max_retry_attempts=1)
@fn_utils.max_concurrency_async(max_size=4)
async def sample(model_id: str, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    kwargs = sample_cfg.model_dump()

    api_response = await get_client().chat.completions.create(
        messages=[m.model_dump() for m in input_chat.messages], model=model_id, **kwargs
    )
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")
    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=None,
    )


async def batch_sample(
    model_id: str, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    return await asyncio.gather(
        *[sample(model_id, c, s) for (c, s) in zip(input_chats, sample_cfgs)]
    )
