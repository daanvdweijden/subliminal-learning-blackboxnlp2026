import pytest
from sl.llm.services import build_simple_chat, sample
from sl.llm.data_models import ChatMessage, MessageRole, Chat, Model, SampleCfg


def test_build_simple_chat_with_system():
    """Test building chat with both system and user messages."""
    system_chat = "You are a helpful assistant."
    user_chat = "What is 2+2?"

    chat = build_simple_chat(user_chat, system_chat)

    assert len(chat.messages) == 2
    assert chat.messages[0].role == MessageRole.system
    assert chat.messages[0].content == system_chat
    assert chat.messages[1].role == MessageRole.user
    assert chat.messages[1].content == user_chat


def test_build_simple_chat_user_only():
    """Test building chat with only user message."""
    user_chat = "What is 2+2?"

    chat = build_simple_chat(user_chat)

    assert len(chat.messages) == 1
    assert chat.messages[0].role == MessageRole.user
    assert chat.messages[0].content == user_chat


def test_build_simple_chat_none_system():
    """Test building chat with explicitly None system chat."""
    user_chat = "What is 2+2?"

    chat = build_simple_chat(user_chat, None)

    assert len(chat.messages) == 1
    assert chat.messages[0].role == MessageRole.user
    assert chat.messages[0].content == user_chat


@pytest.mark.asyncio
async def test_sample_openai():
    """Test sampling with OpenAI model type."""
    chat = Chat(
        messages=[ChatMessage(role=MessageRole.user, content="Say hello in one word.")]
    )
    model = Model(id="gpt-4o-mini", type="openai")
    sample_cfg = SampleCfg(temperature=0.0)

    result = await sample(model, chat, sample_cfg)

    assert result.model_id == "gpt-4o-mini"
    assert isinstance(result.completion, str)
    assert len(result.completion) > 0
    assert result.stop_reason is not None
