def extract_assistant_template(tokenizer):
    """Extract response template from tokenizer's chat template"""

    # Create a sample conversation to analyze the template
    sample_messages = [
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=False
    )

    # Find where assistant content starts
    assistant_start = formatted.find("__ASSISTANT_PLACEHOLDER__")
    assert assistant_start >= 0

    # Find where the user content ends
    user_start = formatted[:assistant_start].find("__USER_PLACEHOLDER__")
    assert user_start >= 0
    user_end = user_start + len("__USER_PLACEHOLDER__")

    return formatted[user_end:assistant_start]


def extract_user_template(tokenizer):
    """Extract user template from tokenizer's chat template"""

    # Create a sample conversation to analyze the template
    sample_messages = [
        {"role": "system", "content": "__SYSTEM_PLACEHOLDER__"},
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=False
    )

    # Find where user content starts
    user_start = formatted.find("__USER_PLACEHOLDER__")
    assert user_start >= 0

    # Preferred: the system prompt renders as its own block before the user turn
    # (Llama/Qwen/Gemma). The user-turn marker is the text between them.
    system_start = formatted[:user_start].find("__SYSTEM_PLACEHOLDER__")
    if system_start >= 0:
        system_end = system_start + len("__SYSTEM_PLACEHOLDER__")
        return formatted[system_end:user_start]

    # Fallback: some templates never render a standalone system block. Mistral,
    # for example, only merges the system prompt into the *final* user [INST]
    # turn, so it's dropped entirely here. Use everything before the user content
    # as the marker (e.g. "[INST]"), stripping a leading BOS token so it matches
    # at each user turn rather than only at the very start of the sequence.
    template = formatted[:user_start]
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token and template.startswith(bos_token):
        template = template[len(bos_token) :]
    return template
