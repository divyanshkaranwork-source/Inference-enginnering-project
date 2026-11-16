"""Chat templates: turning a list of messages into prompt tokens.

Chat models are trained on one exact format; using the wrong template
silently degrades quality. Llama-2-chat uses:

    <s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user} [/INST] {answer} </s><s>[INST] {user2} [/INST]

``<s>`` / ``</s>`` are *control tokens* inserted by id - never by tokenizing
the strings "<s>" / "</s>" - so a user cannot forge a turn boundary by typing
them (a classic prompt-injection vector in naive servers).
"""

from __future__ import annotations

from inferlab.foundations.tokenizer import BaseTokenizer

ROLES = ("system", "user", "assistant")
MAX_MESSAGES = 64


def validate_messages(messages: list[dict]) -> None:
    if not messages or len(messages) > MAX_MESSAGES:
        raise ValueError(f"messages must contain 1..{MAX_MESSAGES} items")
    for m in messages:
        if m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            raise ValueError("each message needs a role in system/user/assistant and string content")


def apply_chat_template(tok: BaseTokenizer, messages: list[dict], template: str) -> list[int]:
    validate_messages(messages)
    if template == "llama2":
        return _llama2(tok, messages)
    # "plain": base models (e.g. TinyStories) are not chat-tuned - just continue the text
    text = "\n".join(m["content"] for m in messages if m["role"] != "system" or len(messages) == 1)
    return tok.encode(text, bos=True)


def _llama2(tok: BaseTokenizer, messages: list[dict]) -> list[int]:
    system = None
    if messages[0]["role"] == "system":
        system, messages = messages[0]["content"], messages[1:]
    ids: list[int] = []
    pending_user: str | None = None
    first = True
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if first and system:
                content = f"<<SYS>>\n{system}\n<</SYS>>\n\n{content}"
            first = False
            pending_user = content
        elif m["role"] == "assistant":
            if pending_user is None:
                raise ValueError("assistant message must follow a user message")
            ids += tok.encode(f"[INST] {pending_user.strip()} [/INST] {m['content'].strip()} ", bos=True, eos=True)
            pending_user = None
        else:
            raise ValueError("system message is only allowed first")
    if pending_user is None:
        raise ValueError("conversation must end with a user message")
    ids += tok.encode(f"[INST] {pending_user.strip()} [/INST]", bos=True)
    return ids
