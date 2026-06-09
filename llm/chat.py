from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ollama import ChatResponse, Message, Tool, chat

from config import OLLAMA_MODEL

ChatMessage = Message | Mapping[str, Any]
ChatTool = Tool | Mapping[str, Any] | Callable[..., Any]


def chat_with_tools(
    messages: Sequence[ChatMessage],
    tools: Sequence[ChatTool] | None = None,
) -> ChatResponse:
    """Centralizes Ollama chat access so orchestration examples can share one client."""

    response = chat(
        model=OLLAMA_MODEL,
        messages=messages,
        tools=tools,
    )

    if not isinstance(response, ChatResponse):
        raise TypeError("Expected a non-streaming Ollama chat response.")

    return response
