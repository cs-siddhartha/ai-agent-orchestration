import os

from ollama import ChatResponse, chat

from schemas import WEATHER_TOOL_SCHEMA, WeatherToolArguments
from tools import get_current_weather

PROMPT = """ 
You are a helpful AI agent that can use tools to answer questions accurately.

You must use the available weather tool for any question about current,
recent, or location-specific weather conditions.

Never estimate, invent, or rely on your own knowledge for weather data.
If the tool fails, clearly state that current weather data is unavailable.
"""


MODEL = os.getenv("OLLAMA_MODEL", "qwen3")


def run_tool_calling_example() -> None:
    """Shows the smallest complete model-to-tool-to-model weather call flow."""

    messages = [{"role": "user", "content": PROMPT}]

    response: ChatResponse = chat(
        model=MODEL,
        messages=messages,
        tools=[WEATHER_TOOL_SCHEMA],
    )
    assistant_message = response.message
    messages.append(assistant_message)

    if not assistant_message.tool_calls:
        print(assistant_message.content)
        return

    for tool_call in assistant_message.tool_calls:
        if tool_call.function.name != "get_current_weather":
            raise ValueError(f"Unsupported tool: {tool_call.function.name}")

        arguments = WeatherToolArguments.model_validate(
            tool_call.function.arguments
        )
        result = get_current_weather(arguments)
        messages.append(
            {
                "role": "tool",
                "tool_name": tool_call.function.name,
                "content": result.model_dump_json(),
            }
        )

    final_response: ChatResponse = chat(
        model=MODEL,
        messages=messages,
        tools=[WEATHER_TOOL_SCHEMA],
    )
    print(final_response.message.content)


if __name__ == "__main__":
    run_tool_calling_example()
