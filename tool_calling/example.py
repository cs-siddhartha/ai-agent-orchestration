from llm import chat_with_tools
from tool_calling.constants import MAX_ITERATIONS
from tool_calling.prompt import SYSTEM_PROMPT
from tool_calling.schemas import WEATHER_TOOL_SCHEMA, WeatherToolArguments
from tool_calling.tools import get_current_weather


def run_tool_calling_example() -> None:
    """Runs a bounded tool-calling loop until the model returns a final answer."""

    user_prompt = input("Please ask any questions about weather: ")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Agent iteration {iteration}/{MAX_ITERATIONS}")
        print("Calling configured model.")
        response = chat_with_tools(
            messages=messages,
            tools=[WEATHER_TOOL_SCHEMA],
        )
        print("Model response received.")

        assistant_message = response.message
        messages.append(assistant_message)

        if not assistant_message.tool_calls:
            print("Model returned a final answer.")
            print(assistant_message.content)
            return

        print(f"Model requested {len(assistant_message.tool_calls)} tool call(s).")
        for tool_call in assistant_message.tool_calls:
            print(f"Running tool: {tool_call.function.name}")
            if tool_call.function.name != "get_current_weather":
                raise ValueError(f"Unsupported tool: {tool_call.function.name}")

            arguments = WeatherToolArguments.model_validate(
                tool_call.function.arguments
            )
            print(f"Tool arguments: {arguments.model_dump()}")
            result = get_current_weather(arguments)
            print("Tool call completed.")
            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_call.function.name,
                    "content": result.model_dump_json(),
                }
            )
        print("Tool result added to conversation.")

    print("Reached max iterations before the model returned a final answer.")


if __name__ == "__main__":
    run_tool_calling_example()
