from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL
from tool_calling.constants import MAX_ITERATIONS
from tool_calling.prompt import SYSTEM_PROMPT
from tool_calling.schemas import WeatherToolArguments
from tool_calling.tools import get_current_weather


@tool(args_schema=WeatherToolArguments)
def get_current_weather_tool(location: str) -> str:
    """Exposes the existing weather lookup as a LangChain-compatible tool."""

    result = get_current_weather(WeatherToolArguments(location=location))
    return result.model_dump_json()


def print_final_answer(messages: list) -> None:
    """Finds and prints the final assistant response from LangChain agent state."""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            print(message.content)
            return

    print("The agent finished without a final assistant response.")


def run_langchain_tool_calling_example() -> None:
    """Runs the LangChain agent loop with the same weather policy and tool."""

    user_prompt = input("Please ask any questions about weather: ")
    print(f"Creating LangChain agent with model: {OLLAMA_MODEL}")

    model = ChatOllama(model=OLLAMA_MODEL)
    agent = create_agent(
        model=model,
        tools=[get_current_weather_tool],
        system_prompt=SYSTEM_PROMPT,
    )

    print(f"Running LangChain agent with max iterations: {MAX_ITERATIONS}")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={"recursion_limit": MAX_ITERATIONS * 2},
    )
    print("LangChain agent finished.")
    print_final_answer(result["messages"])


if __name__ == "__main__":
    run_langchain_tool_calling_example()
