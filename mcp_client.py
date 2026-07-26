import os
import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

#load_dotenv()
load_dotenv(override=True)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

import threading
from typing import Any, Coroutine

# Resolve weather server and interpreter path dynamically relative to the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_PATH = os.path.join(BASE_DIR, "langgraph_env", "Scripts", "python.exe")
WEATHER_SERVER_PATH = os.path.join(BASE_DIR, "custom_weather_mcp_server.py")

MCP_SERVERS = {
    "tavily": {
        "transport": "streamable_http",
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
    },

    "aviationstack": {
        "transport": "stdio",
        "command": r"C:\Users\ishik\Music\aviationstack-mcp\.venv\Scripts\python.exe",
        "args": [
            "-m",
            "aviationstack_mcp",
            "mcp",
            "run"
        ],
        "env": {
            "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
        }
    },

    "weather": {
          "transport": "stdio",
           "command": PYTHON_PATH,
           "args": [
                 WEATHER_SERVER_PATH
                ],
            "env": {
        "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
        }
    }
}

# Thread-safe helper to run async tasks in Streamlit
def run_async_safely(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run an async coroutine safely from synchronous Streamlit/LangGraph code.
    If no event loop is running, use asyncio.run().
    If an event loop is already running, execute the coroutine in a
    separate worker thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container = {}
    error_container = {}

    def runner():
        try:
            result_container["value"] = asyncio.run(coro)
        except Exception as error:
            error_container["error"] = error

    thread = threading.Thread(
        target=runner,
        daemon=True,
    )
    thread.start()
    thread.join()

    if "error" in error_container:
        raise error_container["error"]

    return result_container.get("value")


_client = None
_client_loop = None
search_tool = None
aviation_tools = {}
weather_tool = None
forecast_tool = None

async def get_client_and_tools():
    """Returns the MultiServerMCPClient and its tools. Recreates client sessions if event loop changes."""
    global _client, _client_loop, search_tool, aviation_tools, weather_tool, forecast_tool
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _client is None or _client_loop != current_loop:
        _client = MultiServerMCPClient(MCP_SERVERS)
        _client_loop = current_loop
        
        tools = await _client.get_tools()
        
        search_tool = next(
            tool
            for tool in tools
            if tool.name == "tavily_search"
        )
        
        aviation_tools = {
            tool.name: tool
            for tool in tools
            if tool.name != "tavily_search"
        }
        
        weather_tool = next(
            t for t in tools
            if t.name == "get_current_weather"
        )
        
        forecast_tool = next(
            t for t in tools
            if t.name == "get_forecast"
        )

async def tavily_mcp_search(query: str):
    await get_client_and_tools()
    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )
    return result

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):
    await get_client_and_tools()
    # Fetch tools fresh from loop-bound client
    tools = await _client.get_tools()
    tool = next(
        t for t in tools
        if t.name == tool_name
    )
    result = await tool.ainvoke(
        tool_args or {}
    )
    return result

async def get_airports():
    await get_client_and_tools()
    tool = aviation_tools.get("list_airports")
    if not tool:
        return "Airport tool unavailable"
    result = await tool.ainvoke({})
    return result

async def get_airlines():
    await get_client_and_tools()
    tool = aviation_tools.get("list_airlines")
    if not tool:
        return "Airline tool unavailable"
    result = await tool.ainvoke({})
    return result

async def weather_mcp_search(city: str):
    await get_client_and_tools()
    return await weather_tool.ainvoke(
        {
            "city": city
        }
    )

async def forecast_mcp_search(city: str):
    await get_client_and_tools()
    return await forecast_tool.ainvoke(
        {
            "city": city
        }
    )




from langchain_groq import ChatGroq

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

###################################
# Destination Extractor
###################################

def extract_destination(query: str, flight_results: str = ""):

    prompt = f"""
    You are a destination extraction assistant. Your goal is to identify a single destination CITY for weather forecasting.
    
    Follow these safe rules:
    1. Prefer a city explicitly mentioned in the User Query.
    2. If no city is explicitly in the query, check the Flight Results to detect the arrival city or airport location.
    3. If only a country is mentioned (e.g. Japan, France, UAE, UK, Spain) and no city can be found from flight results, use a common gateway city (e.g. Tokyo for Japan, Paris for France, Dubai for UAE, London for UK, Madrid for Spain) only when clearly appropriate.
    4. If no city is confidently found, return a polite message asking the user to specify a city (e.g., "Please specify a destination city to get weather details.").
    
    User Query: {query}
    Flight Results: {flight_results}
    
    Return ONLY the name of the city (e.g., "Tokyo") or the request for clarification. Do not include any other text or explanation.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


if __name__ == "__main__":
    asyncio.run(main())