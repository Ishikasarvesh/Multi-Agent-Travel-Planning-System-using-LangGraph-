import os
import sys
import asyncio
import traceback
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
RAILRADAR_API_KEY = os.getenv("RAILRADAR_API_KEY")

import threading
from typing import Any, Coroutine

# Resolve weather, railway, and aviation server interpreter paths dynamically
BASE_DIR = Path(__file__).resolve().parent
PYTHON_PATH = sys.executable

AVIATION_VENV_PYTHON = r"C:\Users\ishik\Music\aviationstack-mcp\.venv\Scripts\python.exe"
AVIATION_PYTHON = AVIATION_VENV_PYTHON if os.path.exists(AVIATION_VENV_PYTHON) else PYTHON_PATH

WEATHER_SERVER_PATH = str(BASE_DIR / "custom_weather_mcp_server.py")
RAILWAY_SERVER_PATH = str(BASE_DIR / "custom_railway_mcp_server.py")

ALL_MCP_SERVERS = {
    "tavily": {
        "transport": "streamable_http",
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
    },

    "aviationstack": {
        "transport": "stdio",
        "command": AVIATION_PYTHON,
        "args": [
            "-m",
            "aviationstack_mcp",
            "mcp",
            "run"
        ],
        "env": {
            "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY or ""
        }
    },

    "weather": {
        "transport": "stdio",
        "command": PYTHON_PATH,
        "args": [
            WEATHER_SERVER_PATH
        ],
        "env": {
            "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY or ""
        }
    },

    "railway": {
        "transport": "stdio",
        "command": PYTHON_PATH,
        "args": [
            RAILWAY_SERVER_PATH
        ],
        "env": {
            "RAILRADAR_API_KEY": RAILRADAR_API_KEY or "",
            "TAVILY_API_KEY": TAVILY_API_KEY or ""
        }
    }
}

def log_nested_exception(exc, prefix=""):
    """Recursively logs nested sub-exceptions inside BaseExceptionGroup / ExceptionGroup."""
    if isinstance(exc, (BaseExceptionGroup, ExceptionGroup)):
        print(f"{prefix}TaskGroup ExceptionGroup ({len(exc.exceptions)} sub-exceptions):")
        for idx, sub in enumerate(exc.exceptions, start=1):
            print(f"{prefix}  Sub-exception {idx}: {type(sub).__name__}: {sub}")
            log_nested_exception(sub, prefix + "    ")
    else:
        print(f"{prefix}Inner Exception [{type(exc).__name__}]: {exc}")
        traceback.print_exception(type(exc), exc, exc.__traceback__)

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


_server_clients = {}

async def get_mcp_tools_for_server(server_name: str):
    """Initializes and returns tools for a specific MCP server safely in isolation."""
    global _server_clients
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    cached = _server_clients.get(server_name)
    if cached and cached.get("loop") == current_loop:
        return cached["tools"]

    if server_name not in ALL_MCP_SERVERS:
        return []

    server_cfg = ALL_MCP_SERVERS[server_name]
    print(f"\n[MCP LOG] Initializing MCP server '{server_name}'...")
    try:
        client = MultiServerMCPClient({server_name: server_cfg})
        tools = await client.get_tools()
        _server_clients[server_name] = {"client": client, "tools": tools, "loop": current_loop}
        return tools
    except (BaseExceptionGroup, ExceptionGroup, Exception) as exc:
        print(f"\n[MCP ERROR] Failed to initialize server '{server_name}':")
        log_nested_exception(exc)
        return []

async def tavily_mcp_search(query: str):
    tools = await get_mcp_tools_for_server("tavily")
    search_tool = next((t for t in tools if t.name == "tavily_search"), None)
    if search_tool:
        return await search_tool.ainvoke({"query": query})
    return ""

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):
    tools = await get_mcp_tools_for_server("aviationstack")
    tool = next((t for t in tools if t.name == tool_name), None)
    if tool:
        return await tool.ainvoke(tool_args or {})
    return "Aviation tool unavailable"

async def get_airports():
    tools = await get_mcp_tools_for_server("aviationstack")
    tool = next((t for t in tools if t.name == "list_airports"), None)
    if not tool:
        return "Airport tool unavailable"
    return await tool.ainvoke({})

async def get_airlines():
    tools = await get_mcp_tools_for_server("aviationstack")
    tool = next((t for t in tools if t.name == "list_airlines"), None)
    if not tool:
        return "Airline tool unavailable"
    return await tool.ainvoke({})

async def weather_mcp_search(city: str):
    tools = await get_mcp_tools_for_server("weather")
    weather_tool = next((t for t in tools if t.name == "get_current_weather"), None)
    if weather_tool:
        return await weather_tool.ainvoke({"city": city})
    return {"status": "unavailable"}

async def forecast_mcp_search(city: str):
    tools = await get_mcp_tools_for_server("weather")
    forecast_tool = next((t for t in tools if t.name == "get_forecast"), None)
    if forecast_tool:
        return await forecast_tool.ainvoke({"city": city})
    return {"status": "unavailable"}

async def railway_mcp_search(source: str, destination: str):
    tools = await get_mcp_tools_for_server("railway")
    railway_tool = next((t for t in tools if t.name == "search_trains"), None)
    if railway_tool:
        return await railway_tool.ainvoke({"source": source, "destination": destination})
    
    # Fallback to direct Tavily web search if railway server process failed
    print("\n[RAILWAY FALLBACK] Using Tavily direct search for Indian Railways options...")
    tav_res = await tavily_mcp_search(f"popular direct trains schedule timing fare from {source} to {destination} vande bharat express irctc")
    return {
        "status": "success",
        "source": source,
        "destination": destination,
        "raw_search_notes": str(tav_res)[:2000],
        "booking_platforms": [
            {"name": "IRCTC", "url": "https://www.irctc.co.in/nget/train-search"},
            {"name": "ConfirmTkt", "url": "https://www.confirmtkt.com/"},
            {"name": "RailYatri", "url": "https://www.railyatri.in/train-tickets"}
        ],
        "tip": "Book AC coaches early during peak travel seasons as berths fill quickly."
    }




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