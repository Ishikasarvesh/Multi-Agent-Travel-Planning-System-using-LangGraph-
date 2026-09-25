# pip install mcp requests

from mcp.server.fastmcp import FastMCP
import requests
import os

from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("Weather Server")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

@mcp.tool()
def get_current_weather(city: str):
    if not OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY is not configured in environment"}

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            },
            timeout=10
        )
        data = response.json()
        if response.status_code != 200:
            return {"error": data.get("message", "Failed to fetch current weather")}

        return {
            "city": data.get("name", city),
            "temperature_c": data.get("main", {}).get("temp"),
            "feels_like_c": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "condition": data.get("weather", [{}])[0].get("description", "Clear"),
            "wind_speed": data.get("wind", {}).get("speed")
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_forecast(city: str):
    if not OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY is not configured in environment"}

    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return {"error": data.get("message", "Failed to fetch forecast")}

        raw_list = data.get("list", [])
        forecast = []

        for item in raw_list:
            forecast.append({
                "dt_txt": item.get("dt_txt"),
                "temp": item.get("main", {}).get("temp"),
                "temp_min": item.get("main", {}).get("temp_min"),
                "temp_max": item.get("main", {}).get("temp_max"),
                "weather": item.get("weather", [{}])[0].get("description", "Clear"),
                "pop": item.get("pop", 0)
            })

        return {
            "city": city,
            "list": forecast,
            "raw": data
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()