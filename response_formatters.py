import json
import re
import html
from datetime import datetime

# ── Hotel response formatting ──

def extract_text_content(data) -> str:
    """Recursively extract string content from lists, dicts, or objects."""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "\n".join(extract_text_content(item) for item in data)
    if isinstance(data, dict):
        if "text" in data:
            return extract_text_content(data["text"])
        return json.dumps(data)
    if hasattr(data, "text"):
        return extract_text_content(getattr(data, "text"))
    return str(data)

def parse_json_content(text: str):
    """Parse JSON strings safely."""
    text_stripped = text.strip()
    if (text_stripped.startswith("{") and text_stripped.endswith("}")) or \
       (text_stripped.startswith("[") and text_stripped.endswith("]")):
        try:
            return json.loads(text_stripped)
        except Exception:
            pass
    return None

def format_hotel_response(hotel_data) -> str:
    """Format Hotel Agent response into clean Markdown."""
    text = extract_text_content(hotel_data)
    
    # If the text does not contain typical JSON indicators, assume it is already formatted
    if not ("{" in text or "[" in text):
        return text

    parsed = parse_json_content(text)
    results = []
    
    if isinstance(parsed, dict):
        if "results" in parsed:
            results = parsed["results"]
        else:
            results = [parsed]
    elif isinstance(parsed, list):
        # Could be a list of results or list of content blocks
        for item in parsed:
            if isinstance(item, dict) and ("title" in item or "url" in item or "content" in item):
                results.append(item)
            elif isinstance(item, dict) and "text" in item:
                # Nested text in a list item
                inner = parse_json_content(item["text"])
                if isinstance(inner, dict) and "results" in inner:
                    results.extend(inner["results"])
                elif isinstance(inner, list):
                    results.extend(inner)

    if not results:
        # Fallback if parsing failed but it was JSON-like
        return text

    extracted = []
    for r in results:
        if not isinstance(r, dict):
            continue
        title = r.get("title") or r.get("name")
        url = r.get("url") or r.get("link") or r.get("href")
        content = r.get("content") or r.get("snippet") or r.get("description") or r.get("text")
        score = r.get("score")
        
        if title or content or url:
            extracted.append({
                "title": title or "Recommended Accommodation",
                "url": url or "",
                "content": content or "",
                "score": score if isinstance(score, (int, float)) else 0.0
            })

    # Deduplicate by url or title
    seen_urls = set()
    seen_titles = set()
    unique_results = []
    for item in extracted:
        url = item["url"]
        title_lower = item["title"].lower()
        if url and url in seen_urls:
            continue
        if title_lower in seen_titles:
            continue
        unique_results.append(item)
        if url:
            seen_urls.add(url)
        seen_titles.add(title_lower)

    # Sort by score descending (if non-zero)
    if any(item["score"] > 0 for item in unique_results):
        unique_results.sort(key=lambda x: x["score"], reverse=True)

    # Display 3 to 5 results
    final_results = unique_results[:5]

    if not final_results:
        return text

    markdown_lines = ["### Recommended Hotels and Areas\n"]
    for i, item in enumerate(final_results, 1):
        escaped_title = html.escape(item["title"])
        escaped_content = html.escape(item["content"])
        
        # Truncate content if excessively long
        if len(escaped_content) > 300:
            escaped_content = escaped_content[:300] + "..."
            
        markdown_lines.append(f"#### {i}. {escaped_title}")
        if escaped_content:
            markdown_lines.append(escaped_content)
        if item["url"]:
            markdown_lines.append(f"\n[View source]({item['url']})\n")
        else:
            markdown_lines.append("")

    return "\n".join(markdown_lines).strip()


# ── Weather response formatting ──

def format_datetime(dt_str: str) -> str:
    """Format datetime string cleanly."""
    try:
        dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d %b, %I:%M %p")
    except Exception:
        return dt_str

def format_weather_response(weather_data_raw, forecast_data_raw, city: str) -> str:
    """Format Weather Agent response into clean Markdown using LLM for travel weather assistant features."""
    from chat_history import llm
    
    weather_text = extract_text_content(weather_data_raw)
    forecast_text = extract_text_content(forecast_data_raw)
    
    prompt = f"""
    You are an expert travel weather assistant. Format the following raw weather and forecast data for the city of "{city}" into a complete, clean, professional travel weather report.
    
    Raw Weather Data:
    {weather_text}
    
    Raw Forecast Data:
    {forecast_text}
    
    Your report MUST follow this exact structure:
    
    ### Current Weather — {city}
    - Temperature: [X]°C
    - Feels like: [X]°C
    - Condition: [Condition]
    - Humidity: [X]%
    - Wind speed: [X] m/s or km/h
    - Visibility: [X] km or miles
    - UV Index: [X]
    - Sunrise: [Time]
    - Sunset: [Time]
    
    ### Forecast
    [Provide a clean, beautifully formatted markdown forecast table or list for the travel duration, e.g.:
    | Day | Condition | Temp (Max/Min) |
    |---|---|---|
    | Day 1 | 🌤️ Sunny | 27°C / 19°C |
    | Day 2 | 🌧️ Light Rain | 24°C / 18°C |
    ]
    
    ### Alerts
    [If there are active weather warnings/alerts in the data, display them with a ⚠ emoji, e.g. "⚠ Heavy Rain Warning".
    If there are no alerts/warnings, output exactly: "✅ No active weather alerts."]
    
    ### Recommended Clothing
    [Provide relevant clothing recommendations based on the weather, e.g. light cotton clothes, jackets, umbrellas, etc., using emojis.]
    
    ### Travel Essentials
    [Provide a bulleted list of travel essentials relevant to this weather, e.g., sunscreen, umbrella, water bottle, etc., using emojis.]
    
    ### Weather Tip
    💡 Weather Tip:
    [Provide a short, practical travel weather tip based on the forecast, e.g., schedule outdoor sightseeing in the morning if afternoon rain is expected.]
    
    Keep the presentation clean, avoiding large paragraphs. Ensure all values (UV Index, sunrise/sunset, visibility, feels like, etc.) are extracted or estimated reasonably if not explicitly present.
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"### Weather Report — {city}\n\n{weather_text}\n\n### Forecast\n\n{forecast_text}"
