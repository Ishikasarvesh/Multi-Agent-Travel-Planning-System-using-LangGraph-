import json
import re
import ast
import html
from datetime import datetime
from url_builders import is_valid_url

# ── Safe Parser Utility ──

def parse_structured_output(value):
    """Safely parse dict, list, JSON string, code-fenced JSON, or python literal strings."""
    if isinstance(value, (dict, list)):
        return value

    if not isinstance(value, str):
        return {}

    cleaned = value.strip()

    cleaned = re.sub(
        r"^```(?:json|python)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (ValueError, SyntaxError):
        pass

    return {}

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


def has_display_value(value) -> bool:
    """Check if value is present and not a placeholder like None, N/A, null, No data."""
    if value is None:
        return False
    text = str(value).strip()
    return text.lower() not in {
        "",
        "none",
        "null",
        "n/a",
        "no data",
        "not found",
        "undefined",
        "not available"
    }

import urllib.parse
import re
import asyncio
import concurrent.futures

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b(hotel|hotels|resort|resorts|homestay|homestays|inn|inns|villas|villa|stay|stays|spa|suites|suite)\b', '', text)
    return ' '.join(text.split())

def match_hotel_title(hotel_name: str, page_title: str, url: str) -> bool:
    h_norm = normalize_text(hotel_name)
    t_norm = normalize_text(page_title)
    u_norm = normalize_text(url)
    if not h_norm:
        return False
    h_words = set(h_norm.split())
    if not h_words:
        return False
    t_words = set(t_norm.split())
    u_words = set(u_norm.split())

    t_overlap = h_words.intersection(t_words)
    u_overlap = h_words.intersection(u_words)
    
    return (len(t_overlap) / len(h_words) >= 0.3) or (len(u_overlap) / len(h_words) >= 0.3) or h_norm in t_norm or h_norm in u_norm

async def resolve_hotel_booking_link_async(hotel_name: str, location: str = "", raw_url: str = "") -> tuple[str, str]:
    """
    Resolves booking_url and link_type ('official', 'direct', or 'invalid').
    NEVER returns a Google search page or 'Search This Hotel' fallback!
    """
    clean_url = str(raw_url or "").strip()
    ota_domains = ["booking.com", "agoda.com", "hotels.com", "expedia.com", "makemytrip.com", "goibibo.com", "tripadvisor.com", "trivago.com"]

    if clean_url.startswith("http") and not any(g in clean_url.lower() for g in ["google.com/search", "/search", "searchresults", "top-10", "best-hotels", "tripadvisor.com/article"]):
        domain = urllib.parse.urlparse(clean_url).netloc.lower()
        if not any(ota in domain for ota in ota_domains):
            return clean_url, "official"
        else:
            if any(p in clean_url.lower() for p in ["/hotel/", "/hotels/", "hotel-details", "property", "homestay"]):
                return clean_url, "direct"

    query = f"{hotel_name} {location} official website booking"
    try:
        from mcp_client import tavily_mcp_search
        raw_res = await tavily_mcp_search(query)
        text_content = extract_text_content(raw_res)
        parsed = parse_structured_output(text_content)
        
        results = []
        if isinstance(parsed, dict) and "results" in parsed:
            results = parsed["results"]
        elif isinstance(parsed, list):
            results = parsed

        official_link = None
        direct_link = None

        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("link") or "").strip()
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()

            urls_in_content = re.findall(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', content)
            for u_str in urls_in_content:
                u_clean = u_str.rstrip(".,;)")
                if not u_clean.startswith("http"):
                    u_clean = "https://" + u_clean
                u_domain = urllib.parse.urlparse(u_clean).netloc.lower()
                if not any(ota in u_domain for ota in ota_domains) and not any(g in u_clean.lower() for g in ["google.com", "facebook.com", "instagram.com", "twitter.com", "wikipedia.org"]):
                    if match_hotel_title(hotel_name, u_domain, u_domain):
                        official_link = u_clean
                        break

            if official_link:
                break

            domain = urllib.parse.urlparse(url).netloc.lower()

            if not url.startswith("http"):
                continue

            if any(g in url.lower() for g in ["google.com/search", "/search", "searchresults", "top-10", "best-hotels", "tripadvisor.com/article", "blog", "news", "listicle"]):
                continue

            if not match_hotel_title(hotel_name, title, url):
                continue

            if not any(ota in domain for ota in ota_domains):
                if not official_link:
                    official_link = url
            else:
                if not direct_link:
                    if any(p in url.lower() for p in ["/hotel/", "/hotels/", "hotel-details", "property", "homestay"]):
                        direct_link = url

        if official_link:
            return official_link, "official"
        if direct_link:
            return direct_link, "direct"
    except Exception as e:
        print(f"Error resolving hotel link for {hotel_name}: {e}")

    return "", "invalid"

def resolve_hotel_booking_link(hotel_name: str, location: str = "", raw_url: str = "") -> tuple[str, str]:
    """
    Synchronous wrapper for resolving hotel booking links strictly without Google Search fallbacks.
    Returns (url, 'official'|'direct'|'invalid').
    """
    clean_url = str(raw_url or "").strip()
    ota_domains = ["booking.com", "agoda.com", "hotels.com", "expedia.com", "makemytrip.com", "goibibo.com", "tripadvisor.com", "trivago.com"]

    if clean_url.startswith("http") and not any(g in clean_url.lower() for g in ["google.com/search", "/search", "searchresults", "top-10", "best-hotels", "tripadvisor.com/article"]):
        domain = urllib.parse.urlparse(clean_url).netloc.lower()
        if not any(ota in domain for ota in ota_domains):
            return clean_url, "official"
        else:
            if any(p in clean_url.lower() for p in ["/hotel/", "/hotels/", "hotel-details", "property", "homestay"]):
                return clean_url, "direct"

    return "", "invalid"

def format_hotel_response(hotel_data, category: str = "all") -> str:
    """Format Hotel Agent response into a clean JSON string containing 7 to 9 hotel recommendations."""
    from chat_history import llm
    text = extract_text_content(hotel_data)
    
    category_clean = (category or "all").lower().strip()
    if category_clean not in ["budget", "mid-range", "luxury", "all"]:
        category_clean = "all"

    cat_instructions = {
        "budget": "Find affordable and budget-friendly hotels that offer good value and convenient transport access. Set 'budget_category': 'Budget' for all hotels.",
        "mid-range": "Find comfortable mid-range hotels with reliable amenities, good ratings and convenient locations. Set 'budget_category': 'Mid-range' for all hotels.",
        "luxury": "Find premium and luxury hotels with high-quality service, excellent facilities and strong ratings. Set 'budget_category': 'Luxury' for all hotels.",
        "all": "Return a balanced list of 7 to 9 hotels grouped into Budget (2-3 hotels), Mid-range (2-3 hotels), and Luxury (2-3 hotels) categories. Set 'budget_category' to 'Budget', 'Mid-range', or 'Luxury' appropriately for each hotel."
    }

    prompt = f"""
    You are an expert travel hotel assistant. Extract 7 to 9 real hotel recommendations from the following raw hotel search results into a clean JSON object.
    
    Category Requirement:
    {cat_instructions.get(category_clean, cat_instructions['all'])}

    Raw Search Data:
    {text}
    
    Output MUST be a single valid JSON object strictly matching this schema:
    {{
        "hotels": [
            {{
                "name": "Hotel Name",
                "why_stay": "One short concise sentence explaining why the hotel is suitable.",
                "location": "Neighborhood, area, or city location",
                "budget_category": "Budget / Mid-range / Luxury",
                "booking_url": "verified direct HTTP/HTTPS hotel property URL if present in raw search data, else null"
            }}
        ],
        "booking_platforms": [],
        "tip": "Choose accommodation near a metro or railway station to reduce daily travel time."
    }}

    Rules:
    - Extract actual individual hotel names. Do NOT use article/listicle titles (like 'TOP 10 Budget Hotels' or 'Best Hotels in Tokyo') as hotel names unless they refer to a real individual hotel.
    - For booking_url: preserve the exact direct http/https hotel property URL from the raw search data if present for that specific hotel. If unavailable or generic search list, set booking_url to null. Never output the literal string 'None' or 'N/A' as a URL.
    - Do NOT include rating, star score, review, or price fields.
    - Keep why_stay to one short, practical sentence.
    - Return ONLY the JSON object. Do not include markdown code block formatting or explanations.
    """
    try:
        response = llm.invoke(prompt)
        parsed = parse_structured_output(response.content)
        if parsed and isinstance(parsed, dict) and "hotels" in parsed and parsed["hotels"]:
            clean_hotels = []
            for h in parsed["hotels"]:
                if isinstance(h, dict) and has_display_value(h.get("name")):
                    h_name = str(h.get("name")).strip()
                    loc = str(h.get("location", "")).strip()
                    raw_u = h.get("booking_url")
                    final_url, link_type = resolve_hotel_booking_link(h_name, loc, raw_u)
                    if link_type in ["official", "direct"] and is_valid_url(final_url):
                        h["name"] = h_name
                        h["booking_url"] = final_url
                        h["link_type"] = link_type
                        clean_hotels.append(h)
            if clean_hotels:
                parsed["hotels"] = clean_hotels[:9]
                return json.dumps(parsed)
    except Exception:
        pass

    # Python Fallback parser if LLM fails or is rate-limited:
    parsed_raw = parse_structured_output(text)
    results = []
    if isinstance(parsed_raw, dict) and "results" in parsed_raw:
        results = parsed_raw["results"]
    elif isinstance(parsed_raw, list):
        results = parsed_raw

    extracted_hotels = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("name") or ""
        url = item.get("url") or item.get("link") or ""
        content = item.get("content") or item.get("snippet") or ""
        
        title_clean = re.sub(r"\s*-\s*Booking\.com|\s*-\s*Tripadvisor|\s*\|.*", "", title).strip()
        if title_clean and len(title_clean) > 3 and not title_clean.lower().startswith("best hotels") and not title_clean.lower().startswith("top 10"):
            why_stay = content[:120].strip() + "." if content else "Conveniently located accommodation."
            why_stay = why_stay.replace("..", ".")
            
            loc = title_clean.split()[-1] if len(title_clean.split()) > 1 else ""
            final_url, link_type = resolve_hotel_booking_link(title_clean, loc, url)
            
            if link_type in ["official", "direct"] and is_valid_url(final_url):
                extracted_hotels.append({
                    "name": title_clean,
                    "why_stay": why_stay,
                    "location": loc,
                    "budget_category": category_clean.capitalize() if category_clean != "all" else "Mid-range",
                    "booking_url": final_url,
                    "link_type": link_type
                })
                if len(extracted_hotels) >= 9:
                    break

    fallback_dict = {
        "hotels": extracted_hotels,
        "booking_platforms": [],
        "tip": "Choose accommodation near a metro or railway station to reduce daily travel time."
    }
    return json.dumps(fallback_dict)


# ── Weather response formatting ──

def format_weather_response(weather_data_raw, forecast_data_raw, city: str, duration: str = "") -> str:
    """Format Weather Agent response into a clean JSON string representing current weather, forecast, alerts, packing, and tip."""
    from chat_history import llm
    
    weather_text = extract_text_content(weather_data_raw)
    forecast_text = extract_text_content(forecast_data_raw)
    
    prompt = f"""
    You are an expert travel weather assistant. Format the raw weather data for "{city}" (Trip duration: {duration or 'N/A'}) into a clean JSON object.
    
    Raw Weather Data:
    {weather_text}
    
    Raw Forecast Data:
    {forecast_text}
    
    Output MUST be a single valid JSON object strictly matching this schema:
    {{
        "current": {{
            "location": "{city}",
            "temperature": "23.9°C",
            "feels_like": "24.5°C",
            "condition": "Overcast clouds",
            "humidity": "81%",
            "wind_speed": "3.8 m/s"
        }},
        "forecast": [
            {{
                "day": "Day 1",
                "date": "27 Jul 2026",
                "condition": "Overcast clouds",
                "minimum": "23°C",
                "maximum": "29°C",
                "rain_chance": "10%"
            }}
        ],
        "alerts": [],
        "packing_suggestions": [
            "Light cotton clothes",
            "Sunscreen",
            "Refillable water bottle"
        ],
        "tip": "Schedule outdoor activities during the coolest part of the day."
    }}

    Rules:
    - Include ONLY forecast entries for dates actually present in the Raw Forecast Data. Group 3-hour entries by date before outputting.
    - Do NOT invent dates or add placeholder rows (such as "No data", "N/A", "null", or empty cells) to force matching the trip duration. Output only distinct dates that exist in the raw forecast data.
    - If active severe weather warnings exist, list them in alerts (e.g. ["⚠️ Heavy Rain Warning"]). If no alerts exist, keep alerts as an empty list [].
    - Return ONLY the JSON object. Do not include markdown code block formatting or explanations.
    """
    try:
        response = llm.invoke(prompt)
        parsed = parse_structured_output(response.content)
        if parsed and isinstance(parsed, dict) and "current" in parsed and parsed["current"]:
            # Filter out any forecast entries containing 'no data' or 'n/a'
            clean_forecast = []
            for f_entry in parsed.get("forecast", []):
                if isinstance(f_entry, dict):
                    cond = str(f_entry.get("condition", "")).lower()
                    dt = str(f_entry.get("date", "")).lower()
                    if has_display_value(cond) and has_display_value(dt):
                        clean_forecast.append(f_entry)
            parsed["forecast"] = clean_forecast
            return json.dumps(parsed)
    except Exception:
        pass

    # Python Fallback parser if LLM is offline or rate-limited:
    weather_dict = parse_structured_output(weather_text)
    forecast_dict = parse_structured_output(forecast_text)

    main_w = weather_dict.get("main", {}) if isinstance(weather_dict, dict) else {}
    wind_w = weather_dict.get("wind", {}) if isinstance(weather_dict, dict) else {}
    cond_list = weather_dict.get("weather", []) if isinstance(weather_dict, dict) else []
    cond_str = cond_list[0].get("description", "Clear").capitalize() if cond_list and isinstance(cond_list[0], dict) else "Clear"

    temp_val = f"{main_w.get('temp', '24')}°C" if main_w.get('temp') else "24°C"
    feels_val = f"{main_w.get('feels_like', '25')}°C" if main_w.get('feels_like') else "25°C"
    humidity_val = f"{main_w.get('humidity', '70')}%" if main_w.get('humidity') else "70%"
    wind_val = f"{wind_w.get('speed', '3.5')} m/s" if wind_w.get('speed') else "3.5 m/s"

    raw_list = forecast_dict.get("list", []) if isinstance(forecast_dict, dict) else []
    daily_groups = {}
    for entry in raw_list:
        if isinstance(entry, dict):
            dt_txt = entry.get("dt_txt", "")
            date_key = dt_txt.split()[0] if dt_txt else "Date"
            if date_key not in daily_groups:
                daily_groups[date_key] = []
            daily_groups[date_key].append(entry)

    forecast_entries = []
    day_idx = 1
    for d_key, items in daily_groups.items():
        temps = [i.get("main", {}).get("temp") for i in items if isinstance(i.get("main"), dict) and i.get("main", {}).get("temp") is not None]
        min_t = f"{int(min(temps))}°C" if temps else "Not available"
        max_t = f"{int(max(temps))}°C" if temps else "Not available"
        
        daytime_items = [i for i in items if "12:00" in i.get("dt_txt", "")]
        rep_item = daytime_items[0] if daytime_items else items[0]
        
        w_raw = rep_item.get("weather")
        if isinstance(w_raw, str):
            cond_item = w_raw.capitalize()
        elif isinstance(w_raw, list) and len(w_raw) > 0 and isinstance(w_raw[0], dict):
            cond_item = w_raw[0].get("description", "Partly cloudy").capitalize()
        else:
            cond_item = "Partly cloudy"
        
        pop_val = rep_item.get("pop", 0)
        pop_str = f"{int(pop_val * 100)}%" if pop_val is not None else "10%"

        formatted_date = d_key
        try:
            d_obj = datetime.strptime(d_key, "%Y-%m-%d")
            formatted_date = d_obj.strftime("%d %b %Y")
        except Exception:
            pass

        forecast_entries.append({
            "day": f"Day {day_idx}",
            "date": formatted_date,
            "condition": cond_item,
            "minimum": min_t,
            "maximum": max_t,
            "rain_chance": pop_str
        })
        day_idx += 1

    if not forecast_entries:
        forecast_entries = [
            {"day": "Day 1", "date": "Today", "condition": cond_str, "minimum": "20°C", "maximum": "28°C", "rain_chance": "10%"}
        ]

    fallback_weather_dict = {
        "current": {
            "location": city or "Destination",
            "temperature": temp_val,
            "feels_like": feels_val,
            "condition": cond_str,
            "humidity": humidity_val,
            "wind_speed": wind_val
        },
        "forecast": forecast_entries,
        "alerts": [],
        "packing_suggestions": ["Light cotton clothes", "Sunscreen", "Refillable water bottle"],
        "tip": "Check local weather updates daily during travel."
    }
    return json.dumps(fallback_weather_dict)

def answer_hotel_proximity_question(hotel_results_json_or_text: str, destination: str, query: str) -> str:
    parsed = parse_structured_output(hotel_results_json_or_text)
    hotels = []
    if isinstance(parsed, dict):
        hotels = parsed.get("hotels") or parsed.get("results") or []
    elif isinstance(parsed, list):
        hotels = parsed
    
    if not hotels:
        return "I don't have hotel details available in the current plan to check station proximity."

    dest_clean = destination.strip().title() if destination else "the destination"
    station_name = "the central railway station"
    if "kerala" in dest_clean.lower() or "kochi" in dest_clean.lower():
        station_name = "Ernakulam Junction (ERS)"
    elif "tokyo" in dest_clean.lower():
        station_name = "Tokyo Station"
    elif "jaipur" in dest_clean.lower():
        station_name = "Jaipur Junction"
    elif "mumbai" in dest_clean.lower():
        station_name = "Mumbai Central / CSMT"
    elif "goa" in dest_clean.lower():
        station_name = "Madgaon Junction"

    best_hotel = None
    for h in hotels:
        if not isinstance(h, dict):
            continue
        why = str(h.get("why_stay", "")).lower()
        loc = str(h.get("location", "")).lower()
        name = str(h.get("name", "")).lower()
        if "station" in why or "station" in loc or "rail" in why or "transit" in why or "central" in loc or "transport" in why:
            best_hotel = h
            break
            
    if not best_hotel:
        best_hotel = hotels[0] if isinstance(hotels[0], dict) else {"name": "Recommended Hotel"}
        
    h_name = best_hotel.get("name", "Recommended Hotel")
    why_text = best_hotel.get("why_stay") or f"It is conveniently located in {dest_clean} with easy access to transport."
    raw_url = best_hotel.get("booking_url") or best_hotel.get("url") or ""
    
    link_md = ""
    if raw_url:
        final_url, link_type = resolve_hotel_booking_link(h_name, dest_clean, raw_url)
        if final_url:
            label = "Open Hotel Website" if link_type == "official" else "View Hotel Booking"
            link_md = f"\n\n[{label}]({final_url})"

    resp = f"**{h_name}** is the most convenient option near {station_name}. {why_text}{link_md}"
    return resp
