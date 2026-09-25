import os
import uuid
import json
import ast
import copy
import streamlit as st

from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from main import app
from attraction_images import render_itinerary_with_inline_images
from chat_database import (
    create_chat_tables, create_chat, list_chats, rename_chat, delete_chat,
    save_message, load_messages, delete_last_message, update_last_user_message,
    save_chat_state, load_chat_state
)
from chat_history import generate_chat_title, parse_assistant_message
from response_formatters import parse_structured_output
import pandas as pd
import re
from urllib.parse import quote_plus
from agent_visualizer import (
    render_agent_visualizer,
    reset_visualizer_for_run,
    record_node_start,
    record_node_complete,
    record_node_failure,
    record_download_builder,
    init_visualizer_state
)
from url_builders import (
    build_flight_platform_links,
    build_railway_platform_links,
    build_hotel_platform_links,
    build_google_maps_link,
    is_valid_url
)

def get_active_trip_plan() -> dict:
    """Helper to safely fetch current plan parameters from session state."""
    try:
        curr_chat = st.session_state.get("current_chat_id")
        if curr_chat and curr_chat in st.session_state.get("chat_sessions", {}):
            return st.session_state.chat_sessions[curr_chat].get("current_plan", {}) or {}
    except Exception:
        pass
    return {}

def convert_markdown_links_to_target_blank(text: str) -> str:
    """
    Converts markdown links [Label](URL) in text to HTML anchor tags with target="_blank"
    so links open in a new browser tab. Replaces broken/invalid URLs with plain text.
    """
    if not text or not isinstance(text, str):
        return ""
    
    def replacer(match):
        label = match.group(1).strip()
        url = match.group(2).strip()
        if is_valid_url(url):
            return f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="color: #4ea8f0; text-decoration: underline;">{label}</a>'
        return label

    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replacer, text)

def render_platform_links(links: list):
    """Renders a list of platform dictionaries with target='_blank' clickable HTML links."""
    if not links:
        return
    html_items = []
    for link in links:
        if isinstance(link, dict):
            name = link.get("name", "").strip()
            url = link.get("url", "").strip()
            if name and is_valid_url(url):
                html_items.append(f'<li><a href="{url}" target="_blank" rel="noopener noreferrer" style="color: #4ea8f0; text-decoration: underline;">{name}</a></li>')
            elif name:
                html_items.append(f'<li>{name}</li>')
    if html_items:
        st.markdown(f'<ul style="margin-top: 0; padding-left: 1.2rem;">{"".join(html_items)}</ul>', unsafe_allow_html=True)
        st.markdown("")

def build_hotel_booking_links(destination: str = "") -> list:
    dest_clean = destination.strip() if destination else ""
    encoded_dest = quote_plus(dest_clean) if dest_clean else ""
    
    booking_url = f"https://www.booking.com/searchresults.html?ss={encoded_dest}" if encoded_dest else "https://www.booking.com/"
    mmt_url = f"https://www.makemytrip.com/hotels/hotel-listing/?city={encoded_dest}" if encoded_dest else "https://www.makemytrip.com/hotels/"
    agoda_url = f"https://www.agoda.com/search?city={encoded_dest}" if encoded_dest else "https://www.agoda.com/"
    expedia_url = "https://www.expedia.com/Hotels"
    hotels_url = "https://www.hotels.com/"
    goibibo_url = "https://www.goibibo.com/hotels/"
    
    return [
        {"name": "Booking.com", "url": booking_url},
        {"name": "MakeMyTrip Hotels", "url": mmt_url},
        {"name": "Agoda", "url": agoda_url},
        {"name": "Expedia", "url": expedia_url},
        {"name": "Hotels.com", "url": hotels_url},
        {"name": "Goibibo Hotels", "url": goibibo_url},
    ]

def get_download_filename(destination: str = "", chat_title: str = "") -> str:
    dest_source = destination or chat_title or "trip"
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', dest_source.lower()).strip('_')
    return f"{slug or 'travel'}_plan_latest.md"

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

def format_hotels_for_markdown(hotel_results: str) -> str:
    if not hotel_results:
        return "Hotel recommendations unavailable."
    if "preferred hotel budget" in hotel_results.lower():
        return hotel_results
    parsed = parse_hotel_output(hotel_results)
    if isinstance(parsed, dict) and parsed.get("parse_failed"):
        return "Hotel recommendations unavailable."
    norm = normalize_hotel_data(parsed)
    hotels = norm.get("hotels", [])
    if not hotels:
        return "No hotel recommendations available."
    
    lines = []
    categories = set(h.get("budget_category", "").capitalize() for h in hotels if h.get("budget_category"))
    if len(categories) > 1:
        for cat_key, cat_title in [("Budget", "Budget Hotels"), ("Mid-range", "Mid-Range Hotels"), ("Luxury", "Luxury Hotels")]:
            group = [h for h in hotels if h.get("budget_category", "").lower() == cat_key.lower()]
            if group:
                lines.append(f"### {cat_title}\n")
                for h in group:
                    lines.append(f"### {h['name']}\n")
                    if has_display_value(h.get("why_stay")):
                        lines.append(f"**Why Stay:** {h['why_stay']}\n")
                    if has_display_value(h.get("location")):
                        lines.append(f"**Location:** {h['location']}\n")
                    b_url = h.get("booking_url")
                    if has_display_value(b_url) and is_valid_url(b_url):
                        lines.append(f"**Website:** [{h['name']} - {str(b_url).strip()}]({str(b_url).strip()})\n")
                    else:
                        lines.append(f"**Website:** ❌ Official website not found\n")
                    lines.append("")
    else:
        for h in hotels:
            lines.append(f"### {h['name']}\n")
            if has_display_value(h.get("why_stay")):
                lines.append(f"**Why Stay:** {h['why_stay']}\n")
            if has_display_value(h.get("location")):
                lines.append(f"**Location:** {h['location']}\n")
            b_url = h.get("booking_url")
            if has_display_value(b_url) and is_valid_url(b_url):
                lines.append(f"**Website:** [{h['name']} - {str(b_url).strip()}]({str(b_url).strip()})\n")
            else:
                lines.append(f"**Website:** ❌ Official website not found\n")
            lines.append("")
            
    if norm.get("tip"):
        lines.append(f"> **Hotel Tip:** {norm['tip']}")
        
    return "\n".join(lines)

def build_download_document(current_plan: dict, travel_results: dict, chat_title: str = "") -> str:
    dest = current_plan.get("destination") or "N/A"
    duration = current_plan.get("duration") or "N/A"
    
    header_md = format_itinerary_header(current_plan)
    t_mode = (current_plan.get("transport_mode") or "").lower()

    sections = [
        header_md,
        "\n---",
    ]

    railway_content = travel_results.get("railway_results", "") or current_plan.get("railway_results", "")
    flight_content = travel_results.get("flight_results", "") or current_plan.get("flight_results", "")

    if t_mode in ["railway", "cheaper"]:
        if railway_content:
            sections.extend([
                "\n## 🚆 Railway Agent",
                railway_content,
                "\n---"
            ])
    elif t_mode in ["flight", "luxury"]:
        if flight_content:
            sections.extend([
                "\n## ✈️ Flight Information",
                flight_content,
                "\n---"
            ])
    elif t_mode in ["all", "both", "select all"]:
        if railway_content:
            sections.extend([
                "\n## 🚆 Railway Agent",
                railway_content,
                "\n---"
            ])
        if flight_content:
            sections.extend([
                "\n## ✈️ Flight Information",
                flight_content,
                "\n---"
            ])
    else:
        if railway_content:
            sections.extend([
                "\n## 🚆 Railway Agent",
                railway_content,
                "\n---"
            ])
        if flight_content:
            sections.extend([
                "\n## ✈️ Flight Information",
                flight_content,
                "\n---"
            ])

    sections.extend([
        "\n## 🏨 Hotel Recommendations",
        format_hotels_for_markdown(travel_results.get("hotel_results", "")),
        "\n---",
        "\n## 🌦️ Weather Information & Alerts",
        travel_results.get("weather_results", "") or "Weather information unavailable.",
        "\n---",
        "\n## 🗓️ Day-wise Travel Itinerary",
        clean_itinerary_body_text(travel_results.get("itinerary", "")) or "Itinerary unavailable.",
    ])
    
    if travel_results.get("final_response"):
        sections.extend([
            "\n---",
            "\n## 🧠 Final Recommendations & Notes",
            travel_results.get("final_response", "")
        ])

    return "\n\n".join(sections)

# Reusable formatting helper functions
def render_section_title(title: str):
    st.markdown(f"## **{title}**")
    st.markdown("")

def render_subsection_title(title: str):
    st.markdown(f"### **{title}**")
    st.markdown("")

def render_tip(label: str, text: str):
    st.markdown(f"> **{label}:** {text}")
    st.markdown("")

def parse_flights_to_df(flight_text: str) -> pd.DataFrame:
    """Parses flight text into a structured DataFrame using a quick LLM call."""
    from chat_history import llm
    prompt = f"""
    You are a data extraction assistant. Parse the following flight recommendations into a structured JSON list of flights.
    
    Flight Text:
    {flight_text}
    
    Extract columns: "Airline", "Flight", "Departure", "Arrival", "Duration", "Stops", "Price".
    Only include columns where data is available. Price should be formatted cleanly (e.g. ₹50,000).
    
    Output JSON format:
    [
        {{"Airline": "...", "Flight": "...", "Departure": "...", "Arrival": "...", "Duration": "...", "Stops": "...", "Price": "..."}}
    ]
    
    Return ONLY the JSON list. Do not include markdown code block syntax (like ```json) or explanations.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return pd.DataFrame(data)
    except Exception:
        pass
    return pd.DataFrame()

def parse_hotels_to_df(hotel_text: str) -> pd.DataFrame:
    """Parses hotel text into a structured DataFrame using a quick LLM call."""
    from chat_history import llm
    prompt = f"""
    You are a data extraction assistant. Parse the following hotel recommendations into a structured JSON list of hotels.
    
    Hotel Text:
    {hotel_text}
    
    Extract columns: "Hotel", "Why Stay", "Booking Link".
    Only include columns where data is available.
    
    Output JSON format:
    [
        {{"Hotel": "...", "Why Stay": "...", "Booking Link": "..."}}
    ]
    
    Return ONLY the JSON. Do not include markdown code block syntax or explanations.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return pd.DataFrame(data)
    except Exception:
        pass
    return pd.DataFrame()

def parse_weather_to_data(weather_text: str) -> dict:
    """Parses weather assistant report into structured current weather, forecast, alerts, clothing, essentials, and tip."""
    from chat_history import llm
    prompt = f"""
    You are a data extraction assistant. Parse the following weather report into structured components.
    
    Weather Report:
    {weather_text}
    
    Output JSON format:
    {{
        "current": {{"location": "...", "temperature": "...", "condition": "..."}},
        "forecast": [{{"day": "...", "date": "...", "condition": "...", "minimum": "...", "maximum": "...", "rain_chance": "..."}}],
        "alerts": [],
        "clothing": ["..."],
        "essentials": ["..."],
        "tip": "..."
    }}
    
    Return ONLY the JSON object. Do not include markdown code block syntax or explanations.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    return {}

def parse_trains_to_df(railway_text: str) -> pd.DataFrame:
    """Parses train text into a structured DataFrame using a quick LLM call."""
    from chat_history import llm
    prompt = f"""
    You are a data extraction assistant. Parse the following train recommendations into a structured JSON list of trains.
    
    Train Text:
    {railway_text}
    
    Extract columns: "Train", "Departure", "Arrival", "Duration", "Classes", "Fare".
    Only include columns where data is available.
    
    Output JSON format:
    [
        {{"Train": "...", "Departure": "...", "Arrival": "...", "Duration": "...", "Classes": "...", "Fare": "..."}}
    ]
    
    Return ONLY the JSON list. Do not include markdown code block syntax (like ```json) or explanations.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return pd.DataFrame(data)
    except Exception:
        pass
    return pd.DataFrame()

def render_railway_section(railway_results: str):
    if not railway_results:
        return
    render_section_title("Railway Agent")
    if "⚠️" in railway_results or "encountered an issue" in railway_results.lower() or "unavailable" in railway_results.lower():
        st.warning(railway_results)
        return
    df = parse_trains_to_df(railway_results)
    if not df.empty:
        columns_priority = ["Train", "Departure", "Arrival", "Duration", "Classes", "Fare"]
        cols = [c for c in columns_priority if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.markdown(convert_markdown_links_to_target_blank(railway_results), unsafe_allow_html=True)
    render_platform_links(build_railway_platform_links())

def render_flight_section(flight_results: str):
    if not flight_results:
        return
    render_section_title("Flight Information")
    if "⚠️" in flight_results or "encountered an issue" in flight_results.lower() or "unavailable" in flight_results.lower():
        st.warning(flight_results)
        return
    df = parse_flights_to_df(flight_results)
    if not df.empty:
        columns_priority = ["Airline", "Flight", "Departure", "Arrival", "Duration", "Stops", "Price"]
        cols = [c for c in columns_priority if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.markdown(convert_markdown_links_to_target_blank(flight_results), unsafe_allow_html=True)
    render_platform_links(build_flight_platform_links())

def clean_structured_text(value: str) -> str:
    cleaned = value.strip()

    cleaned = re.sub(
        r"^```(?:json|python)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    return cleaned.strip()


def parse_hotel_output(value):
    """Parse raw LLM hotel agent output into a dictionary with 'hotels' list."""
    if isinstance(value, dict):
        if "hotels" in value:
            return value
        return {"hotels": [value]}

    if isinstance(value, list):
        return {"hotels": value}

    if not isinstance(value, str):
        return {
            "hotels": [],
            "booking_platforms": [],
            "tip": "",
        }

    cleaned = clean_structured_text(value)

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, list):
            return {"hotels": parsed}

    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(cleaned)

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, list):
            return {"hotels": parsed}

    except (ValueError, SyntaxError):
        pass

    return {
        "hotels": [],
        "booking_platforms": [],
        "tip": "",
        "parse_failed": True,
    }


def normalize_hotel_data(data: dict) -> dict:
    hotels = data.get("hotels", [])

    if not isinstance(hotels, list):
        hotels = []

    normalized_hotels = []

    for hotel in hotels:
        if not isinstance(hotel, dict):
            continue

        name = str(hotel.get("name", "")).strip()

        if not has_display_value(name):
            continue

        b_url = str(hotel.get("booking_url", "")).strip()
        valid_url = b_url if (has_display_value(b_url) and is_valid_url(b_url)) else ""
        link_type = str(hotel.get("link_type", "direct")).strip()

        normalized_hotels.append(
            {
                "name": name,
                "why_stay": str(
                    hotel.get("why_stay", "")
                ).strip(),
                "location": str(
                    hotel.get("location", "")
                ).strip(),
                "budget_category": str(
                    hotel.get("budget_category", "")
                ).strip(),
                "booking_url": valid_url,
                "link_type": link_type,
            }
        )

    return {
        "hotels": normalized_hotels[:9],
        "booking_platforms": data.get(
            "booking_platforms", []
        ),
        "tip": str(data.get("tip", "")).strip(),
    }


def render_hotel_booking_platforms(booking_platforms: list, destination: str = ""):
    st.markdown("### **Hotel Booking Platforms**")
    plan = get_active_trip_plan()
    dst = destination or plan.get("destination", "")
    checkin = plan.get("departure_date", "")
    dur = plan.get("duration", "")
    
    platforms = build_hotel_platform_links(
        destination=dst,
        checkin_date=checkin,
        duration=dur
    )
    render_platform_links(platforms)


def render_hotel_agent(hotel_data: dict, destination: str = ""):
    st.markdown("## **Hotel Agent**")

    hotels = hotel_data.get("hotels", [])

    if not hotels:
        st.info(
            "No properly formatted hotel recommendations are available."
        )
        return

    def render_single_hotel(hotel):
        st.markdown(f"### {hotel['name']}")

        if has_display_value(hotel.get("why_stay")):
            st.markdown(f"**Why Stay:** {hotel['why_stay']}")

        if has_display_value(hotel.get("location")):
            st.markdown(f"**Location:** {hotel['location']}")

        b_url = hotel.get("booking_url")
        if has_display_value(b_url) and is_valid_url(b_url):
            url_val = str(b_url).strip()
            l_type = hotel.get("link_type", "direct")
            if l_type == "official":
                st.markdown(
                    f'**Official Website:** <a href="{url_val}" target="_blank" rel="noopener noreferrer" style="color: #4ea8f0; text-decoration: underline; font-weight: 600;">Open Hotel Website 🔗</a>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'**Booking:** <a href="{url_val}" target="_blank" rel="noopener noreferrer" style="color: #4ea8f0; text-decoration: underline; font-weight: 600;">View Hotel 🔗</a>',
                    unsafe_allow_html=True
                )

        st.markdown("---")

    categories_present = set(h.get("budget_category", "").strip().capitalize() for h in hotels if h.get("budget_category"))
    has_grouped_categories = len(categories_present) > 1 or any(c in ["Budget", "Mid-range", "Luxury"] for c in categories_present)

    if has_grouped_categories:
        order = [("Budget", "Budget Hotels"), ("Mid-range", "Mid-Range Hotels"), ("Luxury", "Luxury Hotels")]
        rendered_any = False

        for cat_key, cat_title in order:
            group = [h for h in hotels if h.get("budget_category", "").strip().lower() == cat_key.lower()]
            if group:
                rendered_any = True
                st.markdown(f"### **{cat_title}**")
                for hotel in group:
                    render_single_hotel(hotel)

        other_group = [h for h in hotels if h.get("budget_category", "").strip().lower() not in ["budget", "mid-range", "luxury"]]
        if other_group and not rendered_any:
            for hotel in other_group:
                render_single_hotel(hotel)

    else:
        for hotel in hotels:
            render_single_hotel(hotel)

    render_hotel_booking_platforms(
        hotel_data.get("booking_platforms", []),
        destination=destination
    )

    tip = hotel_data.get("tip")

    if tip:
        st.markdown(
            f"> **Hotel Tip:** {tip}"
        )


def render_hotel_section(hotel_results: str, destination: str = ""):
    if not hotel_results:
        return

    if "⚠️" in hotel_results or "encountered an issue" in hotel_results.lower() or "unavailable" in hotel_results.lower():
        st.markdown("## **Hotel Agent**")
        st.warning(hotel_results)
        return

    if "preferred hotel budget" in hotel_results.lower():
        st.markdown("## **Hotel Agent**")
        st.markdown("What is your preferred hotel budget: budget, mid-range, luxury, or all categories?")
        return

    parsed = parse_hotel_output(hotel_results)

    if isinstance(parsed, dict) and parsed.get("parse_failed"):
        st.markdown("## **Hotel Agent**")
        st.error(
            "Unable to format the hotel recommendations. Please regenerate the hotel results."
        )
        return

    normalized_data = normalize_hotel_data(parsed)
    render_hotel_agent(normalized_data, destination=destination)


def render_weather_section(weather_results: str):
    if not weather_results:
        return
    
    st.markdown("## **Weather Agent**")
    
    if "⚠️" in weather_results or "encountered an issue" in weather_results.lower() or "unavailable" in weather_results.lower():
        st.warning(weather_results)
        return
    
    data = parse_structured_output(weather_results)
    
    current = data.get("current", {}) if isinstance(data, dict) else {}
    forecast = data.get("forecast", []) if isinstance(data, dict) else []
    alerts = data.get("alerts", []) if isinstance(data, dict) else []
    packing = data.get("packing_suggestions") or data.get("recommended_clothing") or [] if isinstance(data, dict) else []
    tip = data.get("tip") or data.get("weather_tip") if isinstance(data, dict) else "Schedule outdoor activities during the coolest part of the day."

    # 1. Current Weather
    render_subsection_title("Current Weather")
    if current and isinstance(current, dict):
        curr_df = pd.DataFrame([current])
        cols = [c for c in ["location", "temperature", "feels_like", "condition", "humidity", "wind_speed"] if c in curr_df.columns]
        rename_map = {
            "location": "Location",
            "temperature": "Temperature",
            "feels_like": "Feels Like",
            "condition": "Condition",
            "humidity": "Humidity",
            "wind_speed": "Wind"
        }
        curr_df = curr_df[cols].rename(columns=rename_map)
        st.dataframe(curr_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("Current weather information unavailable.")

    # 2. Trip Forecast
    render_subsection_title("Trip Forecast")
    if forecast and isinstance(forecast, list):
        valid_forecast = []
        for entry in forecast:
            if isinstance(entry, dict):
                cond = str(entry.get("condition", "")).strip()
                dt = str(entry.get("date", "")).strip()
                if has_display_value(cond) and has_display_value(dt):
                    valid_forecast.append(entry)
        if valid_forecast:
            fore_df = pd.DataFrame(valid_forecast)
            cols = [c for c in ["day", "date", "condition", "rain_chance"] if c in fore_df.columns]
            rename_map = {
                "day": "Day",
                "date": "Date",
                "condition": "Condition",
                "rain_chance": "Recommendation"
            }
            fore_df = fore_df[cols].rename(columns=rename_map)
            st.dataframe(fore_df, use_container_width=True, hide_index=True)

            plan = get_active_trip_plan()
            trip_dur = plan.get("duration", "")
            if trip_dur and str(trip_dur).isdigit() and len(valid_forecast) < int(trip_dur):
                st.caption(f"ℹ️ Detailed forecast is currently available for the next {len(valid_forecast)} days.")
        else:
            st.markdown("Forecast data unavailable.")
    else:
        st.markdown("Forecast data unavailable.")

    # 3. Weather Alerts
    render_subsection_title("Weather Alerts")
    if alerts and isinstance(alerts, list) and len(alerts) > 0:
        for alert in alerts:
            st.warning(f"⚠️ {str(alert).lstrip('⚠️').strip()}")
    else:
        st.markdown("No active weather alerts.")

    # 4. Packing Suggestions
    render_subsection_title("Packing Suggestions")
    if packing:
        if isinstance(packing, str):
            st.markdown(packing)
        else:
            for item in packing:
                st.markdown(f"- {item}")
    else:
        st.markdown("- Light cotton clothes\n- Sunscreen\n- Refillable water bottle")

    # 5. Weather Tip
    render_tip("Weather Tip", tip)

def format_itinerary_header(current_plan: dict) -> str:
    dest_raw = str(current_plan.get("destination") or "Destination").strip()
    if dest_raw.lower() in ["kerela", "kerala"]:
        dest = "Kerala"
    else:
        dest = dest_raw.title()
    
    duration = current_plan.get("duration") or "7"
    dur_str = f"{duration} Days" if (str(duration).isdigit() and not str(duration).lower().endswith("days")) else f"{duration}"
    
    dept_raw = str(current_plan.get("departure_city") or "Mumbai").strip()
    dept = "Mumbai" if dept_raw.lower() in ["mumbai", "bombay"] else dept_raw.title()

    t_mode = (current_plan.get("transport_mode") or "").lower()
    if t_mode in ["flight", "luxury"]:
        pref = "Luxury"
    elif t_mode in ["railway", "cheaper"]:
        pref = "Cheaper"
    elif t_mode in ["all", "both", "select all", "select_all"]:
        pref = "Select All"
    else:
        pref = "Luxury" if "luxury" in str(current_plan).lower() else "Cheaper"

    header_md = f"""## Itinerary Agent

### {dest} {pref} Trip — {dur_str}

**From:** {dept}  
**To:** {dest}  
**Duration:** {dur_str}  
**Preference:** {pref}"""
    return header_md.strip()

def clean_itinerary_body_text(itinerary_text: str) -> str:
    if not itinerary_text:
        return ""
    text = itinerary_text.strip()
    day1_match = re.search(r"(?:^|\n)(#{1,4}\s*Day\s*1\b|Day\s*1:)", text, re.IGNORECASE)
    if day1_match:
        start_pos = day1_match.start()
        if text[start_pos] == '\n':
            start_pos += 1
        return text[start_pos:].strip()
    return text

def render_itinerary_section(itinerary_text: str, user_query: str):
    if not itinerary_text:
        return
        
    plan = get_active_trip_plan()
    header_md = format_itinerary_header(plan)
    body_text = clean_itinerary_body_text(itinerary_text)
    
    st.markdown(header_md)
    st.markdown("")
    
    clean_itinerary = re.split(r"💡 Itinerary Tip:|\*\*Itinerary Tip:\*\*", body_text, flags=re.IGNORECASE)[0].strip()
    render_itinerary_with_inline_images(clean_itinerary, user_query)
    
    tip_text = "Visit outdoor attractions early in the morning to avoid crowds and heat."
    match = re.search(r"Itinerary Tip:\s*(.*)", itinerary_text, re.IGNORECASE)
    if match:
        tip_text = match.group(1).strip()
    render_tip("Itinerary Tip", tip_text)

def render_assistant_response(parsed: dict, assoc_user_query: str):
    request_type = parsed.get("request_type", "full_plan")
    
    dest = ""
    t_mode = ""
    try:
        curr_chat = st.session_state.get("current_chat_id")
        if curr_chat and curr_chat in st.session_state.chat_sessions:
            plan = st.session_state.chat_sessions[curr_chat].get("current_plan", {})
            dest = plan.get("destination", "")
            t_mode = (plan.get("transport_mode") or "").lower()
    except Exception:
        pass

    final_resp = parsed.get("final_response") or ""

    # 1. Missing information prompt or direct question answer
    if final_resp and ("before i plan" in final_resp.lower() or "i just need" in final_resp.lower() or "preferred hotel budget" in final_resp.lower() or "how would you like to travel" in final_resp.lower()):
        st.markdown(final_resp)
        return

    # If it's a specific question or conversation with no plan results
    if request_type in ["plan_question", "general_conversation", "specific_question", "comparison", "summary"] and not (parsed.get("itinerary") or parsed.get("hotel_results") or parsed.get("flight_results") or parsed.get("railway_results")):
        st.markdown(parsed.get("displayed_response") or final_resp or "")
        return

    # Render every available executed agent output independently in order
    rendered_any = False

    # 1. Railway Agent
    railway_res = parsed.get("railway_results", "")
    if railway_res and has_display_value(railway_res):
        render_railway_section(railway_res)
        st.markdown("---")
        rendered_any = True

    # 2. Flight Agent
    flight_res = parsed.get("flight_results", "")
    if flight_res and has_display_value(flight_res):
        render_flight_section(flight_res)
        st.markdown("---")
        rendered_any = True

    # 3. Hotel Agent
    hotel_res = parsed.get("hotel_results", "")
    if hotel_res and has_display_value(hotel_res):
        render_hotel_section(hotel_res, destination=dest)
        st.markdown("---")
        rendered_any = True

    # 4. Weather Agent
    weather_res = parsed.get("weather_results", "")
    if weather_res and has_display_value(weather_res):
        render_weather_section(weather_res)
        st.markdown("---")
        rendered_any = True

    # 5. Itinerary Agent
    itinerary_res = parsed.get("itinerary", "")
    if itinerary_res and has_display_value(itinerary_res):
        if request_type in ["specific_day_change", "attraction_add", "attraction_remove", "attraction_change"]:
            disp = parsed.get("displayed_response") or final_resp or itinerary_res
            render_itinerary_section(disp, assoc_user_query)
        else:
            render_itinerary_section(itinerary_res, assoc_user_query)
        rendered_any = True

    if not rendered_any:
        st.markdown(parsed.get("displayed_response") or final_resp or "")


# Helper functions defined at the top to avoid NameError/execution order issues
def create_empty_chat_state():
    return {
        "messages": [],
        "current_plan": {},
        "travel_results": {
            "railway_results": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
        },
        "generated_itinerary": "",
        "attraction_images": {},
        "show_attraction_images": True,
        "latest_download_content": "",
    }

def activate_chat(chat_id: str) -> None:
    st.session_state.current_chat_id = chat_id
    msgs = load_messages(chat_id)
    state_data = load_chat_state(chat_id)
    
    st.session_state.chat_sessions[chat_id] = {
        "messages": msgs,
        "current_plan": state_data.get("current_plan") or {},
        "travel_results": state_data.get("travel_results") or {
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
        },
        "generated_itinerary": state_data.get("generated_itinerary") or "",
        "attraction_images": {},
        "show_attraction_images": True,
        "latest_download_content": "",
    }

def get_plan_summary(results, user_query):
    if not results.get("itinerary"):
        return ""
    prompt = f"""
    Extract a compact travel summary from this itinerary and user request.
    Itinerary:
    {results.get('itinerary')[:1200]}
    
    Latest Request:
    {user_query}
    
    Output format MUST be exactly like this (fill in the details, keep it brief, do not add other headers):
    **Destination:** [City/Country]
    **Duration:** [X days]
    **Budget:** [Amount]
    **Departure:** [Departure city if mentioned, or N/A]
    **Main cities:** [List main cities visited]
    
    Do not include any other text or explanation. Just the 5 lines.
    """
    try:
        from chat_history import llm
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return f"**Destination:** Detected from query\n**Duration:** Checked itinerary\n**Budget:** Checked query"


st.set_page_config(
    page_title="AI Travel Booking System",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #080d14;
}

/* ── Hero ── */
.hero-wrapper {
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 2rem;
    height: 280px;
}
.hero-bg {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    filter: brightness(0.35);
    position: absolute;
    top: 0; left: 0;
}
.hero-content {
    position: relative;
    z-index: 2;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
}
.hero-badge {
    background: rgba(58,123,213,0.25);
    border: 1px solid rgba(58,123,213,0.5);
    color: #7ab8f5 !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    margin-bottom: 0.9rem;
    display: inline-block;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.6rem;
    line-height: 1.2;
}
.hero-sub {
    color: #94adc8;
    font-size: 1rem;
    max-width: 560px;
}

/* ── Input card ── */
.input-card {
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
}
.input-label {
    color: #7ab8f5;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Quick destinations ── */
.dest-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 1.2rem;
}
.dest-chip {
    background: #111b2b;
    border: 1px solid #1e3050;
    color: #f7fdf4;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s;
}
.dest-chip:hover { background: #1a2e47; border-color: #3a7bd5; color: #fff; }

/* ── Generate button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a6bbf 0%, #0d4a8a 50%, #0a3d75 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    box-shadow: 0 0 24px rgba(26,107,191,0.35), 0 4px 15px rgba(0,0,0,0.4) !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 0 40px rgba(26,107,191,0.6), 0 6px 20px rgba(0,0,0,0.5) !important;
    transform: translateY(-2px) !important;
    background: linear-gradient(135deg, #2278d4 0%, #1057a0 50%, #0d4a8a 100%) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Agent status cards ── */
[data-testid="stStatusWidget"] {
    background: #0e1a2e !important;
    border: 1px solid #1e3050 !important;
    border-radius: 12px !important;
}
[data-testid="stStatusWidget"] > div:first-child {
    background: #0e1a2e !important;
    border-radius: 12px 12px 0 0 !important;
}
[data-testid="stStatusWidget"] details,
[data-testid="stStatusWidget"] details > div,
[data-testid="stStatusWidget"] [data-testid="stVerticalBlock"] {
    background: #0a1520 !important;
    color: #ffffff !important;
    padding: 0.25rem 0.5rem !important;
}
[data-testid="stStatusWidget"] * { color: #ffffff !important; }
[data-testid="stStatusWidget"] a { color: #4ea8f0 !important; }
[data-testid="stStatusWidget"] hr { border-color: #1e3050 !important; }

/* ── Section headers ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2e44;
}
.sec-head span { font-size: 1.15rem; font-weight: 600; color: #e0edf8; }

/* ── Metric bar ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-box {
    flex: 1;
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #4ea8f0; }
.metric-lbl { font-size: 0.78rem; color: #5a7a96; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Final plan ── */
.final-card {
    background: linear-gradient(160deg, #0c1a2e 0%, #0a1520 100%);
    border: 1px solid #1e3a5c;
    border-left: 4px solid #3a7bd5;
    border-radius: 14px;
    padding: 1.8rem;
    line-height: 1.8;
    color: #cce0f5;
    font-size: 0.95rem;
}

/* ── Save bar ── */
.save-bar {
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    color: #5a8ab0;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #090e18 !important;
    border-right: 1px solid #141f30 !important;
}
.sidebar-chip {
    background: #0e1a2b;
    border: 1px solid #1a2e44;
    border-radius: 8px;
    padding: 0.45rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.83rem;
    color: #7aa8cc;
}
.sidebar-title { color: #e0edf8; font-size: 1rem; font-weight: 600; margin: 1rem 0 0.5rem; }

/* Hide branding */
#MainMenu, footer, header { visibility: hidden; }

/* Textarea */
.stTextArea textarea {
    background: #0a1520 !important;
    border: 1px solid #1e2e44 !important;
    border-radius: 10px !important;
    color: #e8f4ff !important;
    font-size: 0.95rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #3a7bd5 !important;
    box-shadow: 0 0 0 2px rgba(58,123,213,0.2) !important;
}
.stTextArea textarea::placeholder { color: #4a6a85 !important; }

/* Text input (sidebar User ID field) */
input[type="text"], .stTextInput input {
    background: #0e1a2b !important;
    border: 1px solid #1a2e44 !important;
    border-radius: 8px !important;
    color: #e0edf8 !important;
}
input[type="text"]:focus, .stTextInput input:focus {
    border-color: #3a7bd5 !important;
    box-shadow: 0 0 0 2px rgba(58,123,213,0.2) !important;
}
input[type="text"]::placeholder { color: #3a5570 !important; }

/* All Streamlit labels — dark bg → light text */
.stTextInput label, .stTextArea label,
.stSelectbox label, .stNumberInput label {
    color: #7ab8f5 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
}

/* General markdown / paragraph text */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
    color: #cce0f5 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e8f4ff !important; }
.stMarkdown code {
    background: #0e1a2b !important;
    color: #7ab8f5 !important;
    padding: 0.15em 0.4em;
    border-radius: 4px;
}

/* Metric labels — was #5a7a96 (too dim on dark bg) */
.metric-lbl { color: #7aa8cc !important; }

/* Save bar — was #5a8ab0 (slightly dim) */
.save-bar { color: #8ab8d8 !important; }
.save-bar code { color: #7ab8f5 !important; background: #0a1520 !important; }

/* Streamlit warning / info / success on dark bg */
.stAlert { background: #0e1a2b !important; border-radius: 10px !important; }
.stAlert p, .stAlert div { color: #e0edf8 !important; }

/* Sidebar text & dividers */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown { color: #a0c4e0 !important; }
section[data-testid="stSidebar"] hr { border-color: #1a2e44 !important; }

/* Download button — light bg → dark text  */
div[data-testid="stDownloadButton"] > button {
    background: #1a3a5c !important;
    color: #e8f4ff !important;
    border: 1px solid #2a5080 !important;
    border-radius: 10px !important;
}

/* ── Attraction Card Custom CSS ── */
.attraction-card {
    width: 300px;
    min-width: 300px;
    max-width: 300px;
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 16px;
    overflow: hidden;
    margin: 0.5rem 0 1rem 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

.attraction-card img {
    width: 100%;
    height: 140px;
    object-fit: cover;
    display: block;
}

.attraction-card-body {
    padding: 0.7rem 0.9rem;
}

.attraction-name {
    color: #e8f4ff !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.3rem !important;
}

.attraction-card a {
    color: #4ea8f0 !important;
    font-size: 0.75rem !important;
    text-decoration: none !important;
}

.attraction-card a:hover {
    text-decoration: underline !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🌍 AI Travel Planner</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Chat history persistence ──
    # Initialize tables
    create_chat_tables()

    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_chat_id = None
        st.session_state.is_new_chat = True
        st.session_state.current_plan_summary = ""
        st.session_state.saved_filename = ""
        st.session_state.saved_file_content = ""
        st.rerun()

    st.markdown("<div class='sidebar-title'>💬 Recent Chats</div>", unsafe_allow_html=True)
    chats = list_chats()
    active_chat_id = st.session_state.get("current_chat_id")

    for c in chats:
        c_id = c["chat_id"]
        c_title = c["title"]
        
        is_active = (c_id == active_chat_id and not st.session_state.get("is_new_chat", False))
        btn_label = f"📍 {c_title}" if is_active else f"📄 {c_title}"
        
        if st.button(btn_label, key=f"chat_select_{c_id}", use_container_width=True):
            st.session_state.current_chat_id = c_id
            st.session_state.is_new_chat = False
            activate_chat(c_id)
            st.session_state.current_plan_summary = ""
            st.rerun()

        if is_active:
            col_rename, col_delete = st.columns(2)
            with col_rename:
                with st.popover("✏️ Rename", use_container_width=True):
                    new_title = st.text_input("New Title", value=c_title, key=f"rename_input_{c_id}")
                    if st.button("Save", key=f"rename_save_{c_id}", use_container_width=True):
                        rename_chat(c_id, new_title)
                        st.rerun()
            with col_delete:
                with st.popover("🗑️ Delete", use_container_width=True):
                    st.write("Confirm delete?")
                    if st.button("Yes, Delete", key=f"delete_confirm_{c_id}", use_container_width=True):
                        delete_chat(c_id)
                        st.session_state.chat_sessions.pop(c_id, None)
                        remaining_chats = list_chats()
                        if remaining_chats:
                            next_id = remaining_chats[0]["chat_id"]
                            st.session_state.current_chat_id = next_id
                            activate_chat(next_id)
                        else:
                            new_id = str(uuid.uuid4())
                            create_chat(new_id, "New Chat")
                            save_chat_state(new_id, {}, {}, "")
                            st.session_state.current_chat_id = new_id
                            activate_chat(new_id)
                        st.rerun()

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>Powered by</div>", unsafe_allow_html=True)
    for tech in ["🔗 LangGraph", "🧠 Groq · LLaMA 3.3 70B", "🐘 PostgreSQL", "🔍 Tavily Search", "✈️ AviationStack"]:
        st.markdown(f"<div class='sidebar-chip'>{tech}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-title'>Agent Pipeline</div>", unsafe_allow_html=True)
    for step in ["① Flight Agent", "② Hotel Agent", "③ Itinerary Agent", "④ Final Agent"]:
        st.markdown(f"<div class='sidebar-chip'>{step}</div>", unsafe_allow_html=True)

# End of helper declarations

# Initialize Session State Variables
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

# Ensure database tables exist
create_chat_tables()

# Fresh app startup: if current_chat_id is not in session state, set to None and is_new_chat = True
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
    st.session_state.is_new_chat = True

current_chat_id = st.session_state.current_chat_id

# If current_chat_id is set and not a new unsaved chat, load or sync state
if current_chat_id and not st.session_state.get("is_new_chat", False):
    if current_chat_id not in st.session_state.chat_sessions:
        existing_chats = [c["chat_id"] for c in list_chats()]
        if current_chat_id in existing_chats:
            activate_chat(current_chat_id)
        else:
            st.session_state.chat_sessions[current_chat_id] = create_empty_chat_state()

    active_session = st.session_state.chat_sessions[current_chat_id]
    st.session_state.chat_messages = active_session.get("messages", [])
    st.session_state.travel_results = active_session.get("travel_results", {})
    st.session_state.generated_itinerary = active_session.get("generated_itinerary", "")
    st.session_state.plan_generated = bool(st.session_state.travel_results.get("itinerary"))
else:
    st.session_state.chat_messages = []
    st.session_state.travel_results = {}
    st.session_state.generated_itinerary = ""
    st.session_state.plan_generated = False

if "saved_filename" not in st.session_state:
    st.session_state.saved_filename = ""

# ── Render Hero or Compact Header ──────────────────────────────────────────────
if not st.session_state.chat_messages:
    st.markdown("""
    <div class="hero-wrapper">
        <img class="hero-bg"
             src="https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1400&q=80"
             alt="airplane above clouds"/>
        <div class="hero-content">
            <div class="hero-badge">✦ Multi-Agent AI System</div>
            <div class="hero-title">✈️ AI Travel Booking System</div>
            <div class="hero-sub">Four specialized agents work together — searching flights, hotels, building an itinerary, and delivering your perfect trip plan.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Destination chips template
    st.markdown("<div class='input-label'>🗺️ Quick Suggestions</div>", unsafe_allow_html=True)
    QUICK = ["Plan a 7-day Japan trip under ₹2 lakhs.", "Plan a Paris trip for 5 days.", "Plan a Dubai weekend trip.", "Plan a Bali backpacking 10 days."]
    qcols = st.columns(len(QUICK))
    for qc, label in zip(qcols, QUICK):
        with qc:
            if st.button(label, key=f"q_{label}", use_container_width=True):
                st.session_state.trigger_query = label
                st.rerun()
else:
    # Show compact header with View Agent Flow button when in chat mode
    h_col1, h_col2 = st.columns([3.2, 1.2])
    with h_col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.8rem;">✈️</span>
            <h2 style="margin: 0; color: #e8f4ff;">Conversational Travel Assistant</h2>
        </div>
        """, unsafe_allow_html=True)
    with h_col2:
        if st.session_state.current_chat_id in st.session_state.get("chat_execution_snapshots", {}):
            if st.button("⚡ View Agent Flow", key=f"btn_flow_{st.session_state.current_chat_id}", use_container_width=True):
                st.session_state.show_visualizer = True
                st.session_state.visualizer_mode = "replay"
                st.rerun()




# ── Current Plan Summary Card ──────────────────────────────────────────────────
if st.session_state.plan_generated:
    config = {"configurable": {"thread_id": st.session_state.current_chat_id}}
    try:
        state_snapshot = app.get_state(config)
        current_plan_json = state_snapshot.values.get("current_plan", "{}") if state_snapshot and state_snapshot.values else "{}"
        plan_dict = json.loads(current_plan_json)
    except Exception:
        plan_dict = {}

    if plan_dict and plan_dict.get("destination"):
        cities_val = plan_dict.get('cities', [])
        cities_str = ', '.join(cities_val) if isinstance(cities_val, list) else str(cities_val)
        
        st.session_state.current_plan_summary = f"""**Destination:** {plan_dict.get('destination', 'N/A')}
**Duration:** {plan_dict.get('duration', 'N/A')} days
**Budget:** {plan_dict.get('budget', 'N/A')}
**Departure:** {plan_dict.get('departure_city', 'N/A')}
**Main cities:** {cities_str}"""

        st.markdown(f"""
        <div style="background: #0e1623; border: 1px solid #1e2e44; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;">
            <div style="color: #7ab8f5; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;">📍 Current Plan Summary</div>
            <div style="font-size: 0.92rem; line-height: 1.6; color: #cce0f5;">
                {st.session_state.current_plan_summary.replace('\n', '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Render Chat Messages ───────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.chat_messages):
    role = msg["role"]
    content = msg["content"]
    
    with st.chat_message(role):
        if role == "user":
            st.markdown(content)
            
            # Show edit popover only for the last user message
            last_user_idx = -1
            for i, m in enumerate(st.session_state.chat_messages):
                if m["role"] == "user":
                    last_user_idx = i
            
            if idx == last_user_idx:
                with st.popover("✏️ Edit Last Request"):
                    edited_query = st.text_area("Modify your request", value=content, key=f"edit_query_{idx}")
                    if st.button("Submit Changes", key=f"edit_submit_{idx}"):
                        update_last_user_message(st.session_state.current_chat_id, edited_query)
                        # Delete corresponding assistant message if exists
                        if idx + 1 < len(st.session_state.chat_messages) and st.session_state.chat_messages[idx+1]["role"] == "assistant":
                            delete_last_message(st.session_state.current_chat_id)
                        st.session_state.chat_messages = load_messages(st.session_state.current_chat_id)
                        st.session_state.trigger_query = edited_query
                        st.rerun()
        
        elif role == "assistant":
            parsed = parse_assistant_message(content)
            
            assoc_user_query = ""
            if idx > 0 and st.session_state.chat_messages[idx-1]["role"] == "user":
                assoc_user_query = st.session_state.chat_messages[idx-1]["content"]
                
            render_assistant_response(parsed, assoc_user_query)
            
            # Show Download button and auto-save info block for any plan message
            if (parsed.get("itinerary") or parsed.get("flight_results") or st.session_state.plan_generated) and st.session_state.current_chat_id:
                active_sess = st.session_state.chat_sessions.get(st.session_state.current_chat_id, {})
                c_plan = active_sess.get("current_plan", {})
                t_results = active_sess.get("travel_results", {})
                
                chat_title_val = ""
                try:
                    chats_list = list_chats()
                    m_chat = next((c for c in chats_list if c["chat_id"] == st.session_state.current_chat_id), None)
                    if m_chat:
                        chat_title_val = m_chat.get("title", "")
                except Exception:
                    pass

                dl_doc = build_download_document(c_plan, t_results, chat_title=chat_title_val)
                dl_name = get_download_filename(c_plan.get("destination", ""), chat_title=chat_title_val)

                if idx == len(st.session_state.chat_messages) - 1:
                    download_column, info_column = st.columns([1, 3])
                    with download_column:
                        st.download_button(
                            "⬇️ Download Plan",
                            data=dl_doc,
                            file_name=dl_name,
                            mime="text/markdown",
                            key=f"dl_latest_{idx}",
                            use_container_width=True,
                        )
                    with info_column:
                        st.markdown(
                            "<div class='save-bar'>"
                            f"📁 Auto-saved → "
                            f"<code>travel_plans/{dl_name}</code>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    col_dl, col_space = st.columns([1, 2])
                    with col_dl:
                        st.download_button(
                            "⬇️ Download Plan",
                            data=dl_doc,
                            file_name=dl_name,
                            mime="text/markdown",
                            key=f"dl_{idx}",
                            use_container_width=True
                        )
                
            # If last assistant response, show Regenerate option
            if idx == len(st.session_state.chat_messages) - 1:
                if st.button("🔄 Regenerate Last Response", key=f"regenerate_{idx}"):
                    delete_last_message(st.session_state.current_chat_id)
                    user_msgs = [m for m in st.session_state.chat_messages if m["role"] == "user"]
                    if user_msgs:
                        last_query = user_msgs[-1]["content"]
                        st.session_state.chat_messages = load_messages(st.session_state.current_chat_id)
                        st.session_state.trigger_query = last_query
                        st.rerun()


# ── Agent pipeline execution live stream ──────────────────────────────────────
AGENT_META = {
    "railway_agent": ("🚆", "Railway Agent"),
    "flight_agent": ("✈️", "Flight Agent"),
    "hotel_agent": ("🏨", "Hotel Agent"),
    "weather_agent": ("🌦️", "Weather Agent"),
    "itinerary_agent": ("🗓️", "Itinerary Agent"),
}

if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
    latest_user_query = st.session_state.chat_messages[-1]["content"].strip()
    
    # Dedicated placeholder for live streaming visualizer overlay
    viz_placeholder = st.empty()

    from main import is_international_trip, extract_transport_preference
    active_plan = get_active_trip_plan()
    dest_check = active_plan.get("destination", "") or ""
    dep_check = active_plan.get("departure_city", "") or ""
    t_mode_check = extract_transport_preference(latest_user_query) or active_plan.get("transport_mode", "") or ""

    is_intl_check = is_international_trip(dep_check, dest_check) or is_international_trip("", latest_user_query)

    exec_nodes = ["user_query", "planner_agent", "transport_decision"]
    if is_intl_check:
        exec_nodes.append("flight_agent")
    else:
        if t_mode_check in ["railway", "cheaper"]:
            exec_nodes.append("railway_agent")
        elif t_mode_check in ["flight", "luxury"]:
            exec_nodes.append("flight_agent")
        else:
            exec_nodes.extend(["railway_agent", "flight_agent"])

    exec_nodes.extend(["hotel_agent", "weather_agent", "itinerary_agent", "final_response", "download_builder"])

    # Initialize / Reset visualizer state for pipeline run
    reset_visualizer_for_run(executed_nodes=exec_nodes)
    st.session_state.visualizer_runtime_data = {
        "source": dep_check or "Origin",
        "destination": dest_check or "Destination",
        "budget": active_plan.get("budget_category", "") or active_plan.get("budget", "") or "Flexible",
        "dates": active_plan.get("travel_dates", "") or active_plan.get("duration", "") or "Scheduled",
        "user_query": latest_user_query
    }
    record_node_start("user_query", input_summary=latest_user_query, process_msg="Received user query")
    record_node_complete("user_query", output_summary=latest_user_query, passed_to="planner_agent")

    record_node_start("planner_agent", input_summary=latest_user_query, process_msg="Extracting intent, budget, dates...")
    record_node_complete("planner_agent", output_summary="Plan parameters extracted", passed_to="transport_decision")

    record_node_start("transport_decision", input_summary="Determining transport mode", process_msg="Evaluating flight vs railway preference...")
    record_node_complete("transport_decision", output_summary="Transport route selected", passed_to="flight_agent" if is_intl_check else "flight_agent / railway_agent")

    # Render visualizer modal immediately before backend graph runs!
    render_agent_visualizer(target=viz_placeholder, visible=True)


    with st.chat_message("assistant"):
        st.markdown("---")
        st.markdown(
            "<div class='sec-head'><span>🤖 Agent Pipeline — Live</span></div>",
            unsafe_allow_html=True,
        )
        
        collected = {
            "railway_results": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
        }
        
        # Format conversation history
        graph_messages = []
        for m in st.session_state.chat_messages:
            if m["role"] == "user":
                graph_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                parsed = parse_assistant_message(m["content"])
                assistant_text = parsed.get("itinerary", "")
                if parsed.get("final_response"):
                    assistant_text += "\n" + parsed["final_response"]
                graph_messages.append(AIMessage(content=assistant_text))
                
        config = {"configurable": {"thread_id": st.session_state.current_chat_id}}
        
        state_snapshot = app.get_state(config)
        if state_snapshot.values:
            inputs = {
                "messages": [HumanMessage(content=latest_user_query)],
                "user_query": latest_user_query,
            }
        else:
            inputs = {
                "messages": [HumanMessage(content=latest_user_query)],
                "user_query": latest_user_query,
                "railway_results": "",
                "flight_results": "",
                "hotel_results": "",
                "weather_results": "",
                "itinerary": "",
                "llm_calls": 0,
            }
            
        try:
            current_classification = {}
            for chunk in app.stream(
                inputs,
                config=config,
                stream_mode="updates",
            ):
                for node_name, state_update in chunk.items():
                    # Parse classification if available in railway_agent or flight_agent update
                    if node_name in ["railway_agent", "flight_agent"]:
                        mod_type = state_update.get("modification_type", "{}")
                        try:
                            current_classification = json.loads(mod_type)
                            if current_classification.get("is_international"):
                                if "railway_agent" in st.session_state.visualizer_executed_nodes:
                                    st.session_state.visualizer_executed_nodes = [n for n in st.session_state.visualizer_executed_nodes if n != "railway_agent"]
                                    if "railway_agent" in st.session_state.visualizer_node_states:
                                        st.session_state.visualizer_node_states["railway_agent"]["status"] = "skipped"
                                        st.session_state.visualizer_node_states["railway_agent"]["output"] = "Skipped (international trip)"
                        except Exception:
                            pass
                            
                    # Check execution flags
                    show_node = True
                    if current_classification:
                        req_type = current_classification.get("request_type", "full_plan")
                        if req_type in ["plan_question", "general_conversation"]:
                            show_node = False
                        else:
                            if node_name == "railway_agent":
                                show_node = current_classification.get("needs_railway_agent", True) and not current_classification.get("is_international", False)
                            elif node_name == "flight_agent":
                                show_node = current_classification.get("needs_flight_agent", True)
                            elif node_name == "hotel_agent":
                                show_node = current_classification.get("needs_hotel_agent", True)
                            elif node_name == "weather_agent":
                                show_node = current_classification.get("needs_weather_agent", True)
                            elif node_name == "itinerary_agent":
                                show_node = current_classification.get("needs_itinerary_agent", True)
                                
                    if not show_node:
                        continue
                        
                    icon, label = AGENT_META.get(
                        node_name,
                        ("🔧", node_name),
                    )

                    record_node_start(node_name, input_summary=f"Running {label}", process_msg=f"Executing {label} backend agent logic...")
                    render_agent_visualizer(target=viz_placeholder, visible=True)

                    with st.status(
                        f"{icon}  {label}",
                        state="complete",
                        expanded=True,
                    ):
                        if node_name == "railway_agent":
                            agent_text = state_update.get("railway_results", "")
                            collected["railway_results"] = agent_text
                            render_railway_section(agent_text)
                            record_node_complete("railway_agent", output_summary="Train options & IRCTC links retrieved", passed_to="flight_agent / hotel_agent")

                        elif node_name == "flight_agent":
                            agent_text = state_update.get("flight_results", "")
                            collected["flight_results"] = agent_text
                            render_flight_section(agent_text)
                            record_node_complete("flight_agent", output_summary="Flight options & fares retrieved", passed_to="hotel_agent")

                        elif node_name == "hotel_agent":
                            agent_text = state_update.get("hotel_results", "")
                            collected["hotel_results"] = agent_text
                            dest_for_hotels = ""
                            if state_snapshot and state_snapshot.values and "current_plan" in state_snapshot.values:
                                try:
                                    cp = json.loads(state_snapshot.values["current_plan"])
                                    dest_for_hotels = cp.get("destination", "")
                                except Exception:
                                    pass
                            render_hotel_section(agent_text, destination=dest_for_hotels)
                            record_node_complete("hotel_agent", output_summary="Hotels categorized by budget", passed_to="weather_agent")

                        elif node_name == "weather_agent":
                            agent_text = state_update.get("weather_results", "")
                            collected["weather_results"] = agent_text
                            render_weather_section(agent_text)
                            record_node_complete("weather_agent", output_summary="Weather forecast & packing tips fetched", passed_to="itinerary_agent")

                        elif node_name == "itinerary_agent":
                            agent_text = state_update.get("itinerary", "")
                            collected["itinerary"] = agent_text
                            st.session_state.generated_itinerary = agent_text
                            
                            # If specific day change, render day confirm only
                            if current_classification.get("request_type") in ["specific_day_change", "attraction_add", "attraction_remove"]:
                                disp_text = state_update.get("messages", [AIMessage(content="")])[-1].content
                                render_itinerary_section(disp_text, latest_user_query)
                            else:
                                render_itinerary_section(agent_text, latest_user_query)
                            record_node_complete("itinerary_agent", output_summary="Day-wise travel itinerary built", passed_to="final_response")

                        collected["llm_calls"] = state_update.get(
                            "llm_calls",
                            collected["llm_calls"],
                        )
                        render_agent_visualizer(target=viz_placeholder, visible=True)



                        
            final_snapshot = app.get_state(config)
            classification_str = final_snapshot.values.get("modification_type", "{}") if final_snapshot and final_snapshot.values else "{}"
            try:
                classification = json.loads(classification_str) if classification_str else {}
            except Exception:
                classification = {}
            request_type = classification.get("request_type", "full_plan")
            
            final_messages = final_snapshot.values.get("messages", []) if final_snapshot and final_snapshot.values else []
            if final_messages:
                collected["final_response"] = final_messages[-1].content
                
            current_plan_json = final_snapshot.values.get("current_plan", "{}") if final_snapshot and final_snapshot.values else "{}"
            try:
                current_plan_dict = json.loads(current_plan_json)
            except Exception:
                current_plan_dict = {}

            t_mode = (current_plan_dict.get("transport_mode") or "").lower()
            is_intl_resp = classification.get("is_international") or is_international_trip(current_plan_dict.get("departure_city", ""), current_plan_dict.get("destination", ""))

            if classification.get("needs_missing_info") or (collected.get("final_response") and ("before i plan" in collected["final_response"].lower() or "i just need" in collected["final_response"].lower() or "preferred hotel budget" in collected["final_response"].lower() or "how would you like to travel" in collected["final_response"].lower())):
                collected["displayed_response"] = collected.get("final_response")
                st.session_state.show_visualizer = False
                viz_placeholder.empty()

            elif request_type in ["plan_question", "general_conversation", "specific_question", "comparison", "summary"] and not (collected.get("itinerary") or collected.get("hotel_results") or collected.get("flight_results") or collected.get("railway_results")):
                collected["displayed_response"] = collected.get("final_response") or "I have answered your question."

            else:
                resp_parts = []
                if collected.get("railway_results") and has_display_value(collected["railway_results"]):
                    resp_parts.append(f"### 🚆 Railway Agent\n\n{collected['railway_results']}")
                if collected.get("flight_results") and has_display_value(collected["flight_results"]):
                    resp_parts.append(f"### ✈️ Flight Information\n\n{collected['flight_results']}")
                if collected.get("hotel_results") and has_display_value(collected["hotel_results"]):
                    resp_parts.append(f"### 🏨 Hotel Information\n\n{collected['hotel_results']}")
                if collected.get("weather_results") and has_display_value(collected["weather_results"]):
                    resp_parts.append(f"### 🌦️ Weather Information\n\n{collected['weather_results']}")
                if collected.get("itinerary") and has_display_value(collected["itinerary"]):
                    resp_parts.append(f"### 🗓️ Travel Itinerary\n\n{collected['itinerary']}")

                collected["displayed_response"] = "\n\n---\n\n".join(resp_parts) if resp_parts else (collected.get("final_response") or "")
            
            collected["request_type"] = request_type
            record_node_start("final_response", input_summary="Assembling final travel response", process_msg="Formatting response sections...")
            record_node_complete("final_response", output_summary="Final response assembled", passed_to="download_builder")
            
            save_message(st.session_state.current_chat_id, "assistant", json.dumps(collected))
                
            # Parse target plan details
            current_plan_json = final_snapshot.values.get("current_plan", "{}") if final_snapshot and final_snapshot.values else "{}"
            try:
                current_plan_dict = json.loads(current_plan_json)
            except Exception:
                current_plan_dict = {}
                
            # Persist plan state to PostgreSQL
            save_chat_state(
                st.session_state.current_chat_id,
                current_plan_dict,
                collected,
                collected.get("itinerary", "")
            )
            
            # Sync active session state dictionary
            active_session = st.session_state.chat_sessions[st.session_state.current_chat_id]
            active_session["messages"] = load_messages(st.session_state.current_chat_id)
            active_session["travel_results"] = collected
            active_session["generated_itinerary"] = collected.get("itinerary", "")
            active_session["current_plan"] = current_plan_dict
            
            st.session_state.chat_messages = active_session["messages"]
            st.session_state.travel_results = active_session["travel_results"]
            st.session_state.generated_itinerary = active_session["generated_itinerary"]
            st.session_state.plan_generated = bool(collected.get("itinerary"))
            
            # Save final file (Auto-save)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"travel_plan_{timestamp}.md"
            save_dir = os.path.join(
                os.path.dirname(__file__),
                "travel_plans",
            )
            os.makedirs(save_dir, exist_ok=True)

            # Rebuild download file content dynamically from latest current_plan & travel_results
            c_title = ""
            try:
                chats_list = list_chats()
                match_chat = next((c for c in chats_list if c["chat_id"] == st.session_state.current_chat_id), None)
                if match_chat:
                    c_title = match_chat.get("title", "")
            except Exception:
                pass

            file_content = build_download_document(current_plan_dict, collected, chat_title=c_title)
            filename = get_download_filename(current_plan_dict.get("destination", ""), chat_title=c_title)

            save_dir = os.path.join(os.path.dirname(__file__), "travel_plans")
            os.makedirs(save_dir, exist_ok=True)
            record_download_builder("start", details="Compiling trip plan document...")
            with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as file:
                file.write(file_content)
            record_download_builder("completed", details="Dossier saved successfully!")
            if current_plan_dict:
                st.session_state.visualizer_runtime_data.update({
                    "source": current_plan_dict.get("departure_city") or current_plan_dict.get("source") or st.session_state.visualizer_runtime_data.get("source"),
                    "destination": current_plan_dict.get("destination") or st.session_state.visualizer_runtime_data.get("destination"),
                    "budget": current_plan_dict.get("budget_category") or current_plan_dict.get("budget") or st.session_state.visualizer_runtime_data.get("budget"),
                    "dates": current_plan_dict.get("travel_dates") or current_plan_dict.get("duration") or st.session_state.visualizer_runtime_data.get("dates")
                })
            st.session_state.chat_execution_snapshots[st.session_state.current_chat_id] = copy.deepcopy({
                "node_states": st.session_state.visualizer_node_states,
                "timeline": st.session_state.visualizer_timeline,
                "executed_nodes": st.session_state.visualizer_executed_nodes,
                "overall_status": "completed",
                "runtime_data": st.session_state.get("visualizer_runtime_data", {})
            })
            render_agent_visualizer(target=viz_placeholder, visible=True, mode="live")



            st.session_state.saved_filename = filename
            st.session_state.saved_file_content = file_content
            st.session_state.current_plan_summary = ""
            st.rerun()


        except (BaseExceptionGroup, ExceptionGroup, Exception) as error:
            from mcp_client import log_nested_exception
            print("\n[FRONTEND ERROR] Pipeline execution exception caught:")
            log_nested_exception(error)

            err_msg = str(error)
            if "TaskGroup" in err_msg or "sub-exception" in err_msg or isinstance(error, (BaseExceptionGroup, ExceptionGroup)):
                st.error("One of the external MCP travel services encountered a connection issue. The plan is continuing with available services.")
            elif "429" in err_msg or "rate limit" in err_msg.lower() or "ratelimit" in err_msg.lower():
                st.error("The AI model has temporarily reached its daily rate limit. Please try again in a few moments.")
            elif "Event loop is closed" in err_msg:
                st.error("The async runtime connection was reset. Please retry your request.")
            else:
                st.error(f"The travel plan encountered an issue: {err_msg}")

# Handle Chat Input & Triggers
user_query = st.chat_input("Ask a follow-up or plan a new trip...")

if "trigger_query" in st.session_state and st.session_state.trigger_query:
    user_query = st.session_state.trigger_query
    st.session_state.trigger_query = ""

if user_query:
    user_query_clean = user_query.strip()
    if user_query_clean:
        # If this is a fresh unsaved new chat, convert it to a real saved chat in DB now!
        if st.session_state.current_chat_id is None or st.session_state.get("is_new_chat", False):
            new_chat_id = str(uuid.uuid4())
            new_title = generate_chat_title(user_query_clean)
            create_chat(new_chat_id, new_title)
            save_chat_state(new_chat_id, {}, {}, "")
            
            st.session_state.current_chat_id = new_chat_id
            st.session_state.is_new_chat = False
            st.session_state.chat_sessions[new_chat_id] = create_empty_chat_state()
        else:
            chats = list_chats()
            active_chat = next((c for c in chats if c["chat_id"] == st.session_state.current_chat_id), None)
            if active_chat and active_chat["title"] == "New Chat":
                new_title = generate_chat_title(user_query_clean)
                rename_chat(st.session_state.current_chat_id, new_title)
                
        save_message(st.session_state.current_chat_id, "user", user_query_clean)
        st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"] = load_messages(st.session_state.current_chat_id)
        st.session_state.chat_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]
        
        if "current_plan_summary" in st.session_state:
            st.session_state.current_plan_summary = ""
            
        st.rerun()

# ── Render Floating Visualizer Modal Overlay ──
if st.session_state.get("show_visualizer"):
    viz_mode = st.session_state.get("visualizer_mode", "live")
    snapshot = st.session_state.get("chat_execution_snapshots", {}).get(st.session_state.current_chat_id) if viz_mode == "replay" else None
    render_agent_visualizer(visible=True, snapshot_data=snapshot, mode=viz_mode)




