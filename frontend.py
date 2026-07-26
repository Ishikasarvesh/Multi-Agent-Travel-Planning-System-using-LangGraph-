import os
import uuid
import json
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
import pandas as pd
import re

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
    
    Extract columns: "Hotel", "Location", "Rating", "Price per Night", "Total Price", "Distance".
    Only include columns where data is available. Rating should be a string or number (e.g., 4.5/5 or 4.5 star).
    
    Output JSON format:
    [
        {{"Hotel": "...", "Location": "...", "Rating": "...", "Price per Night": "...", "Total Price": "...", "Distance": "..."}}
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
        "current_weather": {{
            "Temperature": "...",
            "Feels Like": "...",
            "Condition": "...",
            "Humidity": "...",
            "Wind": "...",
            "UV Index": "..."
        }},
        "forecast": [
            {{"Day": "...", "Condition": "...", "Minimum": "...", "Maximum": "...", "Rain Chance": "..."}}
        ],
        "alerts": ["warning or alert text string", "or 'No active weather alerts'"],
        "recommended_clothing": ["item 1", "item 2"],
        "travel_essentials": ["item 1", "item 2"],
        "weather_tip": "..."
    }}
    
    Return ONLY the JSON. Do not include markdown code block syntax or explanations.
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

def render_flight_section(flight_results: str):
    if not flight_results or "unavailable" in flight_results.lower():
        return
    render_section_title("Flight Options")
    df = parse_flights_to_df(flight_results)
    if not df.empty:
        columns_priority = ["Airline", "Flight", "Departure", "Arrival", "Duration", "Stops", "Price"]
        df_cols = [col for col in columns_priority if col in df.columns]
        st.dataframe(df[df_cols], use_container_width=True, hide_index=True)
    else:
        st.markdown(re.split(r"Book on:", flight_results, flags=re.IGNORECASE)[0].strip())
    
    render_subsection_title("Booking Platforms")
    st.markdown("- [MakeMyTrip](https://www.makemytrip.com/)")
    st.markdown("- [Booking.com Flights](https://www.booking.com/flights.html)")
    st.markdown("- [Skyscanner](https://www.skyscanner.net/)")
    st.markdown("- [Google Flights](https://www.google.com/travel/flights)")
    st.markdown("- [EaseMyTrip](https://www.easemytrip.com/)")
    st.markdown("")
    
    tip_text = "Booking 30–60 days in advance usually offers better prices for international trips."
    match = re.search(r"Flight Tip:\s*(.*)", flight_results, re.IGNORECASE)
    if match:
        tip_text = match.group(1).strip()
    render_tip("Flight Tip", tip_text)

def render_hotel_section(hotel_results: str):
    if not hotel_results or "unavailable" in hotel_results.lower():
        return
    render_section_title("Hotel Recommendations")
    df = parse_hotels_to_df(hotel_results)
    if not df.empty:
        columns_priority = ["Hotel", "Location", "Rating", "Price per Night", "Total Price", "Distance"]
        df_cols = [col for col in columns_priority if col in df.columns]
        st.dataframe(df[df_cols], use_container_width=True, hide_index=True)
    else:
        st.markdown(re.split(r"💡 Hotel Tip:", hotel_results, flags=re.IGNORECASE)[0].strip())
        
    tip_text = "Choose hotels near metro or railway stations to reduce local transport costs."
    match = re.search(r"Hotel Tip:\s*(.*)", hotel_results, re.IGNORECASE)
    if match:
        tip_text = match.group(1).strip()
    render_tip("Hotel Tip", tip_text)

def render_weather_section(weather_results: str):
    if not weather_results or "unavailable" in weather_results.lower():
        return
    
    data = parse_weather_to_data(weather_results)
    if not data:
        st.markdown(weather_results)
        return
        
    render_section_title("Weather Information")
    
    render_subsection_title("Current Weather")
    curr = data.get("current_weather", {})
    if curr:
        curr_df = pd.DataFrame([curr])
        st.dataframe(curr_df, use_container_width=True, hide_index=True)
        
    render_subsection_title("Forecast")
    forecast = data.get("forecast", [])
    if forecast:
        fore_df = pd.DataFrame(forecast)
        st.dataframe(fore_df, use_container_width=True, hide_index=True)
        
    render_subsection_title("Weather Alerts")
    alerts = data.get("alerts", [])
    if isinstance(alerts, str):
        alerts = [alerts]
    for alert in alerts:
        alert_lower = alert.lower()
        if "no active" in alert_lower or "no warnings" in alert_lower or "no alert" in alert_lower:
            st.success(alert)
        elif "severe" in alert_lower or "extreme" in alert_lower or "thunderstorm" in alert_lower:
            st.error(alert)
        else:
            st.warning(alert)
            
    render_subsection_title("Recommended Clothing")
    clothing = data.get("recommended_clothing", [])
    if clothing:
        if isinstance(clothing, str):
            st.markdown(clothing)
        else:
            for item in clothing:
                st.markdown(f"- {item}")
                
    render_subsection_title("Travel Essentials")
    essentials = data.get("travel_essentials", [])
    if essentials:
        if isinstance(essentials, str):
            st.markdown(essentials)
        else:
            for item in essentials:
                st.markdown(f"- {item}")
                
    tip_text = data.get("weather_tip", "Schedule outdoor sightseeing before afternoon rain.")
    render_tip("Weather Tip", tip_text)

def render_itinerary_section(itinerary_text: str, user_query: str):
    if not itinerary_text:
        return
    render_section_title("Day-wise Itinerary")
    
    clean_itinerary = re.split(r"💡 Itinerary Tip:", itinerary_text, flags=re.IGNORECASE)[0].strip()
    render_itinerary_with_inline_images(clean_itinerary, user_query)
    
    tip_text = "Visit outdoor attractions early in the morning to avoid crowds and heat."
    match = re.search(r"Itinerary Tip:\s*(.*)", itinerary_text, re.IGNORECASE)
    if match:
        tip_text = match.group(1).strip()
    render_tip("Itinerary Tip", tip_text)

def render_assistant_response(parsed: dict, assoc_user_query: str):
    request_type = parsed.get("request_type", "full_plan")
    
    if request_type == "flight_request":
        render_flight_section(parsed.get("flight_results", ""))
    elif request_type == "hotel_request":
        render_hotel_section(parsed.get("hotel_results", ""))
    elif request_type == "weather_request":
        render_weather_section(parsed.get("weather_results", ""))
    elif request_type in ["duration_change", "budget_change", "itinerary_change"]:
        render_itinerary_section(parsed.get("itinerary", ""), assoc_user_query)
    elif request_type in ["specific_day_change", "attraction_add", "attraction_remove"]:
        disp = parsed.get("displayed_response") or parsed.get("final_response") or ""
        render_itinerary_section(disp, assoc_user_query)
    elif request_type in ["plan_question", "general_conversation"]:
        st.markdown(parsed.get("displayed_response") or parsed.get("final_response") or "")
    else: # full_plan or destination_change or default fallback
        if parsed.get("itinerary") or parsed.get("flight_results"):
            render_flight_section(parsed.get("flight_results", ""))
            st.markdown("---")
            render_hotel_section(parsed.get("hotel_results", ""))
            st.markdown("---")
            render_weather_section(parsed.get("weather_results", ""))
            st.markdown("---")
            render_itinerary_section(parsed.get("itinerary", ""), assoc_user_query)
        else:
            st.markdown(parsed.get("final_response") or "")


# Helper functions defined at the top to avoid NameError/execution order issues
def create_empty_chat_state():
    return {
        "messages": [],
        "current_plan": {},
        "travel_results": {
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
        new_id = str(uuid.uuid4())
        create_chat(new_id, "New Chat")
        save_chat_state(new_id, {}, {}, "")
        activate_chat(new_id)
        st.session_state.current_plan_summary = ""
        st.session_state.saved_filename = ""
        st.session_state.saved_file_content = ""
        st.rerun()

    st.markdown("<div class='sidebar-title'>💬 Recent Chats</div>", unsafe_allow_html=True)
    chats = list_chats()
    
    if not chats:
        default_id = str(uuid.uuid4())
        create_chat(default_id, "New Chat")
        st.session_state.current_chat_id = default_id
        chats = list_chats()
        
    if "current_chat_id" not in st.session_state or st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = chats[0]["chat_id"]

    active_chat_id = st.session_state.current_chat_id

    for c in chats:
        c_id = c["chat_id"]
        c_title = c["title"]
        
        is_active = (c_id == active_chat_id)
        btn_label = f"📍 {c_title}" if is_active else f"📄 {c_title}"
        
        if st.button(btn_label, key=f"chat_select_{c_id}", use_container_width=True):
            activate_chat(c_id)
            st.session_state.current_plan_summary = ""
            
            # Rebuild saved file content from the active session state
            active_session = st.session_state.chat_sessions[c_id]
            parsed = active_session["travel_results"]
            
            if parsed.get("itinerary"):
                latest_user_msgs = [m for m in active_session["messages"] if m["role"] == "user"]
                latest_user_query = latest_user_msgs[-1]["content"] if latest_user_msgs else ""
                timestamp = c["updated_at"].strftime("%Y%m%d_%H%M%S") if c["updated_at"] else "now"
                st.session_state.saved_filename = f"travel_plan_{timestamp}.md"
                st.session_state.saved_file_content = f"""# Travel Plan
**Title:** {c_title}
**Query:** {latest_user_query}
**User ID:** {c_id}

---

## ✈️ Flight Information
{parsed.get('flight_results') or 'N/A'}

---

## 🏨 Hotel Information
{parsed.get('hotel_results') or 'N/A'}

---

## 🌦️ Weather Information
{parsed.get('weather_results') or 'N/A'}

---

## 🗓️ Itinerary
{parsed.get('itinerary') or 'N/A'}

---

## 🧠 Final Travel Plan
{parsed.get('final_response') or 'N/A'}
"""
            else:
                st.session_state.saved_filename = ""
                st.session_state.saved_file_content = ""
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

# Ensure we have active chat loaded
chats = list_chats()
if not chats:
    new_id = str(uuid.uuid4())
    create_chat(new_id, "New Chat")
    save_chat_state(new_id, {}, {}, "")
    chats = list_chats()

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id is None:
    st.session_state.current_chat_id = chats[0]["chat_id"]

current_chat_id = st.session_state.current_chat_id

# Always sync active chat with database on every run
activate_chat(current_chat_id)

# Bind active chat variables to st.session_state keys
active_session = st.session_state.chat_sessions[current_chat_id]
st.session_state.chat_messages = active_session["messages"]
st.session_state.travel_results = active_session["travel_results"]
st.session_state.generated_itinerary = active_session["generated_itinerary"]
st.session_state.plan_generated = bool(st.session_state.travel_results.get("itinerary"))

if "saved_filename" not in st.session_state:
    st.session_state.saved_filename = ""

if "saved_file_content" not in st.session_state:
    st.session_state.saved_file_content = ""

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
    # Show a simple compact header when in chat mode
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.5rem;">
        <span style="font-size: 1.8rem;">✈️</span>
        <h2 style="margin: 0; color: #e8f4ff;">Conversational Travel Assistant</h2>
    </div>
    """, unsafe_allow_html=True)

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
            if parsed.get("itinerary") or parsed.get("flight_results"):
                if idx == len(st.session_state.chat_messages) - 1:
                    if st.session_state.saved_filename and st.session_state.saved_file_content:
                        download_column, info_column = st.columns([1, 3])
                        with download_column:
                            st.download_button(
                                "⬇️ Download Plan",
                                data=st.session_state.saved_file_content,
                                file_name=st.session_state.saved_filename,
                                mime="text/markdown",
                                use_container_width=True,
                            )
                        with info_column:
                            st.markdown(
                                "<div class='save-bar'>"
                                f"📁 Auto-saved → "
                                f"<code>travel_plans/{st.session_state.saved_filename}</code>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    turn_download = f"""# Travel Plan
**Generated:** {msg['created_at'].strftime("%Y-%m-%d %H:%M:%S") if msg.get('created_at') else 'N/A'}
**User ID:** {st.session_state.current_chat_id}

---

## ✈️ Flight Information
{parsed.get('flight_results') or 'N/A'}

---

## 🏨 Hotel Information
{parsed.get('hotel_results') or 'N/A'}

---

## 🌦️ Weather Information
{parsed.get('weather_results') or 'N/A'}

---

## 🗓️ Itinerary
{parsed.get('itinerary') or 'N/A'}
"""
                    col_dl, col_space = st.columns([1, 2])
                    with col_dl:
                        st.download_button(
                            "⬇️ Download Plan",
                            data=turn_download,
                            file_name=f"travel_plan_{idx}.md",
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
    "flight_agent": ("✈️", "Flight Agent"),
    "hotel_agent": ("🏨", "Hotel Agent"),
    "weather_agent": ("🌦️", "Weather Agent"),
    "itinerary_agent": ("🗓️", "Itinerary Agent"),
}

if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
    last_query = st.session_state.chat_messages[-1]["content"]
    
    with st.chat_message("assistant"):
        st.markdown("---")
        st.markdown(
            "<div class='sec-head'><span>🤖 Agent Pipeline — Live</span></div>",
            unsafe_allow_html=True,
        )
        
        collected = {
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
                "messages": [HumanMessage(content=last_query)],
                "user_query": last_query,
            }
        else:
            inputs = {
                "messages": [HumanMessage(content=last_query)],
                "user_query": last_query,
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
                    # Parse classification if available in flight_agent update
                    if node_name == "flight_agent":
                        mod_type = state_update.get("modification_type", "{}")
                        try:
                            current_classification = json.loads(mod_type)
                        except Exception:
                            pass
                            
                    # Check execution flags
                    show_node = True
                    if current_classification:
                        req_type = current_classification.get("request_type", "full_plan")
                        if req_type in ["plan_question", "general_conversation"]:
                            show_node = False
                        else:
                            if node_name == "flight_agent":
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

                    with st.status(
                        f"{icon}  {label}",
                        state="complete",
                        expanded=True,
                    ):
                        if node_name == "flight_agent":
                            agent_text = state_update.get("flight_results", "")
                            collected["flight_results"] = agent_text
                            render_flight_section(agent_text)

                        elif node_name == "hotel_agent":
                            agent_text = state_update.get("hotel_results", "")
                            collected["hotel_results"] = agent_text
                            render_hotel_section(agent_text)

                        elif node_name == "weather_agent":
                            agent_text = state_update.get("weather_results", "")
                            collected["weather_results"] = agent_text
                            render_weather_section(agent_text)

                        elif node_name == "itinerary_agent":
                            agent_text = state_update.get("itinerary", "")
                            collected["itinerary"] = agent_text
                            st.session_state.generated_itinerary = agent_text
                            
                            # If specific day change, render day confirm only
                            if current_classification.get("request_type") in ["specific_day_change", "attraction_add", "attraction_remove"]:
                                disp_text = state_update.get("messages", [AIMessage(content="")])[-1].content
                                render_itinerary_section(disp_text, user_query)
                            else:
                                render_itinerary_section(agent_text, user_query)

                        collected["llm_calls"] = state_update.get(
                            "llm_calls",
                            collected["llm_calls"],
                        )
                        
            final_snapshot = app.get_state(config)
            classification_str = final_snapshot.values.get("modification_type", "{}") if final_snapshot and final_snapshot.values else "{}"
            classification = json.loads(classification_str)
            request_type = classification.get("request_type", "full_plan")
            
            final_messages = final_snapshot.values.get("messages", []) if final_snapshot and final_snapshot.values else []
            if final_messages:
                collected["final_response"] = final_messages[-1].content
                
            if request_type == "flight_request":
                collected["displayed_response"] = f"### ✈️ Updated Flight Information\n\n{collected['flight_results']}"
            elif request_type == "hotel_request":
                collected["displayed_response"] = f"### 🏨 Updated Hotel Information\n\n{collected['hotel_results']}"
            elif request_type == "weather_request":
                collected["displayed_response"] = f"### 🌦️ Weather Information\n\n{collected['weather_results']}"
            elif request_type in ["specific_day_change", "attraction_add", "attraction_remove"]:
                collected["displayed_response"] = collected.get("final_response") or "I have modified the day in your itinerary."
            elif request_type in ["duration_change", "budget_change", "itinerary_change"]:
                collected["displayed_response"] = f"### 🗓️ Updated Itinerary\n\n{collected['itinerary']}"
            elif request_type in ["plan_question", "general_conversation"]:
                collected["displayed_response"] = collected.get("final_response") or "I have answered your question."
            else:
                collected["displayed_response"] = f"""### ✈️ Flight Information
{collected['flight_results'] or 'N/A'}

---

### 🏨 Hotel Information
{collected['hotel_results'] or 'N/A'}

---

### 🌦️ Weather Information
{collected['weather_results'] or 'N/A'}

---

### 🗓️ Travel Itinerary
{collected['itinerary'] or 'N/A'}
"""
            
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

            file_content = f"""# Travel Plan
**Query:** {last_query}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**User ID:** {st.session_state.current_chat_id}

---

## ✈️ Flight Information
{collected['flight_results'] or 'N/A'}

---

## 🏨 Hotel Information
{collected['hotel_results'] or 'N/A'}

---

## 🌦️ Weather Information
{collected['weather_results'] or 'N/A'}

---

## 🗓️ Itinerary
{collected['itinerary'] or 'N/A'}

---

## 🧠 Final Travel Plan
{collected['final_response'] or 'N/A'}

---
*LLM Calls: {collected['llm_calls']}*
"""

            with open(
                os.path.join(save_dir, filename),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(file_content)

            st.session_state.saved_filename = filename
            st.session_state.saved_file_content = file_content
            
            st.session_state.current_plan_summary = ""
            st.rerun()

        except RuntimeError as error:
            if "Event loop is closed" in str(error):
                st.error(
                    "The async runtime was reset unexpectedly. "
                    "Please retry the request."
                )
            else:
                st.error(
                    "The travel plan could not be completed. "
                    f"Error: {error}"
                )
        except Exception as error:
            st.error(
                "The travel plan could not be completed. "
                f"Error: {error}"
            )

# Handle Chat Input & Triggers
user_query = st.chat_input("Ask a follow-up or plan a new trip...")

if "trigger_query" in st.session_state and st.session_state.trigger_query:
    user_query = st.session_state.trigger_query
    st.session_state.trigger_query = ""

if user_query:
    user_query_clean = user_query.strip()
    if user_query_clean:
        chats = list_chats()
        active_chat = next((c for c in chats if c["chat_id"] == st.session_state.current_chat_id), None)
        
        if active_chat and active_chat["title"] == "New Chat":
            new_title = generate_chat_title(user_query_clean)
            rename_chat(st.session_state.current_chat_id, new_title)
            
        save_message(st.session_state.current_chat_id, "user", user_query_clean)
        st.session_state.chat_messages = load_messages(st.session_state.current_chat_id)
        
        if "current_plan_summary" in st.session_state:
            st.session_state.current_plan_summary = ""
            
        st.rerun()


