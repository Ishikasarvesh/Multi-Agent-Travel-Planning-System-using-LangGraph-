# LangGraph Multi-Agent Travel Booking System with Long-Term Memory

import os
import json
import re
import urllib.parse
from typing import TypedDict, Annotated
import operator
import asyncio
import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    get_airports,
    get_airlines,
    aviation_mcp_call,extract_destination,forecast_mcp_search,weather_mcp_search,railway_mcp_search,
    run_async_safely
)


from dotenv import load_dotenv
load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# State
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    railway_results: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str
    # ── Conversational chat state ──
    current_plan: str
    conversation_summary: str
    last_user_request: str
    modification_type: str

def get_latest_user_message(messages) -> str:
    if not messages:
        return ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content.strip()
        elif isinstance(message, dict) and message.get("role") == "user":
            return message.get("content", "").strip()
    return ""

# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""

def extract_hotel_budget_category(text: str) -> str:
    if not text:
        return ""
    txt = text.lower()
    
    if any(k in txt for k in ["select all", "select_all", "show all", "show me all", "all categories", "all budgets", "every range", "cheap to luxury", "low to high"]):
        return "all"
    if re.search(r'\ball\b', txt) and any(w in txt for w in ["category", "categories", "budget", "hotel", "range", "select"]):
        return "all"
    if txt.strip() in ["all", "show all", "show me all", "all categories", "select all", "select_all"]:
        return "all"
        
    if any(k in txt for k in ["luxury", "premium", "expensive", "five star", "5 star", "high budget"]):
        return "luxury"
        
    if any(k in txt for k in ["medium", "moderate", "mid-range", "mid range", "midrange", "standard", "comfortable"]):
        return "mid-range"
        
    if any(k in txt for k in ["cheap", "cheapest", "affordable", "low budget", "economical", "budget"]):
        return "budget"
        
    return ""

def is_international_trip(departure_city: str, destination: str) -> bool:
    """
    Determines whether a trip is international or domestic.
    If destination is international (e.g. Japan, Dubai, London, Singapore, Thailand), returns True.
    For domestic trips within India (e.g. Mumbai -> Jaipur, Delhi -> Goa), returns False.
    """
    dep = (departure_city or "").strip().lower()
    dest = (destination or "").strip().lower()
    
    if not dest:
        return False
        
    intl_locations = [
        "japan", "tokyo", "osaka", "kyoto", "dubai", "uae", "united arab emirates",
        "london", "uk", "england", "united kingdom", "singapore", "thailand", "bangkok",
        "phuket", "usa", "us", "united states", "america", "new york", "san francisco",
        "los angeles", "chicago", "paris", "france", "italy", "rome", "switzerland",
        "zurich", "germany", "berlin", "europe", "australia", "sydney", "melbourne",
        "canada", "toronto", "vancouver", "maldives", "vietnam", "hanoi", "da nang",
        "ho chi minh", "malaysia", "kuala lumpur", "turkey", "istanbul", "egypt",
        "cairo", "south korea", "seoul", "korea", "china", "beijing", "shanghai",
        "hong kong", "nepal", "kathmandu", "sri lanka", "colombo", "bali", "indonesia",
        "netherlands", "amsterdam", "spain", "barcelona", "madrid", "greece", "athens",
        "new zealand", "auckland", "philippines", "manila", "qatar", "doha", "oman",
        "muscat", "saudi arabia", "riyadh", "jeddah", "kenya", "nairobi", "south africa",
        "cape town", "brazil", "mauritius", "seychelles"
    ]
    
    dom_locations = [
        "mumbai", "delhi", "new delhi", "jaipur", "goa", "pune", "bangalore", "bengaluru",
        "chennai", "kolkata", "hyderabad", "ahmedabad", "udaipur", "agra", "kerala",
        "shimla", "manali", "kashmir", "srinagar", "varanasi", "amritsar", "chandigarh",
        "lucknow", "kochi", "cochin", "trivandrum", "thiruvananthapuram", "mysuru", "mysore",
        "ooty", "rishikesh", "haridwar", "darjeeling", "gangtok", "guwahati", "patna",
        "bhopal", "indore", "nagpur", "surat", "vadodara", "nashik", "aurangabad",
        "coimbatore", "madurai", "visakhapatnam", "vizag", "bhubaneswar", "ranchi",
        "dehradun", "jodhpur", "jaisalmer", "bikaner", "pushkar", "alleppey", "munnar",
        "wayanad", "pondicherry", "puducherry", "kodaikanal", "port blair", "shillong",
        "ladakh", "leh", "kasol", "spiti", "nainital", "mussoorie", "coorg", "gokarna",
        "hampi", "rameshwaram", "kanyakumari", "tirupati", "shirdi", "ayodhya", "mathura",
        "vrindavan", "mount abu", "ranthambore", "kaziranga", "meghalaya", "assam", "sikkim",
        "himachal", "uttarakhand", "rajasthan", "gujarat", "maharashtra", "karnataka",
        "tamil nadu", "andhra pradesh", "telangana", "west bengal", "odisha", "bihar",
        "punjab", "haryana", "uttar pradesh", "madhya pradesh", "jammu", "india"
    ]

    for loc in intl_locations:
        if re.search(r'\b' + re.escape(loc) + r'\b', dest):
            return True

    for loc in dom_locations:
        if re.search(r'\b' + re.escape(loc) + r'\b', dest):
            return False

    return False

def extract_transport_preference(text: str) -> str:
    if not text:
        return ""
    txt = text.lower()
    
    if any(k in txt for k in [
        "select all", "compare both", "both train and flight", "both flight and train", 
        "train and flight", "flight and train", "flight and railway", "railway and flight",
        "show all", "compare all", "all transport", "all options", "both transport"
    ]) or re.search(r'\b(all|both|select all)\b', txt):
        return "select_all"

    if any(k in txt for k in [
        "luxury", "premium", "flight", "flights", "by air", "by flight", "plane", "airplane",
        "luxury (flight)"
    ]) or re.search(r'\b(luxury|flight|air)\b', txt):
        return "luxury"

    if any(k in txt for k in [
        "cheaper", "cheap", "cheapest", "budget", "affordable", "economical",
        "train", "trains", "railway", "railways", "by train", "by rail", "by railway",
        "cheaper (train)"
    ]) or re.search(r'\b(cheaper|cheap|train|railway)\b', txt):
        return "cheaper"

    return ""

def extract_destination_from_text(text: str) -> str:
    if not text:
        return ""
    txt = text.strip()
    
    if re.search(r'\b(kerela|kerala)\b', txt, re.IGNORECASE):
        return "Kerala"
    if re.search(r'\b(goa)\b', txt, re.IGNORECASE):
        return "Goa"
    if re.search(r'\b(jaipur)\b', txt, re.IGNORECASE):
        return "Jaipur"
    if re.search(r'\b(tokyo)\b', txt, re.IGNORECASE):
        return "Tokyo"
    if re.search(r'\b(dubai)\b', txt, re.IGNORECASE):
        return "Dubai"
    if re.search(r'\b(paris)\b', txt, re.IGNORECASE):
        return "Paris"
    if re.search(r'\b(london)\b', txt, re.IGNORECASE):
        return "London"
    if re.search(r'\b(singapore)\b', txt, re.IGNORECASE):
        return "Singapore"
    if re.search(r'\b(thailand|bangkok)\b', txt, re.IGNORECASE):
        return "Thailand"

    match = re.search(r'(?:plan a trip to|trip to|visit|travel to)\s+([A-Za-z\s]+?)(?:from|\bfor\b|\bin\b|\bwith\b|\bby\b|\b\d|\b$)', txt, re.IGNORECASE)
    if match:
        found = match.group(1).strip()
        if len(found) > 2 and found.lower() not in ["a", "the", "my", "our"]:
            return found.title()

    return ""

def extract_departure_city_from_text(text: str, destination: str = "") -> str:
    if not text:
        return ""
    txt = text.strip()
    dest_clean = (destination or "").strip().lower()
    
    match = re.search(r'\bfrom\s+([A-Za-z\s]+?)(?:to|\bfor\b|\bin\b|\bwith\b|\bby\b|\b\d|,|\b$)', txt, re.IGNORECASE)
    if match:
        found = match.group(1).strip()
        if len(found) > 2 and found.lower() not in ["a", "the", "my"]:
            if not dest_clean or found.lower() != dest_clean:
                return found.title()

    match_compact = re.search(r'^\s*([A-Za-z\s]+?)\s*,\s*\d+\s*(?:day|days)?\s*$', txt, re.IGNORECASE)
    if match_compact:
        found = match_compact.group(1).strip()
        if len(found) > 2:
            if not dest_clean or found.lower() != dest_clean:
                return found.title()

    known_cities = ["mumbai", "delhi", "new delhi", "bangalore", "bengaluru", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur"]
    cleaned_txt = re.sub(r'[^\w\s]', '', txt).strip().lower()
    for c in known_cities:
        if c == cleaned_txt or re.search(r'^\s*' + re.escape(c) + r'\s*$', txt, re.IGNORECASE):
            if not dest_clean or c != dest_clean:
                return c.title()

    return ""

def extract_duration_from_text(text: str) -> str:
    if not text:
        return ""
    txt = text.strip()
    match = re.search(r'\b(\d+)\s*-?\s*(?:day|days|d)\b', txt, re.IGNORECASE)
    if match:
        return match.group(1)
    
    match_compact = re.search(r'^\s*[A-Za-z\s]+?,\s*(\d+)\s*-?\s*(?:day|days)?\s*$', txt, re.IGNORECASE)
    if match_compact:
        return match_compact.group(1)

    return ""

def check_missing_trip_info(plan: dict) -> list:
    missing = []
    dest = str(plan.get("destination") or "").strip()
    if not dest or dest.lower() in ["n/a", "none", "null", "specified in query", ""]:
        missing.append("Destination")
        
    dep = str(plan.get("departure_city") or "").strip()
    if not dep or dep.lower() in ["n/a", "none", "null", "specified in query", ""]:
        missing.append("Departure City / Town")
        
    dur = str(plan.get("duration") or "").strip()
    if not dur or dur.lower() in ["n/a", "none", "null", "specified in query", ""]:
        missing.append("Trip Duration")
        
    cat = str(plan.get("hotel_budget_category") or "").strip()
    if not cat or cat.lower() in ["n/a", "none", "null", "specified in query", ""]:
        missing.append("Hotel Budget")
        
    return missing

def format_missing_info_question(missing_items: list) -> str:
    if not missing_items:
        return ""
    if missing_items == ["Hotel Budget"]:
        return "Great! I only need your hotel budget preference: Cheaper, Mid-range, Luxury, or Select All."

    if missing_items == ["Departure City / Town"]:
        return "Great! I just need your departure city (e.g., Mumbai, Delhi)."

    if missing_items == ["Trip Duration"]:
        return "Great! I just need your trip duration in days (e.g., 7 days)."

    if set(missing_items) == {"Departure City / Town", "Trip Duration"}:
        return "Great! I just need your departure city and trip duration (e.g., Mumbai, 7 days)."

    lines = ["Great! I just need:\n"]
    for item in missing_items:
        if item == "Hotel Budget":
            lines.append("• Hotel Budget (Cheaper / Mid-range / Luxury / Select All)")
        else:
            lines.append(f"• {item}")
        
    return "\n".join(lines)

UPDATE_PLAN_PROMPT = """
You are a state manager for a conversational travel planning assistant.
Analyze the conversation history, the previous version of the travel plan, and the latest user request to output the updated travel plan parameters and request classification.

CRITICAL RULES:
1. NEVER invent, guess, or assume default values for fields like departure_city, duration, budget, or hotel_budget_category.
2. If the user has NOT explicitly provided a departure city, set "departure_city" to "" (empty string). Do NOT default departure_city to Mumbai or any other city!
3. If the user has NOT explicitly provided a trip duration, set "duration" to "" (empty string). Do NOT default to 7 days!
4. If the user has NOT explicitly provided a hotel budget category, set "hotel_budget_category" to "" (empty string). Do NOT default to Luxury!

Previous Travel Plan:
{previous_plan}

Conversation History:
{history}

Latest User Request:
{latest_request}

Your job is to:
1. Update the travel plan parameters based on the latest user request (destination, duration, budget, departure_city, cities, hotel_preferences, flight_preferences, transport_mode). Preserve previous values unless modified.
   - transport_mode can be "railway", "flight", or "all".
2. Classify the user's request into one of these types:
   - "full_plan" / "complete_trip": A request to plan a complete trip from scratch or rebuild a trip.
   - "destination_change": User changes primary destination city/country.
   - "source_change": User changes departure city.
   - "duration_change": User changes only the trip duration (e.g. "make it 7 days").
   - "budget_change": User changes overall budget or spending limit.
   - "transport_change": User switches transport mode (e.g. "show flights instead", "show train instead", "compare train and flight").
   - "hotel_change" / "hotel_request": User changes hotel preferences/category or requests hotels (e.g. "show luxury hotels").
   - "flight_change" / "flight_request": User changes flight preferences or requests flights (e.g. "find evening flights").
   - "railway_change" / "railway_request": User requests railway/train details.
   - "weather_question": User asks about weather conditions (e.g. "will it rain on Day 3?").
   - "itinerary_change": User requests general itinerary modifications.
   - "specific_day_change": User modifies a specific day's activities.
   - "attraction_add": User adds an attraction.
   - "attraction_remove": User removes an attraction.
   - "comparison": User asks to compare specific options.
   - "specific_question" / "plan_question": User asks a direct question.
   - "summary": User asks to summarize the trip.
   - "general_conversation": General greeting or conversation.

Ensure the output is a JSON object matching this structure:
{{
    "updated_plan": {{
        "destination": "country or region, e.g. Japan",
        "duration": "number of days, e.g. 5, else null",
        "budget": "budget description, e.g. ₹2.5 lakhs, else null",
        "departure_city": "city/airport, e.g. Mumbai, else null",
        "departure_date": "YYYY-MM-DD or departure date string if mentioned by user, else null",
        "cities": ["list", "of", "cities"],
        "hotel_preferences": "hotel preferences description",
        "hotel_budget_category": "budget / mid-range / luxury / all, else null",
        "flight_preferences": "flight preferences description",
        "transport_mode": "railway / flight / all"
    }},
    "classification": {{
        "request_type": "one of the types listed above",
        "target_day": integer or null,
        "requested_change": "brief summary of request"
    }}
}}

Return ONLY the JSON. Do not include markdown code block syntax (like ```json), explanations or text.
"""

def update_travel_plan_state(messages, user_query, previous_plan_str):
    default_plan = {
        "destination": "",
        "duration": "",
        "budget": "",
        "departure_city": "",
        "departure_date": "",
        "cities": [],
        "hotel_preferences": "",
        "hotel_budget_category": "",
        "flight_preferences": "",
        "transport_mode": "",
        "railway_results": "",
        "flight_results": ""
    }
    
    default_classification = {
        "request_type": "full_plan",
        "target_day": None,
        "requested_change": "",
        "needs_railway_agent": True,
        "needs_flight_agent": True,
        "needs_hotel_agent": True,
        "needs_weather_agent": True,
        "needs_itinerary_agent": True,
        "show_full_plan": True,
        "needs_missing_info": False,
        "missing_info_prompt": ""
    }
    
    if previous_plan_str:
        try:
            old_plan = json.loads(previous_plan_str)
        except Exception:
            old_plan = default_plan
    else:
        old_plan = default_plan

    # Format history
    history_str = ""
    for m in messages[:-1]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content
        if len(content) > 500:
            content = content[:500] + "..."
        history_str += f"{role}: {content}\n"

    # Call LLM to update plan
    try:
        prompt = UPDATE_PLAN_PROMPT.format(
            previous_plan=json.dumps(old_plan, indent=2),
            history=history_str,
            latest_request=user_query
        )
    except KeyError as exc:
        print(f"[ERROR] Prompt formatting KeyError in UPDATE_PLAN_PROMPT: {exc}")
        raise exc

    new_plan = old_plan.copy()
    classification = default_classification.copy()

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        cleaned_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                updated_plan_dict = data.get("updated_plan")
                if isinstance(updated_plan_dict, dict):
                    new_plan.update(updated_plan_dict)
                class_dict = data.get("classification")
                if isinstance(class_dict, dict):
                    classification.update(class_dict)
    except Exception as e:
        print(f"[ERROR] Plan update LLM invoke/parse error: {e}")

    # Explicit multi-turn detail extraction check from history & user_query
    user_history_text = " ".join([m.content for m in messages if isinstance(m, HumanMessage)]) + " " + user_query

    ext_dest = extract_destination_from_text(user_query) or extract_destination_from_text(user_history_text)
    ext_dep = extract_departure_city_from_text(user_query, ext_dest or old_plan.get("destination", "")) or extract_departure_city_from_text(user_history_text, ext_dest or old_plan.get("destination", ""))
    ext_dur = extract_duration_from_text(user_query) or extract_duration_from_text(user_history_text)
    ext_cat = extract_hotel_budget_category(user_query) or extract_hotel_budget_category(user_history_text)
    ext_trans = extract_transport_preference(user_query) or extract_transport_preference(user_history_text)

    # Merge rules: Extracted > New Plan > Old Plan
    if ext_dest:
        new_plan["destination"] = ext_dest
    elif not new_plan.get("destination") or str(new_plan.get("destination")).lower() in ["n/a", "none", "null", "specified in query", ""]:
        new_plan["destination"] = old_plan.get("destination", "")

    if ext_dep:
        new_plan["departure_city"] = ext_dep
    elif not new_plan.get("departure_city") or str(new_plan.get("departure_city")).lower() in ["n/a", "none", "null", "specified in query", ""]:
        new_plan["departure_city"] = old_plan.get("departure_city", "")

    if new_plan.get("departure_city") and new_plan.get("destination") and str(new_plan["departure_city"]).strip().lower() == str(new_plan["destination"]).strip().lower():
        new_plan["departure_city"] = ""

    if ext_dur:
        new_plan["duration"] = ext_dur
    elif not new_plan.get("duration") or str(new_plan.get("duration")).lower() in ["n/a", "none", "null", "specified in query", ""]:
        new_plan["duration"] = old_plan.get("duration", "")

    if ext_cat:
        new_plan["hotel_budget_category"] = ext_cat
    elif not new_plan.get("hotel_budget_category") or str(new_plan.get("hotel_budget_category")).lower() in ["n/a", "none", "null", "specified in query", ""]:
        new_plan["hotel_budget_category"] = old_plan.get("hotel_budget_category", "")

    if ext_trans:
        new_plan["transport_mode"] = ext_trans
    elif not new_plan.get("transport_mode") or str(new_plan.get("transport_mode")).lower() in ["n/a", "none", "null", "specified in query", ""]:
        new_plan["transport_mode"] = old_plan.get("transport_mode", "")
    else:
        cat = (new_plan.get("hotel_budget_category") or "").lower()
        if cat in ["budget", "cheaper"]:
            new_plan["transport_mode"] = "cheaper"
        elif cat == "luxury":
            new_plan["transport_mode"] = "luxury"
        elif cat in ["all", "mid-range", "select all", "select_all"]:
            new_plan["transport_mode"] = "select_all"

    t_mode = (new_plan.get("transport_mode") or "cheaper").lower()
    if t_mode in ["railway", "cheap", "cheaper"]:
        t_mode = "cheaper"
    elif t_mode in ["flight", "luxury", "air"]:
        t_mode = "luxury"
    elif t_mode in ["all", "both", "select all", "select_all"]:
        t_mode = "select_all"
    new_plan["transport_mode"] = t_mode

    # Check focused question pattern
    query_lower = user_query.lower()
    is_q = bool(re.search(r'\b(which|what|where|how|will it|is there|any|can you tell|show me which)\b', query_lower) or "?" in user_query)
    is_mod = bool(re.search(r'\b(change|replace|add|remove|update|switch|make it|modify|plan a|book a)\b', query_lower))

    if is_q and not is_mod:
        if "hotel" in query_lower or "stay" in query_lower or "resort" in query_lower:
            classification["request_type"] = "hotel_question"
        elif "flight" in query_lower or "airline" in query_lower or "airport" in query_lower:
            classification["request_type"] = "flight_question"
        elif "train" in query_lower or "railway" in query_lower or "station" in query_lower:
            classification["request_type"] = "railway_question"
        elif "weather" in query_lower or "rain" in query_lower or "temp" in query_lower:
            classification["request_type"] = "weather_question"
        else:
            classification["request_type"] = "focused_question"

    req_type = classification.get("request_type", "full_plan")
    is_intl = is_international_trip(new_plan.get("departure_city", ""), new_plan.get("destination", ""))
    classification["is_international"] = is_intl

    dep_country = "India"
    dest_country = "Overseas / International" if is_intl else "India"
    trip_type = "International" if is_intl else "Domestic"

    if is_intl:
        selected_agents = ["flight_agent"]
    else:
        if t_mode == "cheaper":
            selected_agents = ["railway_agent"]
        elif t_mode == "luxury":
            selected_agents = ["flight_agent"]
        else:
            selected_agents = ["railway_agent", "flight_agent"]

    print("\n--- TRANSPORT ROUTING DECISION ---")
    print(f"Source country: {dep_country}")
    print(f"Destination country: {dest_country}")
    print(f"Trip type: {trip_type}")
    print(f"Transport preference: {t_mode}")
    print(f"Selected transport agents: {selected_agents}")
    print("-----------------------------------\n")

    missing_items = check_missing_trip_info(new_plan)
    old_missing = check_missing_trip_info(old_plan)

    if missing_items:
        classification["needs_missing_info"] = True
        classification["missing_info_prompt"] = format_missing_info_question(missing_items)
        classification["needs_railway_agent"] = False
        classification["needs_flight_agent"] = False
        classification["needs_hotel_agent"] = False
        classification["needs_weather_agent"] = False
        classification["needs_itinerary_agent"] = True
    else:
        classification["needs_missing_info"] = False
        classification["missing_info_prompt"] = ""
        
        # If completing a missing-info collection workflow or running full plan/changes:
        if req_type in ["full_plan", "complete_trip", "destination_change", "source_change", "duration_change", "budget_change", "hotel_change"] or old_missing:
            if is_intl:
                classification["needs_railway_agent"] = False
                classification["needs_flight_agent"] = True
            else:
                classification["needs_railway_agent"] = t_mode in ["cheaper", "select_all", "railway", "all"]
                classification["needs_flight_agent"] = t_mode in ["luxury", "select_all", "flight", "all"]
            classification["needs_hotel_agent"] = True
            classification["needs_weather_agent"] = True
            classification["needs_itinerary_agent"] = True
        elif req_type == "transport_change":
            if is_intl:
                classification["needs_railway_agent"] = False
                classification["needs_flight_agent"] = True
            else:
                classification["needs_railway_agent"] = t_mode in ["cheaper", "select_all"]
                classification["needs_flight_agent"] = t_mode in ["luxury", "select_all"]
            classification["needs_hotel_agent"] = False
            classification["needs_weather_agent"] = False
            classification["needs_itinerary_agent"] = False
        elif req_type in ["summary", "specific_question", "plan_question", "comparison", "weather_question", "general_conversation"]:
            classification["needs_railway_agent"] = False
            classification["needs_flight_agent"] = False
            classification["needs_hotel_agent"] = False
            classification["needs_weather_agent"] = False
            classification["needs_itinerary_agent"] = True

    if is_intl:
        classification["needs_railway_agent"] = False
        classification["needs_flight_agent"] = True

    classification = {
        "request_type": classification.get("request_type", "full_plan"),
        "target_day": classification.get("target_day", None),
        "requested_change": classification.get("requested_change", ""),
        "is_international": is_intl,
        "needs_railway_agent": False if is_intl else bool(classification.get("needs_railway_agent", t_mode in ["cheaper", "select_all"])),
        "needs_flight_agent": bool(classification.get("needs_flight_agent", t_mode in ["luxury", "select_all"] or is_intl)),
        "needs_hotel_agent": bool(classification.get("needs_hotel_agent", True)),
        "needs_weather_agent": bool(classification.get("needs_weather_agent", True)),
        "needs_itinerary_agent": bool(classification.get("needs_itinerary_agent", True)),
        "show_full_plan": bool(classification.get("show_full_plan", True)),
        "needs_missing_info": bool(classification.get("needs_missing_info", False)),
        "missing_info_prompt": classification.get("missing_info_prompt", "")
    }

    return new_plan, classification

def railway_agent(state: TravelState):
    print("\nRailway Agent Started\n")

    messages = state.get("messages", [])
    query = (
        state.get("user_query", "").strip()
        or get_latest_user_message(messages)
    )
    previous_plan_str = state.get("current_plan", "")

    new_plan, classification = update_travel_plan_state(messages, query, previous_plan_str)
    
    rerun = classification.get("needs_railway_agent", True)

    if not rerun or classification.get("needs_missing_info"):
        print("\nRailway Agent Completed (skipped/kept)\n")
        return {
            "railway_results": state.get("railway_results", "") or new_plan.get("railway_results", ""),
            "messages": [
                AIMessage(
                    content="Railway recommendations kept from previous plan."
                )
            ],
            "llm_calls": state.get("llm_calls", 0),
            "modification_type": json.dumps(classification),
            "current_plan": json.dumps(new_plan)
        }

    try:
        source = new_plan.get("departure_city", "Source")
        destination = new_plan.get("destination", "Destination")
        
        rail_data_raw = run_async_safely(
            railway_mcp_search(source, destination)
        )

        src_code = rail_data_raw.get("source_code", "BCT") if isinstance(rail_data_raw, dict) else "BCT"
        dst_code = rail_data_raw.get("destination_code", "JP") if isinstance(rail_data_raw, dict) else "JP"
        confirmtkt_url = f"https://www.confirmtkt.com/rly-search?from={urllib.parse.quote_plus(src_code)}&to={urllib.parse.quote_plus(dst_code)}"

        prompt = f"""
You are an expert railway and train travel specialist.

User Query: {query}
Source City/Station: {source}
Destination City/Station: {destination}

Raw Railway Data & Search Snippets:
{json.dumps(rail_data_raw, indent=2) if isinstance(rail_data_raw, dict) else str(rail_data_raw)[:3000]}

Generate structured railway travel options between {source} and {destination}.
Include 3 to 5 recommended train options (such as Vande Bharat Express, Rajdhani Express, Tejas, Superfast trains).
For each train, present these details clearly:
- Train Name
- Train Number
- Departure Station & Code
- Arrival Station & Code
- Departure Time
- Arrival Time
- Duration
- Train Type (e.g. Express / Vande Bharat / Rajdhani / Shatabdi / Superfast)
- Available Classes (e.g. 1A, 2A, 3A, SL, CC, EC)
- Estimated Fare (e.g. ₹550 - ₹2,450)
- Running Days (e.g. Daily / Mon, Wed, Fri)

Return concise, accurate railway travel guidance in clean Markdown.
"""
        response = llm.invoke([
            SystemMessage(content="You are an expert Indian Railways travel planner."),
            HumanMessage(content=prompt)
        ])

        train_data = response.content.strip()

        links_and_tip = f"""

### Booking Platforms

- [IRCTC](https://www.irctc.co.in/nget/train-search)
- [ConfirmTkt]({confirmtkt_url})
- [RailYatri](https://www.railyatri.in/train-tickets)

**Railway Tip:** Book AC coaches early during peak travel seasons as berths fill quickly.
"""
        train_data += links_and_tip

    except Exception as e:
        train_data = f"Railway information unavailable: {str(e)}"

    new_plan["railway_results"] = train_data
    new_plan_str = json.dumps(new_plan)
    changes_str = json.dumps(classification)

    print("\nRailway Agent Completed\n")
    return {
        "railway_results": train_data,
        "messages": [
            AIMessage(
                content=f"Fetched railway options for {new_plan.get('departure_city', 'origin')} to {new_plan.get('destination', 'destination')}."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "modification_type": changes_str,
        "current_plan": new_plan_str
    }

def flight_agent(state: TravelState):
    print("\nFlight Agent Started\n")

    messages = state.get("messages", [])
    query = (
        state.get("user_query", "").strip()
        or get_latest_user_message(messages)
    )
    previous_plan_str = state.get("current_plan", "")

    # Run state updater to get new plan and detected changes
    new_plan, classification = update_travel_plan_state(messages, query, previous_plan_str)
    
    new_plan_str = json.dumps(new_plan)
    changes_str = json.dumps(classification)

    rerun = classification.get("needs_flight_agent", True)

    if not rerun or classification.get("needs_missing_info"):
        print("\nFlight Agent Completed (skipped/kept)\n")
        return {
            "flight_results": state.get("flight_results", "") or new_plan.get("flight_results", ""),
            "messages": [
                AIMessage(
                    content="Flight recommendations kept from previous plan."
                )
            ],
            "llm_calls": state.get("llm_calls", 0),
            "modification_type": changes_str,
            "current_plan": new_plan_str
        }

    try:
        airports = run_async_safely(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = run_async_safely(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        history_str = ""
        for m in messages[:-1]:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            history_str += f"{role}: {m.content}\n"

        prompt = f"""
You are a travel flight expert.

Conversation History:
{history_str}

Latest User Query:
{query}

Target Trip Plan Context:
Destination: {new_plan.get('destination', 'N/A')}
Duration: {new_plan.get('duration', 'N/A')} days
Budget: {new_plan.get('budget', 'N/A')}
Departure City: {new_plan.get('departure_city', 'N/A')}
Cities to visit: {', '.join(new_plan.get('cities', [])) if isinstance(new_plan.get('cities'), list) else new_plan.get('cities', 'N/A')}
Flight Preferences: {new_plan.get('flight_preferences', 'N/A')}

Airport Information:
{str(airports)[:3000]}

Airline Information:
{str(airlines)[:3000]}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""

        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content
        
        # Append booking platform links and Flight Tip
        links_and_tip = """

### Booking Platforms

- [MakeMyTrip](https://www.makemytrip.com/)
- [Google Flights](https://www.google.com/travel/flights)
- [Skyscanner](https://www.skyscanner.net/)
- [EaseMyTrip](https://www.easemytrip.com/)
- [Booking.com Flights](https://www.booking.com/flights.html)

**Flight Tip:** Compare baggage allowance and total final price before booking.
"""
        flight_data += links_and_tip

    except Exception as e:
        flight_data = f"Flight information unavailable: {str(e)}"

    new_plan["flight_results"] = flight_data
    new_plan_str = json.dumps(new_plan)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "modification_type": changes_str,
        "current_plan": new_plan_str
    }




# Hotel Agent
def hotel_agent(state: TravelState):
    print("\nHotel Agent Started\n")
    classification_str = state.get("modification_type", "{}")
    classification = json.loads(classification_str) if classification_str else {}

    rerun = classification.get("needs_hotel_agent", True)

    if not rerun or classification.get("needs_missing_info"):
        print("\nHotel Agent Completed (skipped/kept)\n")
        return {
            "hotel_results": state.get("hotel_results", ""),
            "messages": [
                AIMessage(content="Hotel recommendations kept from previous plan.")
            ],
            "llm_calls": state.get("llm_calls", 0)
        }

    plan_details = {}
    if state.get("current_plan"):
        try:
            plan_details = json.loads(state["current_plan"])
        except Exception:
            pass

    messages = state.get("messages", [])
    user_query = (
        state.get("user_query", "").strip()
        or get_latest_user_message(messages)
    )

    # Check if budget category exists in plan_details or user_query
    cat = plan_details.get("hotel_budget_category") or extract_hotel_budget_category(user_query)

    if not cat:
        clarification_msg = "What is your preferred hotel budget: budget, mid-range, luxury, or all categories?"
        print("\nHotel Agent Completed (clarification needed)\n")
        return {
            "hotel_results": clarification_msg,
            "messages": [AIMessage(content=clarification_msg)],
            "llm_calls": state.get("llm_calls", 0)
        }

    plan_details["hotel_budget_category"] = cat

    history_str = ""
    for m in messages[:-1]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        history_str += f"{role}: {m.content}\n"

    target_dest = plan_details.get('destination', '')
    cities_list = plan_details.get('cities', [])
    search_location = ', '.join(cities_list) if cities_list else target_dest

    cat_prompts = {
        "budget": "Find affordable and budget-friendly hotels that offer good value and convenient transport access.",
        "mid-range": "Find comfortable mid-range hotels with reliable amenities, good ratings and convenient locations.",
        "luxury": "Find premium and luxury hotels with high-quality service, excellent facilities and strong ratings.",
        "all": "Return a balanced list grouped into budget, mid-range and luxury categories."
    }

    prompt_guidance = cat_prompts.get(cat, cat_prompts["all"])

    search_query_prompt = f"""
    Based on the conversation history and the latest user request, formulate a single search query for finding hotels in {search_location}.
    
    Category Goal: {prompt_guidance}
    Target Plan Details:
    Destination/Cities: {search_location}
    Hotel Budget Category: {cat}
    Hotel Preferences: {plan_details.get('hotel_preferences', 'N/A')}
    
    Latest Request:
    {user_query}
    
    Return ONLY the search query string suitable for Tavily search engine. Do not include quotes or explanations.
    """
    try:
        search_query_resp = llm.invoke(search_query_prompt)
        search_query = search_query_resp.content.strip().strip('"').strip("'")
    except Exception:
        search_query = f"Best {cat} hotels in {search_location}"

    try:
        raw_hotel_results = run_async_safely(
            tavily_mcp_search(search_query)
        )
    except Exception as e:
        print(f"[HOTEL AGENT ERROR] {e}")
        raw_hotel_results = f"Hotel search unavailable: {str(e)}"

    # ── Hotel response formatting ──
    from response_formatters import format_hotel_response
    try:
        hotel_results = format_hotel_response(raw_hotel_results, category=cat)
    except Exception as e:
        print(f"[HOTEL FORMAT ERROR] {e}")
        hotel_results = raw_hotel_results if isinstance(raw_hotel_results, str) else "Hotel information unavailable."

    # Store normalized hotel list inside current_plan JSON for state persistence
    try:
        parsed_h = json.loads(hotel_results)
        if isinstance(parsed_h, dict) and "hotels" in parsed_h:
            plan_details["hotels"] = parsed_h["hotels"]
    except Exception:
        pass

    print("\nHotel Agent Completed\n")
    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "current_plan": json.dumps(plan_details)
    }


def weather_agent(state: TravelState):
    print("\nWeather Agent Started\n")
    classification_str = state.get("modification_type", "{}")
    classification = json.loads(classification_str) if classification_str else {}

    rerun = classification.get("needs_weather_agent", True)

    if not rerun or classification.get("needs_budget_clarification"):
        print("\nWeather Agent Completed (skipped/kept)\n")
        return {
            "weather_results": state.get("weather_results", ""),
            "messages": [
                AIMessage(content="Weather data kept from previous plan.")
            ]
        }

    user_query = (
        state.get("user_query", "").strip()
        or get_latest_user_message(state.get("messages", []))
    )
    
    plan_details = {}
    if state.get("current_plan"):
        try:
            plan_details = json.loads(state["current_plan"])
        except Exception:
            pass

    cities_list = plan_details.get("cities", [])
    if cities_list:
        city = cities_list[0]
    else:
        city = plan_details.get("destination", "")

    if not city:
        city = extract_destination(user_query, state.get("flight_results", ""))

    city = re.sub(r'(?i)\b(luxury|cheaper|trip|days|day)\b', '', str(city)).strip()
    if city.lower() in ["kerela", "kerala"]:
        city_lookup = "Kochi"
    else:
        city_lookup = city

    if not city_lookup or "specify" in city_lookup.lower() or "clarif" in city_lookup.lower():
        weather_results = f"### Weather Information\n\nPlease specify a destination city to fetch weather details."
        print("\nWeather Agent Completed (unspecified city)\n")
        return {
            "weather_results": weather_results,
            "messages": [
                AIMessage(
                    content="Weather information unavailable: destination city unspecified"
                )
            ]
        }

    try:
        weather_data = run_async_safely(
            weather_mcp_search(city_lookup)
        )
    except Exception as e:
        print(f"Weather Agent error fetching current weather for {city_lookup}: {type(e).__name__}: {e}")
        weather_data = f"Weather data unavailable: {str(e)}"

    try:
        forecast_data = run_async_safely(
            forecast_mcp_search(city_lookup)
        )
    except Exception as e:
        print(f"Weather Agent error fetching forecast for {city_lookup}: {type(e).__name__}: {e}")
        forecast_data = f"Forecast data unavailable: {str(e)}"

    trip_duration = str(plan_details.get("duration", ""))
    from response_formatters import format_weather_response
    formatted_weather = format_weather_response(weather_data, forecast_data, city if city else city_lookup, duration=trip_duration)

    print("\nWeather Agent Completed\n")
    return {
        "weather_results": formatted_weather,
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ]
    }


# Itinerary Agent
def itinerary_agent(state: TravelState):
    print("\nItinerary Agent Started\n")
    classification_str = state.get("modification_type", "{}")
    classification = json.loads(classification_str) if classification_str else {}
    request_type = classification.get("request_type", "full_plan")

    messages = state.get("messages", [])
    user_query = (
        state.get("user_query", "").strip()
        or get_latest_user_message(messages)
    )

    history_str = ""
    for m in messages[:-1]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content
        if len(content) > 1000:
            content = content[:1000] + "..."
        history_str += f"{role}: {content}\n"

    # 0. Check if missing information prompt needs to be returned
    if classification.get("needs_missing_info") and classification.get("missing_info_prompt"):
        missing_prompt = classification["missing_info_prompt"]
        print("\nItinerary Agent Completed (missing info prompt)\n")
        return {
            "itinerary": "",
            "messages": [AIMessage(content=missing_prompt)],
            "llm_calls": state.get("llm_calls", 0)
        }





# Itinerary Agent
def itinerary_agent(state: TravelState):
    classification_str = state.get("modification_type", "{}")
    try:
        classification = json.loads(classification_str) if classification_str else {}
    except Exception:
        classification = {}
    request_type = classification.get("request_type", "full_plan")

    messages = state.get("messages", [])
    user_query = (
        state.get("user_query", "").strip()
        or get_latest_user_message(messages)
    )

    history_str = ""
    for m in messages[:-1]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content
        if len(content) > 1000:
            content = content[:1000] + "..."
        history_str += f"{role}: {content}\n"

    # 0. Check if missing information prompt needs to be returned
    if classification.get("needs_missing_info") and classification.get("missing_info_prompt"):
        missing_prompt = classification["missing_info_prompt"]
        return {
            "itinerary": "",
            "messages": [AIMessage(content=missing_prompt)],
            "llm_calls": state.get("llm_calls", 0)
        }

    # 1. Bypass itinerary generation if not needed
    if not classification.get("needs_itinerary_agent", True):
        return {
            "itinerary": state.get("itinerary", ""),
            "messages": [AIMessage(content="I have updated your plan details.")],
            "llm_calls": state.get("llm_calls", 0)
        }

    # 2. Summary Request
    if request_type == "summary":
        plan_details = {}
        if state.get("current_plan"):
            try:
                plan_details = json.loads(state["current_plan"])
            except Exception:
                pass
                
        prompt = f"""
        Provide a concise, clear summary of the active travel plan.
        
        Trip Details:
        Destination: {plan_details.get('destination', 'N/A')}
        Departure City: {plan_details.get('departure_city', 'N/A')}
        Duration: {plan_details.get('duration', 'N/A')} days
        Hotel Budget: {plan_details.get('hotel_budget_category', 'N/A')}
        Overall Budget: {plan_details.get('budget', 'N/A')}
        
        Flight Highlights:
        {state.get('flight_results', 'N/A')[:400]}
        
        Hotel Highlights:
        {state.get('hotel_results', 'N/A')[:400]}
        
        Weather Highlights:
        {state.get('weather_results', 'N/A')[:400]}
        
        Itinerary Highlights:
        {state.get('itinerary', 'N/A')[:600]}
        
        Format output nicely in bullet points. Do NOT regenerate the complete travel plan.
        """
        response = llm.invoke([
            SystemMessage(content="You are a travel assistant summarizing an existing travel plan."),
            HumanMessage(content=prompt)
        ])
        return {
            "itinerary": state.get("itinerary", ""),
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    # 3. Specific Question Answering, Comparisons, Weather Questions, Hotel Proximity, or General Conversation
    elif request_type in ["plan_question", "general_conversation", "specific_question", "comparison", "weather_question", "focused_question", "hotel_question", "flight_question", "railway_question", "itinerary_question"]:
        plan_details = {}
        if state.get("current_plan"):
            try:
                plan_details = json.loads(state["current_plan"])
            except Exception:
                pass

        if request_type == "hotel_question" or ("hotel" in user_query.lower() and ("near" in user_query.lower() or "closest" in user_query.lower() or "station" in user_query.lower())):
            from response_formatters import answer_hotel_proximity_question
            ans = answer_hotel_proximity_question(state.get("hotel_results", ""), plan_details.get("destination", ""), user_query)
            return {
                "itinerary": state.get("itinerary", ""),
                "messages": [AIMessage(content=ans)],
                "llm_calls": state.get("llm_calls", 0)
            }

        prompt = f"""
        You are a helpful travel assistant answering a user question about their travel plan or options.
        Answer ONLY the user's specific query using the current travel plan details as context.
        
        Current Trip Context:
        Destination: {plan_details.get('destination', 'N/A')}
        Departure City: {plan_details.get('departure_city', 'N/A')}
        Duration: {plan_details.get('duration', 'N/A')} days
        Hotel Budget: {plan_details.get('hotel_budget_category', 'N/A')}
        
        Current Flight Results:
        {state.get('flight_results', 'N/A')}
        
        Current Hotel Results:
        {state.get('hotel_results', 'N/A')}
        
        Current Weather Results:
        {state.get('weather_results', 'N/A')}
        
        Current Itinerary:
        {state.get('itinerary', 'N/A')}
        
        Conversation History:
        {history_str}
        
        User Query:
        {user_query}
        
        Rules:
        - Answer ONLY the specific question asked. (For example: if asked which hotel is closest to Tokyo Station, name that hotel and location; if asked if it will rain on Day 3, describe Day 3 weather only; if comparing two hotels, compare only those two hotels; if asking for best flight, mention that flight only).
        - Do NOT re-display full flight lists, full hotel recommendations, full weather tables, or full itineraries unless requested.
        - Keep the answer direct, concise, and friendly.
        """
        response = llm.invoke([
            SystemMessage(content="You are a helpful travel assistant answering a specific user question."),
            HumanMessage(content=prompt)
        ])
        return {
            "itinerary": state.get("itinerary", ""),
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    # 4. Specific Day Changes or Attraction Modifications
    elif request_type in ["specific_day_change", "attraction_add", "attraction_remove"]:
        target_day = classification.get("target_day")
        prompt = f"""
        You are an expert travel planner. The user wants to modify a specific day's activities in their current itinerary.
        
        Previous Itinerary:
        {state.get('itinerary', 'N/A')}
        
        Modification Request:
        {user_query}
        
        Target Day to Modify: Day {target_day if target_day is not None else "specified in query"}
        
        Your job is to:
        1. Modify only the target day in the itinerary as requested, keeping every other day EXACTLY the same.
        2. Ensure the complete itinerary in "full_itinerary" ends with a section titled "💡 Itinerary Tip".
        3. Output a JSON object containing two keys:
           - "full_itinerary": The complete updated itinerary containing all days.
           - "concise_response": A friendly confirmation, ONLY the updated day's itinerary block, and exactly one section titled "💡 Itinerary Tip" at the end of the response containing a short, practical tip.
           
        Return ONLY the JSON. Do not include markdown code block syntax (like ```json) or explanations.
        """
        response = llm.invoke([
            SystemMessage(content="You are an expert travel planner updating a specific day in the itinerary."),
            HumanMessage(content=prompt)
        ])
        
        full_itinerary = state.get("itinerary", "")
        concise_response = ""
        try:
            text = response.content.strip()
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                full_itinerary = data.get("full_itinerary", full_itinerary)
                concise_response = data.get("concise_response", text)
            else:
                concise_response = text
        except Exception as e:
            print(f"Error parsing day update response: {e}")
            concise_response = response.content
            
        return {
            "itinerary": full_itinerary,
            "messages": [AIMessage(content=concise_response)],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    # 4. Full itinerary regeneration or general updates
    else:
        plan_details = {}
        if state.get("current_plan"):
            try:
                plan_details = json.loads(state["current_plan"])
            except Exception:
                pass

        plan_summary_str = f"""
        Destination: {plan_details.get('destination', 'N/A')}
        Duration: {plan_details.get('duration', 'N/A')} days
        Budget: {plan_details.get('budget', 'N/A')}
        Departure City: {plan_details.get('departure_city', 'N/A')}
        Cities to visit: {', '.join(plan_details.get('cities', [])) if isinstance(plan_details.get('cities'), list) else plan_details.get('cities', 'N/A')}
        Hotel Preferences: {plan_details.get('hotel_preferences', 'N/A')}
        Flight Preferences: {plan_details.get('flight_preferences', 'N/A')}
        """

        prompt = f"""
        Create or update a travel itinerary.
        
        Target Plan Details:
        {plan_summary_str}
        
        Conversation History:
        {history_str}

        Flight Results:
        {state.get('flight_results', 'N/A')}

        Hotel Results:
        {state.get('hotel_results', 'N/A')}

        Weather Information:
        {state.get('weather_results', 'N/A')}

        Previous Itinerary (to be modified if this is a follow-up):
        {state.get('itinerary', 'N/A')}

        Latest User Request/Modification:
        {user_query}
        
        Instructions:
        1. If the user request is a modification (e.g. "make it 5 days", "replace Osaka with Kyoto"), update the previous itinerary accordingly while preserving the rest of the plan's context and cities.
        2. Ensure the itinerary is exactly aligned with the Target Plan Details (especially destination, duration, and cities).
        3. Format the response beautifully in clean Markdown with day-by-day itineraries.
        4. Make sure the output contains the final travel itinerary, including place names, so inline images can be extracted.
        5. At the very end of the itinerary output, always include exactly one section titled "💡 Itinerary Tip" containing a short, practical travel tip (e.g., advising visiting outdoor sights early in the morning to avoid heat/crowds, or recommending swapping indoor and outdoor activities if rain is forecast).
        """

        response = llm.invoke([
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt)
        ])

        print("\nItinerary Agent Completed\n")
        return {
            "itinerary": response.content,
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }







def route_transport_start(state: TravelState) -> str:
    messages = state.get("messages", [])
    query = state.get("user_query", "").strip() or get_latest_user_message(messages)
    
    plan_str = state.get("current_plan", "")
    plan = {}
    if plan_str:
        try:
            plan = json.loads(plan_str)
        except Exception:
            pass

    classification_str = state.get("modification_type", "{}")
    c = {}
    try:
        c = json.loads(classification_str)
    except Exception:
        pass

    destination = plan.get("destination") or extract_destination_from_text(query)
    departure = plan.get("departure_city") or extract_departure_city_from_text(query, destination)

    is_intl = c.get("is_international")
    if is_intl is None:
        is_intl = is_international_trip(departure, destination) or is_international_trip("", query)

    if is_intl:
        return "flight_agent"

    mode = (plan.get("transport_mode", "") or extract_transport_preference(query) or c.get("transport_mode", "")).lower()
    cat = (plan.get("hotel_budget_category", "") or extract_hotel_budget_category(query)).lower()

    if mode in ["flight", "luxury"] or cat == "luxury":
        return "flight_agent"

    return "railway_agent"

def route_from_railway(state: TravelState) -> str:
    messages = state.get("messages", [])
    query = state.get("user_query", "").strip() or get_latest_user_message(messages)

    plan_str = state.get("current_plan", "")
    plan = {}
    if plan_str:
        try:
            plan = json.loads(plan_str)
        except Exception:
            pass

    classification_str = state.get("modification_type", "{}")
    c = {}
    try:
        c = json.loads(classification_str)
    except Exception:
        pass

    destination = plan.get("destination") or extract_destination_from_text(query)
    departure = plan.get("departure_city") or extract_departure_city_from_text(query, destination)

    is_intl = c.get("is_international")
    if is_intl is None:
        is_intl = is_international_trip(departure, destination) or is_international_trip("", query)

    if is_intl:
        return "hotel_agent"

    mode = (plan.get("transport_mode", "") or extract_transport_preference(query) or c.get("transport_mode", "")).lower()
    cat = (plan.get("hotel_budget_category", "") or extract_hotel_budget_category(query)).lower()

    if mode in ["all", "both", "select all", "select_all"] or cat in ["all", "mid-range", "select all", "select_all"]:
        return "flight_agent"

    return "hotel_agent"

graph = StateGraph(TravelState)

graph.add_node("railway_agent", railway_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)

graph.add_conditional_edges(START, route_transport_start, {
    "railway_agent": "railway_agent",
    "flight_agent": "flight_agent"
})
graph.add_conditional_edges("railway_agent", route_from_railway, {
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent"
})
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", END)


# Persistent connection so both CLI and Streamlit can share the compiled app
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    prepare_threshold=0,
    row_factory=dict_row,
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # config = {
    #     "configurable": {
    #         "thread_id": "user_aarohi"
    #     }
    # }

    # every run starts fresh.
    import uuid
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }


    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)
