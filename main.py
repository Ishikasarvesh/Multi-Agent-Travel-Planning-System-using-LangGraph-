# LangGraph Multi-Agent Travel Booking System with Long-Term Memory

import os
import json
import re
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
    aviation_mcp_call,extract_destination,forecast_mcp_search,weather_mcp_search,
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
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
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

UPDATE_PLAN_PROMPT = """
You are a state manager for a conversational travel planning assistant.
Analyze the conversation history, the previous version of the travel plan, and the latest user request to output the updated travel plan parameters and request classification.

Previous Travel Plan:
{previous_plan}

Conversation History:
{history}

Latest User Request:
{latest_request}

Your job is to:
1. Update the travel plan parameters based on the latest user request (destination, duration, budget, departure_city, cities, hotel_preferences, flight_preferences). Preserve previous values unless modified.
2. Classify the user's request into one of these types:
   - "full_plan": A broad request to plan a trip from scratch (e.g. "Plan a 7-day trip to Japan", "rebuild the plan").
   - "destination_change": User wants to change, add, or remove cities or countries (e.g. "replace Osaka with Kyoto").
   - "duration_change": User changes only the duration (e.g. "make it 5 days instead").
   - "budget_change": User changes only the budget (e.g. "increase budget to 2.5L").
   - "flight_request": User changes flight preferences (e.g. "find evening flights").
   - "hotel_request": User changes hotel preferences (e.g. "show cheaper hotels").
   - "weather_request": User asks about weather (e.g. "what is the weather in Tokyo?").
   - "itinerary_change": User wants general changes to itinerary activities.
   - "specific_day_change": User wants changes to a specific day's activities (e.g. "make day 3 more relaxed").
   - "attraction_add": User wants to add an attraction to a day (e.g. "add Nara Park to Day 4").
   - "attraction_remove": User wants to remove an attraction (e.g. "remove Osaka Castle").
   - "plan_question": A simple question about the current plan (e.g. "what is the budget?", "which hotel is for day 2?").
   - "general_conversation": General greeting or conversation (e.g. "hello", "thanks").

3. Set execution flags indicating which agents must run to handle this specific change:
   - "needs_flight_agent": true if destination_change, flight_request, or full_plan.
   - "needs_hotel_agent": true if destination_change, hotel_request, budget_change, or full_plan.
   - "needs_weather_agent": true if destination_change, weather_request, or full_plan.
   - "needs_itinerary_agent": true if request_type is anything other than plan_question or general_conversation.
   - "show_full_plan": true if full_plan, destination_change, or if it is a new trip.

4. If "specific_day_change", "attraction_add", or "attraction_remove" is detected:
   - Identify the "target_day" (integer, e.g. 3 or 4). If not applicable, set to null.

Ensure the output is a JSON object matching this structure:
{{
    "updated_plan": {{
        "destination": "country or region, e.g. Japan",
        "duration": "number of days, e.g. 5",
        "budget": "budget description, e.g. ₹2.5 lakhs",
        "departure_city": "city/airport, e.g. Mumbai",
        "cities": ["list", "of", "cities", "to", "visit"],
        "hotel_preferences": "description of hotel preferences, e.g. cheaper hotels",
        "flight_preferences": "description of flight preferences, e.g. evening flights only"
    }},
    "classification": {{
        "request_type": "one of the types listed above",
        "target_day": integer or null,
        "requested_change": "brief summary of the request",
        "needs_flight_agent": true/false,
        "needs_hotel_agent": true/false,
        "needs_weather_agent": true/false,
        "needs_itinerary_agent": true/false,
        "show_full_plan": true/false
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
        "cities": [],
        "hotel_preferences": "",
        "flight_preferences": ""
    }
    
    default_classification = {
        "request_type": "full_plan",
        "target_day": None,
        "requested_change": "",
        "needs_flight_agent": True,
        "needs_hotel_agent": True,
        "needs_weather_agent": True,
        "needs_itinerary_agent": True,
        "show_full_plan": True
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
    for m in messages[:-1]: # exclude the latest message
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content
        if len(content) > 500:
            content = content[:500] + "..."
        history_str += f"{role}: {content}\n"

    # Call LLM to update plan
    prompt = UPDATE_PLAN_PROMPT.format(
        previous_plan=json.dumps(old_plan, indent=2),
        history=history_str,
        latest_request=user_query
    )

    new_plan = old_plan.copy()
    classification = default_classification.copy()

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                new_plan = data.get("updated_plan", old_plan)
                classification = data.get("classification", default_classification)
    except Exception as e:
        print(f"Plan update error: {e}")

    # Ensure all classification keys are safely initialized
    classification = {
        "request_type": classification.get("request_type", "full_plan"),
        "target_day": classification.get("target_day", None),
        "requested_change": classification.get("requested_change", ""),
        "needs_flight_agent": bool(classification.get("needs_flight_agent", True)),
        "needs_hotel_agent": bool(classification.get("needs_hotel_agent", True)),
        "needs_weather_agent": bool(classification.get("needs_weather_agent", True)),
        "needs_itinerary_agent": bool(classification.get("needs_itinerary_agent", True)),
        "show_full_plan": bool(classification.get("show_full_plan", True))
    }

    # If it is a new trip (first run, old plan is empty), force everything to run
    if not old_plan.get("destination"):
        classification["needs_flight_agent"] = True
        classification["needs_hotel_agent"] = True
        classification["needs_weather_agent"] = True
        classification["needs_itinerary_agent"] = True
        classification["show_full_plan"] = True

    return new_plan, classification

# Flight Agent
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    messages = state.get("messages", [])
    query = state["user_query"]
    previous_plan_str = state.get("current_plan", "")

    # Run state updater to get new plan and detected changes
    new_plan, classification = update_travel_plan_state(messages, query, previous_plan_str)
    
    new_plan_str = json.dumps(new_plan)
    changes_str = json.dumps(classification)

    rerun = classification.get("needs_flight_agent", True)

    if not rerun:
        return {
            "flight_results": state.get("flight_results", ""),
            "messages": [
                AIMessage(
                    content="Flight recommendations kept from previous plan (no changes needed)."
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

Book on:

🔗 [MakeMyTrip](https://www.makemytrip.com/)

🔗 [Booking.com Flights](https://www.booking.com/flights.html)

🔗 [Skyscanner](https://www.skyscanner.net/)

🔗 [Google Flights](https://www.google.com/travel/flights)

🔗 [EaseMyTrip](https://www.easemytrip.com/)

💡 Flight Tip:
Booking 30–60 days in advance usually offers better prices for international trips.
"""
        flight_data += links_and_tip

    except Exception as e:
        flight_data = f"Flight information unavailable: {str(e)}"

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
    classification_str = state.get("modification_type", "{}")
    classification = json.loads(classification_str)

    rerun = classification.get("needs_hotel_agent", True)

    if not rerun:
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
    user_query = state["user_query"]

    history_str = ""
    for m in messages[:-1]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        history_str += f"{role}: {m.content}\n"

    target_dest = plan_details.get('destination', '')
    cities_list = plan_details.get('cities', [])
    search_location = ', '.join(cities_list) if cities_list else target_dest

    search_query_prompt = f"""
    Based on the conversation history and the latest user request, formulate a single search query for finding hotels.
    The search query should be suitable for a search engine like Tavily. It must include the destination city/cities and any specific hotel preferences or budget constraints.
    
    Target Plan Details:
    Destination/Cities: {search_location}
    Budget Constraint: {plan_details.get('budget', 'N/A')}
    Hotel Preferences: {plan_details.get('hotel_preferences', 'N/A')}
    
    Conversation History:
    {history_str}
    
    Latest Request:
    {user_query}
    
    Return ONLY the search query string. Do not include quotes, explanations, or any other text.
    """
    try:
        search_query_resp = llm.invoke(search_query_prompt)
        search_query = search_query_resp.content.strip().strip('"').strip("'")
    except Exception:
        search_query = f"Best hotels in {search_location} budget {plan_details.get('budget', 'N/A')}"

    raw_hotel_results = run_async_safely(
        tavily_mcp_search(search_query)
    )

    # ── Hotel response formatting ──
    from response_formatters import format_hotel_response
    hotel_results = format_hotel_response(raw_hotel_results)

    destination = plan_details.get("destination", "")
    public_transport_note = ""
    major_pt_destinations = ["japan", "tokyo", "kyoto", "osaka", "paris", "london", "singapore", "hong kong", "seoul", "berlin", "new york"]
    if any(pt in destination.lower() for pt in major_pt_destinations):
        public_transport_note = f" Since public transport in {destination} is highly efficient, staying near a transit hub will make commuting extremely convenient."

    hotel_tip = f"\n\n💡 Hotel Tip:\nChoose hotels near metro or railway stations to reduce local transport costs.{public_transport_note}\n"
    hotel_results += hotel_tip

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }




def weather_agent(state: TravelState):
    classification_str = state.get("modification_type", "{}")
    classification = json.loads(classification_str)

    rerun = classification.get("needs_weather_agent", True)

    if not rerun:
        return {
            "weather_results": state.get("weather_results", ""),
            "messages": [
                AIMessage(content="Weather data kept from previous plan.")
            ]
        }

    user_query = state["user_query"]
    
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

    # If the helper determined that the city is unspecified or asked for clarification
    if not city or "specify" in city.lower() or "clarif" in city.lower():
        weather_results = f"### Weather Information\n\n{city or 'Please specify a destination city to fetch weather details.'}"
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
            weather_mcp_search(city)
        )
    except Exception as e:
        weather_data = f"Weather data unavailable: {str(e)}"

    try:
        forecast_data = run_async_safely(
            forecast_mcp_search(city)
        )
    except Exception as e:
        forecast_data = f"Forecast data unavailable: {str(e)}"

    from response_formatters import format_weather_response
    formatted_weather = format_weather_response(weather_data, forecast_data, city)

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
    classification_str = state.get("modification_type", "{}")
    classification = json.loads(classification_str)
    request_type = classification.get("request_type", "full_plan")

    messages = state.get("messages", [])
    user_query = state["user_query"]

    history_str = ""
    for m in messages[:-1]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content
        if len(content) > 1000:
            content = content[:1000] + "..."
        history_str += f"{role}: {content}\n"

    # 1. Bypass itinerary generation if not needed
    if not classification.get("needs_itinerary_agent", True):
        return {
            "itinerary": state.get("itinerary", ""),
            "messages": [AIMessage(content="I have updated your plan details.")],
            "llm_calls": state.get("llm_calls", 0)
        }

    # 2. Simple Question or General Conversation
    if request_type in ["plan_question", "general_conversation"]:
        prompt = f"""
        You are a helpful travel assistant answering a user query about their travel plan or having a general conversation.
        Answer the user's latest query accurately using the current travel plan details as context.
        Do not generate any itineraries, flight info, or hotel suggestions unless directly asked. Keep it concise.
        
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
        """
        response = llm.invoke([
            SystemMessage(content="You are a helpful travel assistant answering a user question."),
            HumanMessage(content=prompt)
        ])
        return {
            "itinerary": state.get("itinerary", ""),
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    # 3. Specific Day Changes or Attraction Modifications
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

        return {
            "itinerary": response.content,
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }







graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)


graph.add_edge(START, "flight_agent")
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
