"""
Interactive Multi-Agent Execution Visualizer Component (Inline Full-Width Execution Panel & Replay Mode).
Provides real-time, event-driven visualization of LangGraph multi-agent execution,
graph-style flow diagram with branching and merging, real-time log mirroring of flow transitions,
isolated execution logs per chat session, source-bottom anchored packet cards with strict Z-index hierarchy,
and interactive data payload inspection modals.
"""

import json
import time
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

# Node catalog with metadata and humanized label definitions
NODE_CATALOG = {
    "user_query": {
        "name": "User Query",
        "icon": "👤",
        "category": "Input",
        "api": None,
        "packet_label": "User Trip Details",
        "description": "Captures raw travel prompt and user parameters"
    },
    "planner_agent": {
        "name": "Planner Agent",
        "icon": "🧠",
        "category": "Reasoning",
        "api": "LLM (Groq LLaMA 3.3 70B)",
        "packet_label": "Route & Budget Insights",
        "description": "Extracts destination, budget, duration, dates, and intent"
    },
    "transport_decision": {
        "name": "Transport Router",
        "icon": "🔀",
        "category": "Router",
        "api": None,
        "packet_label": "Transport Decision",
        "description": "Routes query to Flight, Railway, or Both based on preference"
    },
    "flight_agent": {
        "name": "Flight Agent",
        "icon": "✈️",
        "category": "Agent",
        "api": "AviationStack API",
        "packet_label": "Flight Options",
        "description": "Searches flight schedules, airlines, duration, and fare ranges"
    },
    "railway_agent": {
        "name": "Railway Agent",
        "icon": "🚆",
        "category": "Agent",
        "api": "RailRadar / IRCTC",
        "packet_label": "Train Schedules",
        "description": "Finds train options, schedules, classes, and IRCTC links"
    },
    "hotel_agent": {
        "name": "Hotel Agent",
        "icon": "🏨",
        "category": "Agent",
        "api": "Hotel Search API",
        "packet_label": "Hotel Listings",
        "description": "Recommends hotels grouped by budget with booking links"
    },
    "weather_agent": {
        "name": "Weather Agent",
        "icon": "🌦️",
        "category": "Agent",
        "api": "Weather API",
        "packet_label": "Weather Forecast Data",
        "description": "Fetches forecast, UV index, rain warnings & packing tips"
    },
    "itinerary_agent": {
        "name": "Itinerary Agent",
        "icon": "🗓️",
        "category": "Agent",
        "api": "LLM (Groq LLaMA 3.3 70B)",
        "packet_label": "Daily Itinerary Plan",
        "description": "Synthesizes day-by-day travel plan and inline attractions"
    },
    "final_response": {
        "name": "Final Response",
        "icon": "💬",
        "category": "Synthesis",
        "api": None,
        "packet_label": "Plan Summary",
        "description": "Assembles final travel plan and interactive UI sections"
    },
    "download_builder": {
        "name": "Download Builder",
        "icon": "📄",
        "category": "Exporter",
        "api": "Markdown Engine",
        "packet_label": "Trip Dossier",
        "description": "Generates downloadable trip dossier file"
    }
}

API_NODES = {
    "flight_agent": {
        "id": "aviation_api",
        "name": "AviationStack API Search",
        "icon": "🛫"
    },
    "railway_agent": {
        "id": "railradar_api",
        "name": "RailRadar / IRCTC Search",
        "icon": "🚄"
    },
    "hotel_agent": {
        "id": "hotel_api",
        "name": "Hotel Search API",
        "icon": "🏨"
    },
    "weather_agent": {
        "id": "weather_api",
        "name": "Weather API Search",
        "icon": "🌤️"
    },
}

def init_visualizer_state():
    """Initializes session state tracking variables for the visualizer."""
    if "visualizer_node_states" not in st.session_state:
        st.session_state.visualizer_node_states = {
            node_id: {
                "status": "waiting",  # waiting | running | completed | failed | skipped
                "start_time": None,
                "end_time": None,
                "duration": None,
                "input": "Awaiting execution...",
                "output": "Pending...",
                "process": ["Initialized node listener"],
                "passed_to": None,
                "api_called": NODE_CATALOG[node_id].get("api")
            }
            for node_id in NODE_CATALOG
        }

    if "visualizer_api_states" not in st.session_state:
        st.session_state.visualizer_api_states = {
            api_info["id"]: {
                "name": api_info["name"],
                "icon": api_info["icon"],
                "status": "waiting",
                "duration": None
            }
            for api_info in API_NODES.values()
        }

    if "visualizer_timeline" not in st.session_state:
        st.session_state.visualizer_timeline = []

    if "visualizer_overall_status" not in st.session_state:
        st.session_state.visualizer_overall_status = "idle"

    if "show_visualizer" not in st.session_state:
        st.session_state.show_visualizer = False

    if "visualizer_mode" not in st.session_state:
        st.session_state.visualizer_mode = "live"

    if "visualizer_executed_nodes" not in st.session_state:
        st.session_state.visualizer_executed_nodes = list(NODE_CATALOG.keys())

    if "visualizer_transfers" not in st.session_state:
        st.session_state.visualizer_transfers = []

    if "chat_execution_snapshots" not in st.session_state:
        st.session_state.chat_execution_snapshots = {}

    if "visualizer_runtime_data" not in st.session_state:
        st.session_state.visualizer_runtime_data = {
            "source": "Origin",
            "destination": "Destination",
            "budget": "Flexible",
            "dates": "Scheduled",
            "user_query": ""
        }


def reset_visualizer_for_run(executed_nodes=None):
    """Resets visualizer state for a new agent execution run within a chat session."""
    init_visualizer_state()
    st.session_state.visualizer_overall_status = "executing"
    st.session_state.show_visualizer = True
    st.session_state.visualizer_mode = "live"
    now_str = datetime.now().strftime("%H:%M:%S")

    # Per-chat session log preservation: keep existing logs if present, else initialize fresh
    current_chat_id = st.session_state.get("current_chat_id")
    chat_snapshot = st.session_state.get("chat_execution_snapshots", {}).get(current_chat_id)
    if chat_snapshot and chat_snapshot.get("timeline"):
        st.session_state.visualizer_timeline = list(chat_snapshot["timeline"])
        st.session_state.visualizer_timeline.append({
            "time": now_str,
            "icon": "🔄",
            "message": "--- Follow-up Workflow Stream Started ---",
            "type": "info"
        })
    else:
        st.session_state.visualizer_timeline = [{
            "time": now_str,
            "icon": "🚀",
            "message": "AI Multi-Agent pipeline execution started",
            "type": "start"
        }]

    if executed_nodes:
        st.session_state.visualizer_executed_nodes = executed_nodes
        for node_id in NODE_CATALOG:
            if node_id in executed_nodes:
                st.session_state.visualizer_node_states[node_id].update({
                    "status": "waiting",
                    "start_time": None,
                    "end_time": None,
                    "duration": None,
                    "output": "Updating...",
                    "process": ["Re-evaluating agent for modification..."]
                })
            else:
                st.session_state.visualizer_node_states[node_id].update({
                    "status": "skipped",
                    "output": "Skipped (not requested)",
                })
    else:
        st.session_state.visualizer_executed_nodes = list(NODE_CATALOG.keys())
        for node_id in st.session_state.visualizer_node_states:
            st.session_state.visualizer_node_states[node_id].update({
                "status": "waiting",
                "start_time": None,
                "end_time": None,
                "duration": None,
                "input": "Awaiting execution...",
                "output": "Pending...",
                "process": ["Node queued for execution"]
            })

    for api_id in st.session_state.visualizer_api_states:
        st.session_state.visualizer_api_states[api_id].update({
            "status": "waiting",
            "duration": None
        })

    st.session_state.visualizer_transfers = []


def record_node_start(node_id: str, input_summary: str = "", process_msg: str = ""):
    """Records start of node execution."""
    init_visualizer_state()
    now = time.time()
    now_str = datetime.now().strftime("%H:%M:%S")
    meta = NODE_CATALOG.get(node_id, {"name": node_id, "icon": "⚙️"})

    node_data = st.session_state.visualizer_node_states.get(node_id, {})
    node_data["status"] = "running"
    node_data["start_time"] = now
    if input_summary:
        node_data["input"] = input_summary
    if process_msg:
        node_data["process"] = [process_msg] if not isinstance(node_data.get("process"), list) else node_data["process"] + [process_msg]

    st.session_state.visualizer_node_states[node_id] = node_data

    if node_id in API_NODES:
        api_id = API_NODES[node_id]["id"]
        if api_id in st.session_state.visualizer_api_states:
            st.session_state.visualizer_api_states[api_id]["status"] = "running"

    st.session_state.visualizer_timeline.append({
        "time": now_str,
        "icon": "🔍",
        "message": f"Action: {meta['name']} started execution ({input_summary or 'Processing request'})",
        "type": "start"
    })


def record_node_complete(node_id: str, output_summary: str = "", process_msg: str = "", passed_to: str = None):
    """Records completion of node execution and mirrors transition data into timeline log in real-time."""
    init_visualizer_state()
    now = time.time()
    now_str = datetime.now().strftime("%H:%M:%S")
    meta = NODE_CATALOG.get(node_id, {"name": node_id, "icon": "⚙️"})

    node_data = st.session_state.visualizer_node_states.get(node_id, {})
    node_data["status"] = "completed"
    node_data["end_time"] = now
    if node_data.get("start_time"):
        dur = round(now - node_data["start_time"], 2)
        node_data["duration"] = f"{dur}s"
    else:
        node_data["duration"] = "<1s"

    if output_summary:
        node_data["output"] = output_summary
    if process_msg and isinstance(node_data.get("process"), list):
        node_data["process"].append(process_msg)
    if passed_to:
        node_data["passed_to"] = passed_to

    st.session_state.visualizer_node_states[node_id] = node_data

    if node_id in API_NODES:
        api_id = API_NODES[node_id]["id"]
        if api_id in st.session_state.visualizer_api_states:
            st.session_state.visualizer_api_states[api_id]["status"] = "completed"
            st.session_state.visualizer_api_states[api_id]["duration"] = node_data["duration"]

    # Extract real runtime state parameters
    rt = st.session_state.get("visualizer_runtime_data", {})
    src = rt.get("source", "Origin")
    dest = rt.get("destination", "Destination")
    bgt = rt.get("budget", "Flexible")

    # Real-Time Flow Mirror Logging
    if passed_to:
        target_name = NODE_CATALOG.get(passed_to, {}).get("name", passed_to)
        flow_summary = f"Result: {meta['name']} ➡️ Passing to {target_name} | Data: {src} → {dest} • {bgt}"
        st.session_state.visualizer_timeline.append({
            "time": now_str,
            "icon": "✅",
            "message": flow_summary,
            "type": "complete"
        })
    else:
        st.session_state.visualizer_timeline.append({
            "time": now_str,
            "icon": "✅",
            "message": f"Result: {meta['name']} completed ({output_summary or node_data['duration']})",
            "type": "complete"
        })


def record_node_failure(node_id: str, error_msg: str = "", details: dict = None):
    """Records a failure state for a node and logs failure details in real-time."""
    init_visualizer_state()
    now_str = datetime.now().strftime("%H:%M:%S")
    meta = NODE_CATALOG.get(node_id, {"name": node_id, "icon": "⚙️"})

    node_data = st.session_state.visualizer_node_states.get(node_id, {})
    node_data["status"] = "failed"
    node_data["output"] = f"Execution Failed: {error_msg or 'Service request unavailable'}"
    if details:
        node_data["details"] = details
    st.session_state.visualizer_node_states[node_id] = node_data

    if node_id in API_NODES:
        api_id = API_NODES[node_id]["id"]
        if api_id in st.session_state.visualizer_api_states:
            st.session_state.visualizer_api_states[api_id]["status"] = "failed"

    # Real-time failure log entry
    err_txt = error_msg or "Execution Failed"
    st.session_state.visualizer_timeline.append({
        "time": now_str,
        "icon": "❌",
        "message": f"Issue: {meta['name']} Failed ({err_txt})",
        "type": "error"
    })


def record_api_start(agent_id: str, api_name: str):
    """Records API invocation start event."""
    init_visualizer_state()
    now_str = datetime.now().strftime("%H:%M:%S")
    if agent_id in API_NODES:
        api_id = API_NODES[agent_id]["id"]
        if api_id in st.session_state.visualizer_api_states:
            st.session_state.visualizer_api_states[api_id]["status"] = "running"

    st.session_state.visualizer_timeline.append({
        "time": now_str,
        "icon": "🔍",
        "message": f"Action: Querying {api_name}",
        "type": "info"
    })


def record_api_complete(agent_id: str, api_name: str, summary: str = ""):
    """Records API invocation completion event."""
    init_visualizer_state()
    now_str = datetime.now().strftime("%H:%M:%S")
    if agent_id in API_NODES:
        api_id = API_NODES[agent_id]["id"]
        if api_id in st.session_state.visualizer_api_states:
            st.session_state.visualizer_api_states[api_id]["status"] = "completed"

    st.session_state.visualizer_timeline.append({
        "time": now_str,
        "icon": "✅",
        "message": f"Result: {api_name} retrieved ({summary or 'Success'})",
        "type": "info"
    })


def record_data_transfer(from_node: str, to_node: str, packet_label: str = ""):
    """Records data packet movement between agents and logs transition in real-time."""
    init_visualizer_state()
    now_str = datetime.now().strftime("%H:%M:%S")
    from_meta = NODE_CATALOG.get(from_node, {}).get("name", from_node)
    to_meta = NODE_CATALOG.get(to_node, {}).get("name", to_node)
    lbl = packet_label or NODE_CATALOG.get(from_node, {}).get("packet_label", "Payload")

    rt = st.session_state.get("visualizer_runtime_data", {})
    src = rt.get("source", "Origin")
    dest = rt.get("destination", "Destination")

    st.session_state.visualizer_transfers.append({
        "from": from_node,
        "to": to_node,
        "label": lbl
    })

    st.session_state.visualizer_timeline.append({
        "time": now_str,
        "icon": "📄",
        "message": f"{from_meta} → {to_meta} | Data: {lbl} ({src} → {dest}) | Status: Success",
        "type": "info"
    })


def record_download_builder(stage: str, details: str = ""):
    """Triggers Download Builder stage animation and completion status."""
    init_visualizer_state()
    now_str = datetime.now().strftime("%H:%M:%S")

    if stage == "start":
        record_node_start("download_builder", input_summary="Assembling travel dossier", process_msg="Generating download file...")
    elif stage == "completed":
        record_node_complete("download_builder", output_summary="Trip dossier ready", process_msg="File ready for download!")
        st.session_state.visualizer_overall_status = "completed"
        st.session_state.visualizer_timeline.append({
            "time": now_str,
            "icon": "🎉",
            "message": "📄 Travel Plan Generated Successfully!",
            "type": "complete"
        })


def generate_execution_panel_html(snapshot_data=None, mode: str = "live") -> str:
    """
    Generates the inline full-width System Execution Panel HTML/CSS/JS code.
    Features:
    - Hard positioning constraint: packet cards anchored STRICTLY BELOW source node / API cards
    - Strict Z-Index Layering: Nodes & API Cards (3) > Data Packet Labels (2) > SVG Arrows (1)
    - Abundant vertical clearance (76px gap) between node rows
    - Dynamic runtime state data flow binding & real-time log mirroring
    """
    if snapshot_data:
        nodes_state = snapshot_data.get("node_states", {})
        timeline_data = snapshot_data.get("timeline", [])
        executed_nodes = snapshot_data.get("executed_nodes", list(NODE_CATALOG.keys()))
        overall_status = snapshot_data.get("overall_status", "completed")
        api_states = snapshot_data.get("api_states", {})
        runtime_data = snapshot_data.get("runtime_data", {})
    else:
        nodes_state = st.session_state.get("visualizer_node_states", {})
        timeline_data = st.session_state.get("visualizer_timeline", [])
        executed_nodes = st.session_state.get("visualizer_executed_nodes", list(NODE_CATALOG.keys()))
        overall_status = st.session_state.get("visualizer_overall_status", "idle")
        api_states = st.session_state.get("visualizer_api_states", {})
        runtime_data = st.session_state.get("visualizer_runtime_data", {})

    nodes_state_json = json.dumps(nodes_state)
    timeline_json = json.dumps(timeline_data)
    catalog_json = json.dumps(NODE_CATALOG)
    executed_nodes_json = json.dumps(executed_nodes)
    api_nodes_json = json.dumps(API_NODES)
    api_states_json = json.dumps(api_states)
    runtime_data_json = json.dumps(runtime_data)

    # Compute total execution time
    total_dur = "0.0s"
    try:
        times = [v["end_time"] - v["start_time"] for v in nodes_state.values() if v.get("start_time") and v.get("end_time")]
        if times:
            total_dur = f"{round(sum(times), 2)}s"
    except Exception:
        pass

    badge_label = "LIVE RUN" if mode == "live" else "REPLAY SNAPSHOT"
    badge_class = "badge-live" if mode == "live" else "badge-replay"
    status_label = "COMPLETED" if overall_status == "completed" else ("RUNNING" if overall_status == "executing" else "READY")
    status_class = "status-complete" if overall_status == "completed" else "status-running"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }}

            body {{
                background: transparent;
                color: #E5E7EB;
                overflow-x: hidden;
            }}

            /* Inline Full-Width Fixed Execution Panel Container */
            .exec-panel {{
                width: 100%;
                background: #0F172A;
                border: 1px solid #1F2937;
                border-radius: 14px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 16px;
                margin: 12px 0;
                position: relative;
            }}

            /* Header Section */
            .panel-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding-bottom: 12px;
                border-bottom: 1px solid #1F2937;
                flex-wrap: wrap;
                gap: 8px;
            }}

            .panel-title-group {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}

            .panel-icon {{
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 32px;
                height: 32px;
                background: #111827;
                border: 1px solid #1F2937;
                border-radius: 8px;
            }}

            .panel-title {{
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 0.03em;
                color: #F3F4F6;
                text-transform: uppercase;
            }}

            .badge {{
                font-size: 10px;
                font-weight: 600;
                padding: 3px 9px;
                border-radius: 20px;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}
            .badge-live {{ background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.4); }}
            .badge-replay {{ background: rgba(156, 163, 175, 0.15); color: #9CA3AF; border: 1px solid rgba(156, 163, 175, 0.3); }}
            .status-complete {{ background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4); }}
            .status-running {{ background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.4); }}

            .panel-controls {{
                display: flex;
                align-items: center;
                gap: 12px;
            }}

            .duration-pill {{
                font-size: 11px;
                font-family: ui-monospace, monospace;
                color: #9CA3AF;
                background: #111827;
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid #1F2937;
            }}

            .replay-btn {{
                background: #111827;
                border: 1px solid #374151;
                color: #E5E7EB;
                font-size: 11px;
                font-weight: 600;
                padding: 5px 12px;
                border-radius: 6px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 6px;
                transition: all 0.2s ease;
            }}
            .replay-btn:hover {{
                background: #1F2937;
                border-color: #4B5563;
                color: #FFFFFF;
            }}

            /* Body Grid Layout */
            .panel-body {{
                display: grid;
                grid-template-columns: minmax(0, 1fr) 280px;
                gap: 16px;
                align-items: start;
            }}

            @media (max-width: 900px) {{
                .panel-body {{
                    grid-template-columns: 1fr;
                }}
            }}

            /* Graph Container */
            .graph-viewport {{
                background: #111827;
                border: 1px solid #1F2937;
                border-radius: 10px;
                padding: 32px 16px;
                position: relative;
                min-height: 640px;
                overflow-x: auto;
            }}

            /* STRICT Z-INDEX HIERARCHY: Nodes/API Cards (3) > Data Packet Labels (2) > SVG Arrows (1) */
            .svg-layer {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 1;
            }}

            .packet-cards-layer {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 2;
            }}

            .graph-rows {{
                display: flex;
                flex-direction: column;
                gap: 76px; /* Increased vertical clearance between rows */
                position: relative;
                z-index: 3;
            }}

            .graph-row {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 32px;
                position: relative;
                z-index: 3;
            }}

            .row-split {{
                justify-content: space-around;
            }}

            /* Agent Node Box */
            .node-card {{
                background: #1E293B;
                border: 1.5px solid #1F2937;
                border-radius: 10px;
                padding: 8px 12px;
                min-width: 175px;
                max-width: 215px;
                display: flex;
                align-items: center;
                gap: 10px;
                position: relative;
                z-index: 3;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                cursor: pointer;
            }}
            .node-card:hover {{
                border-color: #4B5563;
            }}

            .node-icon-wrapper {{
                width: 32px;
                height: 32px;
                border-radius: 8px;
                background: #0F172A;
                border: 1px solid #1F2937;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                flex-shrink: 0;
            }}

            .node-info {{
                display: flex;
                flex-direction: column;
                gap: 2px;
                overflow: hidden;
            }}

            .node-name {{
                font-size: 12px;
                font-weight: 700;
                color: #F3F4F6;
                white-space: nowrap;
                text-overflow: ellipsis;
                overflow: hidden;
            }}

            .node-meta {{
                font-size: 10px;
                color: #9CA3AF;
                display: flex;
                align-items: center;
                gap: 4px;
            }}

            /* Node States */
            .state-waiting {{
                opacity: 0.5;
                border-style: dashed;
                border-color: #374151;
            }}

            .state-running {{
                opacity: 1;
                border-color: #F59E0B;
                box-shadow: 0 0 14px rgba(245, 158, 11, 0.25);
                animation: subtleAmberPulse 2s infinite ease-in-out;
            }}
            .state-running .node-name {{ color: #F59E0B; }}

            .state-completed {{
                opacity: 1;
                border-color: #10B981;
                background: #1E293B;
            }}
            .state-completed .node-icon-wrapper {{ border-color: #10B981; }}

            .state-failed {{
                opacity: 1;
                border-color: #EF4444;
                background: rgba(239, 68, 68, 0.12);
                box-shadow: 0 0 12px rgba(239, 68, 68, 0.25);
            }}
            .state-failed .node-name {{ color: #EF4444; }}
            .state-failed .node-icon-wrapper {{ border-color: #EF4444; background: rgba(239, 68, 68, 0.2); }}

            .state-skipped {{
                opacity: 0.4;
                border-style: dashed;
                border-color: #374151;
                background: #1E293B;
            }}
            .state-skipped .node-meta {{ color: #6B7280; }}

            /* MCP / API Group & Cards */
            .agent-group {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 6px;
                position: relative;
                z-index: 3;
            }}

            .api-card {{
                background: #0F172A;
                border: 1px solid #1F2937;
                border-radius: 8px;
                padding: 6px 10px;
                display: flex;
                flex-direction: column;
                gap: 4px;
                min-width: 185px;
                position: relative;
                z-index: 3;
                transition: all 0.3s ease;
                cursor: pointer;
            }}
            .api-card:hover {{ border-color: #4B5563; }}

            .api-card-active {{
                border-color: #F59E0B;
                background: rgba(245, 158, 11, 0.08);
            }}
            .api-card-complete {{
                border-color: #10B981;
                background: rgba(16, 185, 129, 0.06);
            }}
            .api-card-failed {{
                border-color: #EF4444;
                background: rgba(239, 68, 68, 0.1);
            }}

            .api-header {{
                font-size: 10px;
                font-weight: 700;
                color: #E5E7EB;
                display: flex;
                align-items: center;
                gap: 5px;
            }}

            .human-pill {{
                font-size: 9px;
                line-height: 1.3;
                padding: 3px 6px;
                border-radius: 4px;
                background: #1E293B;
                border: 1px solid #1F2937;
                white-space: nowrap;
                text-overflow: ellipsis;
                overflow: hidden;
                max-width: 195px;
            }}
            .pill-input {{ color: #F59E0B; border-color: rgba(245, 158, 11, 0.3); }}
            .pill-output {{ color: #10B981; border-color: rgba(16, 185, 129, 0.3); }}
            .pill-failed {{ color: #EF4444; border-color: rgba(239, 68, 68, 0.3); }}

            /* Inline Data Packet Cards (Positioned STRICTLY BELOW Source Cards) */
            .packet-card {{
                position: absolute;
                transform: translate(-50%, 0%); /* Top aligned to target top Y */
                pointer-events: auto;
                background: #1E293B;
                border: 1px solid #1F2937;
                border-radius: 6px;
                padding: 4px 9px;
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 9.5px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
                cursor: pointer;
                transition: all 0.25s ease;
                white-space: nowrap;
                max-width: 240px;
                z-index: 2; /* Layer 2: Below Cards (3), Above SVG Arrows (1) */
            }}
            .packet-card:hover {{
                transform: translate(-50%, -2px) scale(1.03);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.45);
            }}

            .packet-card-waiting {{
                opacity: 0.45;
                border-style: dashed;
                color: #6B7280;
            }}
            .packet-card-transferring {{
                border-color: #F59E0B;
                background: rgba(245, 158, 11, 0.12);
                color: #F59E0B;
                animation: packetPulse 1.2s infinite ease-in-out;
            }}
            .packet-card-completed {{
                border-color: #10B981;
                background: #1E293B;
                color: #10B981;
            }}
            .packet-card-failed {{
                border-color: #EF4444;
                background: rgba(239, 68, 68, 0.12);
                color: #EF4444;
            }}

            .packet-summary-text {{
                color: #9CA3AF;
                font-size: 9px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                max-width: 140px;
            }}

            /* Timeline Log Panel */
            .timeline-panel {{
                background: #111827;
                border: 1px solid #1F2937;
                border-radius: 10px;
                padding: 12px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                height: 100%;
                min-height: 640px;
                max-height: 700px;
            }}

            .timeline-header {{
                font-size: 11px;
                font-weight: 700;
                color: #9CA3AF;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding-bottom: 8px;
                border-bottom: 1px solid #1F2937;
            }}

            .timeline-list {{
                display: flex;
                flex-direction: column;
                gap: 8px;
                overflow-y: auto;
                padding-right: 4px;
            }}

            .timeline-item {{
                font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
                font-size: 10px;
                line-height: 1.4;
                color: #E5E7EB;
                background: #0F172A;
                border: 1px solid #1F2937;
                border-radius: 6px;
                padding: 6px 8px;
                display: flex;
                align-items: flex-start;
                gap: 6px;
            }}

            .timeline-item-failed {{
                border-color: rgba(239, 68, 68, 0.4);
                background: rgba(239, 68, 68, 0.08);
                color: #FCA5A5;
            }}

            .timeline-ts {{
                color: #6B7280;
                flex-shrink: 0;
            }}

            /* Interactive Data Inspector Modal */
            .data-modal-overlay {{
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(6px);
                z-index: 99;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                animation: fadeIn 0.2s ease-out;
            }}

            .data-modal {{
                background: #111827;
                border: 1px solid #1F2937;
                border-radius: 12px;
                width: 92%;
                max-width: 500px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.6);
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 12px;
                position: relative;
            }}

            .data-modal-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding-bottom: 8px;
                border-bottom: 1px solid #1F2937;
            }}

            .data-modal-title {{
                font-size: 13px;
                font-weight: 700;
                color: #F3F4F6;
                display: flex;
                align-items: center;
                gap: 6px;
            }}

            .close-modal-btn {{
                background: transparent;
                border: none;
                color: #9CA3AF;
                font-size: 16px;
                cursor: pointer;
            }}
            .close-modal-btn:hover {{ color: #EF4444; }}

            .human-summary-box {{
                background: #0F172A;
                border: 1px solid #1F2937;
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 11px;
                color: #E5E7EB;
                line-height: 1.5;
            }}

            .json-code-block {{
                background: #0F172A;
                border: 1px solid #1F2937;
                border-radius: 8px;
                padding: 10px;
                font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
                font-size: 10.5px;
                color: #10B981;
                overflow-x: auto;
                max-height: 180px;
                white-space: pre-wrap;
            }}

            /* Connector Path Animations */
            .connector-path {{
                stroke: #374151;
                stroke-width: 1.5;
                fill: none;
                transition: stroke 0.3s ease;
            }}
            .connector-path-transferring {{
                stroke: #F59E0B;
                stroke-width: 2;
                stroke-dasharray: 5 3;
                animation: dashFlow 0.9s linear infinite;
            }}
            .connector-path-completed {{
                stroke: #10B981;
                stroke-width: 1.5;
                stroke-dasharray: none;
                animation: none;
            }}
            .connector-path-failed {{
                stroke: #EF4444;
                stroke-width: 1.5;
                stroke-dasharray: 4 4;
                animation: none;
            }}

            @keyframes dashFlow {{
                to {{ stroke-dashoffset: -16; }}
            }}
            @keyframes subtleAmberPulse {{
                0% {{ border-color: rgba(245, 158, 11, 0.5); box-shadow: 0 0 6px rgba(245, 158, 11, 0.15); }}
                50% {{ border-color: rgba(245, 158, 11, 1); box-shadow: 0 0 14px rgba(245, 158, 11, 0.35); }}
                100% {{ border-color: rgba(245, 158, 11, 0.5); box-shadow: 0 0 6px rgba(245, 158, 11, 0.15); }}
            }}
            @keyframes packetPulse {{
                0% {{ transform: translate(-50%, 0%) scale(1); }}
                50% {{ transform: translate(-50%, 0%) scale(1.03); }}
                100% {{ transform: translate(-50%, 0%) scale(1); }}
            }}
            @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

            /* Scrollbars */
            ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
            ::-webkit-scrollbar-track {{ background: #0F172A; }}
            ::-webkit-scrollbar-thumb {{ background: #374151; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="exec-panel">
            <!-- Header -->
            <div class="panel-header">
                <div class="panel-title-group">
                    <div class="panel-icon">⚡</div>
                    <span class="panel-title">System Execution Panel</span>
                    <span class="badge {badge_class}">{badge_label}</span>
                    <span class="badge {status_class}">{status_label}</span>
                </div>

                <div class="panel-controls">
                    <span class="duration-pill">⏱️ Total: {total_dur}</span>
                </div>
            </div>

            <!-- Body Grid -->
            <div class="panel-body">
                <!-- Left: 2D Graph Viewport -->
                <div class="graph-viewport" id="graphViewport">
                    <svg class="svg-layer" id="svgLayer">
                        <defs>
                            <marker id="arrow-waiting" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                                <polygon points="0 0, 6 3, 0 6" fill="#374151"/>
                            </marker>
                            <marker id="arrow-running" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                                <polygon points="0 0, 6 3, 0 6" fill="#F59E0B"/>
                            </marker>
                            <marker id="arrow-completed" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                                <polygon points="0 0, 6 3, 0 6" fill="#10B981"/>
                            </marker>
                            <marker id="arrow-failed" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                                <polygon points="0 0, 6 3, 0 6" fill="#EF4444"/>
                            </marker>
                        </defs>
                        <g id="connectionsGroup"></g>
                        <g id="packetsGroup"></g>
                    </svg>

                    <div class="packet-cards-layer" id="packetCardsLayer"></div>

                    <div class="graph-rows" id="graphRows">
                        <!-- Row 1: User Query -->
                        <div class="graph-row">
                            <div class="node-card" id="node-user_query" data-node="user_query" onclick="openNodeInspector('user_query')">
                                <div class="node-icon-wrapper">👤</div>
                                <div class="node-info">
                                    <div class="node-name">User Query</div>
                                    <div class="node-meta"><span id="meta-user_query">User Trip Details</span></div>
                                </div>
                            </div>
                        </div>

                        <!-- Row 2: Planner Agent -->
                        <div class="graph-row">
                            <div class="node-card" id="node-planner_agent" data-node="planner_agent" onclick="openNodeInspector('planner_agent')">
                                <div class="node-icon-wrapper">🧠</div>
                                <div class="node-info">
                                    <div class="node-name">Planner Agent</div>
                                    <div class="node-meta"><span id="meta-planner_agent">Route & Budget Insights</span></div>
                                </div>
                            </div>
                        </div>

                        <!-- Row 3: Transport Router -->
                        <div class="graph-row">
                            <div class="node-card" id="node-transport_decision" data-node="transport_decision" onclick="openNodeInspector('transport_decision')">
                                <div class="node-icon-wrapper">🔀</div>
                                <div class="node-info">
                                    <div class="node-name">Transport Router</div>
                                    <div class="node-meta"><span id="meta-transport_decision">Transport Decision</span></div>
                                </div>
                            </div>
                        </div>

                        <!-- Row 4: Railway & Flight Agents (Branching + MCP API Nodes) -->
                        <div class="graph-row row-split">
                            <div class="agent-group">
                                <div class="node-card" id="node-railway_agent" data-node="railway_agent" onclick="openNodeInspector('railway_agent')">
                                    <div class="node-icon-wrapper">🚆</div>
                                    <div class="node-info">
                                        <div class="node-name">Railway Agent</div>
                                        <div class="node-meta"><span id="meta-railway_agent">Train Schedules</span></div>
                                    </div>
                                </div>
                                <div class="api-card" id="api-railradar_api" onclick="openApiInspector('railway_agent')">
                                    <div class="api-header"><span>🚄</span> RailRadar / IRCTC</div>
                                    <div class="human-pill pill-input" id="req-railradar_api">🔍 Action: Train Search</div>
                                    <div class="human-pill pill-output" id="res-railradar_api">✅ Result: Train Options</div>
                                </div>
                            </div>

                            <div class="agent-group">
                                <div class="node-card" id="node-flight_agent" data-node="flight_agent" onclick="openNodeInspector('flight_agent')">
                                    <div class="node-icon-wrapper">✈️</div>
                                    <div class="node-info">
                                        <div class="node-name">Flight Agent</div>
                                        <div class="node-meta"><span id="meta-flight_agent">Flight Options</span></div>
                                    </div>
                                </div>
                                <div class="api-card" id="api-aviation_api" onclick="openApiInspector('flight_agent')">
                                    <div class="api-header"><span>🛫</span> AviationStack API</div>
                                    <div class="human-pill pill-input" id="req-aviation_api">🔍 Action: Flight Search</div>
                                    <div class="human-pill pill-output" id="res-aviation_api">✅ Result: Flight Options</div>
                                </div>
                            </div>
                        </div>

                        <!-- Row 5: Hotel Agent (Merging) -->
                        <div class="graph-row">
                            <div class="agent-group">
                                <div class="node-card" id="node-hotel_agent" data-node="hotel_agent" onclick="openNodeInspector('hotel_agent')">
                                    <div class="node-icon-wrapper">🏨</div>
                                    <div class="node-info">
                                        <div class="node-name">Hotel Agent</div>
                                        <div class="node-meta"><span id="meta-hotel_agent">Hotel Listings</span></div>
                                    </div>
                                </div>
                                <div class="api-card" id="api-hotel_api" onclick="openApiInspector('hotel_agent')">
                                    <div class="api-header"><span>🏨</span> Hotel Search API</div>
                                    <div class="human-pill pill-input" id="req-hotel_api">🔍 Action: Hotel Search</div>
                                    <div class="human-pill pill-output" id="res-hotel_api">✅ Result: Hotel Options</div>
                                </div>
                            </div>
                        </div>

                        <!-- Row 6: Weather Agent -->
                        <div class="graph-row">
                            <div class="agent-group">
                                <div class="node-card" id="node-weather_agent" data-node="weather_agent" onclick="openNodeInspector('weather_agent')">
                                    <div class="node-icon-wrapper">🌦️</div>
                                    <div class="node-info">
                                        <div class="node-name">Weather Agent</div>
                                        <div class="node-meta"><span id="meta-weather_agent">Weather Forecast Data</span></div>
                                    </div>
                                </div>
                                <div class="api-card" id="api-weather_api" onclick="openApiInspector('weather_agent')">
                                    <div class="api-header"><span>🌤️</span> Weather API</div>
                                    <div class="human-pill pill-input" id="req-weather_api">🔍 Action: Weather Check</div>
                                    <div class="human-pill pill-output" id="res-weather_api">✅ Result: Weather Data</div>
                                </div>
                            </div>
                        </div>

                        <!-- Row 7: Itinerary Agent -->
                        <div class="graph-row">
                            <div class="node-card" id="node-itinerary_agent" data-node="itinerary_agent" onclick="openNodeInspector('itinerary_agent')">
                                <div class="node-icon-wrapper">🗓️</div>
                                <div class="node-info">
                                    <div class="node-name">Itinerary Agent</div>
                                    <div class="node-meta"><span id="meta-itinerary_agent">Daily Itinerary Plan</span></div>
                                </div>
                            </div>
                        </div>

                        <!-- Row 8: Final Response & Exporter -->
                        <div class="graph-row row-split">
                            <div class="node-card" id="node-final_response" data-node="final_response" onclick="openNodeInspector('final_response')">
                                <div class="node-icon-wrapper">💬</div>
                                <div class="node-info">
                                    <div class="node-name">Final Response</div>
                                    <div class="node-meta"><span id="meta-final_response">Plan Summary</span></div>
                                </div>
                            </div>

                            <div class="node-card" id="node-download_builder" data-node="download_builder" onclick="openNodeInspector('download_builder')">
                                <div class="node-icon-wrapper">📄</div>
                                <div class="node-info">
                                    <div class="node-name">Download Builder</div>
                                    <div class="node-meta"><span id="meta-download_builder">Trip Dossier</span></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Right: Timeline Event Log Panel -->
                <div class="timeline-panel">
                    <div class="timeline-header">
                        <span>Execution Event Log</span>
                        <span style="font-size:9px; color:#6B7280;">{mode.upper()}</span>
                    </div>
                    <div class="timeline-list" id="timelineList"></div>
                </div>
            </div>

            <!-- Data Inspector Overlay Modal -->
            <div class="data-modal-overlay" id="dataModal" style="display: none;">
                <div class="data-modal">
                    <div class="data-modal-header">
                        <div class="data-modal-title" id="modalTitle">📄 Data Transfer Details</div>
                        <button class="close-modal-btn" onclick="closeDataModal()">&times;</button>
                    </div>

                    <div class="human-summary-box" id="modalHumanSummary"></div>

                    <div style="display:flex; align-items:center; justify-content:space-between; margin-top:4px;">
                        <span style="font-size:10px; color:#9CA3AF;">Structured Data Payload</span>
                        <button class="replay-btn" style="padding:2px 8px; font-size:10px;" onclick="toggleJsonView()">Toggle Raw JSON</button>
                    </div>

                    <pre class="json-code-block" id="modalJson" style="display: none;"></pre>

                    <div style="display:flex; justify-content:flex-end; gap:8px;">
                        <button class="replay-btn" onclick="copyModalJson()">📋 Copy Payload</button>
                        <button class="replay-btn" style="background:#1F2937;" onclick="closeDataModal()">Close</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const nodesState = {nodes_state_json};
            const timelineData = {timeline_json};
            const catalog = {catalog_json};
            const executedNodes = {executed_nodes_json};
            const apiNodes = {api_nodes_json};
            const apiStates = {api_states_json};
            const runtimeData = {runtime_data_json};
            const currentMode = "{mode}";

            // DYNAMIC RUNTIME STATE VALUES
            const src = runtimeData.source || "Origin";
            const dest = runtimeData.destination || "Destination";
            const bgt = runtimeData.budget || "Standard";
            const dts = runtimeData.dates || "Scheduled";

            // Dynamic API Pill text binding
            document.getElementById("req-aviation_api").textContent = `🔍 Action: Flight Search (${{src}} → ${{dest}})`;
            document.getElementById("res-aviation_api").textContent = `✅ Result: Flights Found`;

            document.getElementById("req-railradar_api").textContent = `🔍 Action: Train Search (${{src}} → ${{dest}})`;
            document.getElementById("res-railradar_api").textContent = `✅ Result: Trains Found`;

            document.getElementById("req-hotel_api").textContent = `🔍 Action: Hotel Search (${{dest}} • ${{bgt}})`;
            document.getElementById("res-hotel_api").textContent = `✅ Result: Stays Found`;

            document.getElementById("req-weather_api").textContent = `🔍 Action: Weather Check (${{dest}})`;
            document.getElementById("res-weather_api").textContent = `✅ Result: Forecast Fetched`;

            // Connection Specifications with Dynamic Runtime State Binding & Inter-Agent Parameter Passing
            const connections = [
                {{
                    id: "c_user_planner",
                    from: "user_query", to: "planner_agent",
                    label: "User Query",
                    summary: `➡️ Passing to Planner: ${{src}} → ${{dest}} • ${{bgt}} • ${{dts}}`,
                    raw_data: {{ "origin": src, "destination": dest, "budget": bgt, "dates": dts }}
                }},
                {{
                    id: "c_planner_router",
                    from: "planner_agent", to: "transport_decision",
                    label: "Planner Intent",
                    summary: `➡️ Passing to Transport Router: ${{src}} → ${{dest}} • ${{bgt}}`,
                    raw_data: {{ "origin": src, "destination": dest, "budget": bgt, "duration": dts }}
                }},
                {{
                    id: "c_router_railway",
                    from: "transport_decision", to: "railway_agent",
                    label: "Train Query",
                    summary: `➡️ Passing to Railway Agent: Route ${{src}} → ${{dest}}`,
                    raw_data: {{ "origin": src, "destination": dest, "mode": "railway" }}
                }},
                {{
                    id: "c_router_flight",
                    from: "transport_decision", to: "flight_agent",
                    label: "Flight Query",
                    summary: `➡️ Passing to Flight Agent: Route ${{src}} → ${{dest}}`,
                    raw_data: {{ "origin": src, "destination": dest, "mode": "flight" }}
                }},
                {{
                    id: "c_railway_hotel",
                    from: "railway_agent", to: "hotel_agent",
                    label: "Railway Data",
                    summary: `➡️ Passing to Hotel Agent: Destination ${{dest}} • Budget ${{bgt}}`,
                    raw_data: {{ "origin": src, "destination": dest, "mode": "railway", "status": "retrieved" }}
                }},
                {{
                    id: "c_flight_hotel",
                    from: "flight_agent", to: "hotel_agent",
                    label: "Flight Data",
                    summary: `➡️ Passing to Hotel Agent: Destination ${{dest}} • Budget ${{bgt}}`,
                    raw_data: {{ "origin": src, "destination": dest, "mode": "flight", "status": "retrieved" }}
                }},
                {{
                    id: "c_hotel_weather",
                    from: "hotel_agent", to: "weather_agent",
                    label: "Hotel Data",
                    summary: `➡️ Passing to Weather Agent: Destination ${{dest}}`,
                    raw_data: {{ "destination": dest, "stay_type": bgt, "status": "matched" }}
                }},
                {{
                    id: "c_weather_itinerary",
                    from: "weather_agent", to: "itinerary_agent",
                    label: "Weather Data",
                    summary: `➡️ Passing to Itinerary Agent: Destination ${{dest}} • Dates ${{dts}}`,
                    raw_data: {{ "destination": dest, "days": dts, "status": "retrieved" }}
                }},
                {{
                    id: "c_itinerary_final",
                    from: "itinerary_agent", to: "final_response",
                    label: "Itinerary Plan",
                    summary: `➡️ Passing to Final Response: Full Trip Plan for ${{dest}}`,
                    raw_data: {{ "destination": dest, "dates": dts, "status": "generated" }}
                }},
                {{
                    id: "c_final_download",
                    from: "final_response", to: "download_builder",
                    label: "Plan Summary",
                    summary: `➡️ Passing to Exporter: Dossier for ${{dest}}`,
                    raw_data: {{ "destination": dest, "format": "Markdown", "status": "ready" }}
                }}
            ];

            // Render ALL Nodes in Graph (Skipped nodes marked "Not Used", Failed marked "Execution Failed")
            function applyNodeStates(statesMap, activeExecuted) {{
                Object.keys(catalog).forEach(nodeId => {{
                    const card = document.getElementById(`node-${{nodeId}}`);
                    if (!card) return;

                    const isExecuted = activeExecuted.includes(nodeId);
                    const state = statesMap[nodeId] || {{ status: "waiting" }};

                    card.className = "node-card";
                    if (!isExecuted || state.status === "skipped") {{
                        card.classList.add("state-skipped");
                    }} else {{
                        card.classList.add(`state-${{state.status}}`);
                    }}

                    // Node Subtext Handling
                    const metaEl = document.getElementById(`meta-${{nodeId}}`);
                    if (metaEl) {{
                        if (state.status === "failed") {{
                            metaEl.innerHTML = `<span style="color:#EF4444; font-weight:600;">⚠️ Execution Failed</span>`;
                        }} else if (!isExecuted || state.status === "skipped") {{
                            metaEl.innerHTML = `<span>Not Used</span>`;
                        }} else {{
                            const durText = state.duration ? ` • ${{state.duration}}` : "";
                            metaEl.innerHTML = `<span>${{catalog[nodeId].packet_label}}${{durText}}</span>`;
                        }}
                    }}
                }});

                // API Cards
                Object.keys(apiNodes).forEach(agentId => {{
                    const apiInfo = apiNodes[agentId];
                    const apiCard = document.getElementById(`api-${{apiInfo.id}}`);
                    if (!apiCard) return;

                    const agentState = statesMap[agentId] || {{ status: "waiting" }};
                    apiCard.className = "api-card";
                    if (agentState.status === "running") {{
                        apiCard.classList.add("api-card-active");
                    }} else if (agentState.status === "completed") {{
                        apiCard.classList.add("api-card-complete");
                    }} else if (agentState.status === "failed") {{
                        apiCard.classList.add("api-card-failed");
                        const resPill = document.getElementById(`res-${{apiInfo.id}}`);
                        if (resPill) {{
                            resPill.className = "human-pill pill-failed";
                            resPill.textContent = "Execution Failed";
                        }}
                    }}
                }});
            }}

            // Render Timeline Event Log
            function renderTimeline(timelineEvents) {{
                const list = document.getElementById("timelineList");
                list.innerHTML = "";
                if (!timelineEvents || timelineEvents.length === 0) {{
                    list.innerHTML = '<div style="font-size:10px; color:#6B7280; font-family:monospace;">Awaiting pipeline events...</div>';
                    return;
                }}

                timelineEvents.slice().reverse().forEach(item => {{
                    const row = document.createElement("div");
                    const isErr = item.type === "error";
                    row.className = `timeline-item ${{isErr ? 'timeline-item-failed' : ''}}`;
                    row.innerHTML = `
                        <span class="timeline-ts">[${{item.time}}]</span>
                        <span>${{item.icon}} ${{item.message}}</span>
                    `;
                    list.appendChild(row);
                }});
            }}

            // Calculate Per-Edge State Management, Connector Animations & HARD POSITION CONSTRAINTS BELOW CARDS
            function renderConnectors(statesMap, activeExecuted) {{
                const svgLayer = document.getElementById("svgLayer");
                const connGroup = document.getElementById("connectionsGroup");
                const packetsGroup = document.getElementById("packetsGroup");
                const cardsLayer = document.getElementById("packetCardsLayer");
                const viewport = document.getElementById("graphViewport").getBoundingClientRect();

                connGroup.innerHTML = "";
                packetsGroup.innerHTML = "";
                cardsLayer.innerHTML = "";

                connections.forEach(conn => {{
                    const fromCard = document.getElementById(`node-${{conn.from}}`);
                    const toCard = document.getElementById(`node-${{conn.to}}`);
                    if (!fromCard || !toCard) return;

                    // Determine true source bottom (accounting for API cards under agent nodes)
                    let srcBottomRel = fromCard.getBoundingClientRect().bottom - viewport.top;
                    if (apiNodes[conn.from]) {{
                        const apiCard = document.getElementById(`api-${{apiNodes[conn.from].id}}`);
                        if (apiCard) {{
                            srcBottomRel = apiCard.getBoundingClientRect().bottom - viewport.top;
                        }}
                    }}

                    const fromRect = fromCard.getBoundingClientRect();
                    const toRect = toCard.getBoundingClientRect();

                    const x1 = fromRect.left + fromRect.width / 2 - viewport.left;
                    const y1 = srcBottomRel;
                    const x2 = toRect.left + toRect.width / 2 - viewport.left;
                    const y2 = toRect.top - viewport.top;

                    const fromState = statesMap[conn.from] || {{ status: "waiting" }};
                    const toState = statesMap[conn.to] || {{ status: "waiting" }};

                    // Per-Edge State Logic
                    let edgeState = "waiting";
                    let markerId = "arrow-waiting";

                    if (fromState.status === "failed") {{
                        edgeState = "failed"; // Red broken line!
                        markerId = "arrow-failed";
                    }} else if (fromState.status === "completed" && toState.status === "completed") {{
                        edgeState = "completed"; // Solid green line, FIXED packet icon!
                        markerId = "arrow-completed";
                    }} else if (fromState.status === "completed" && (toState.status === "running" || toState.status === "waiting")) {{
                        edgeState = "transferring"; // Animated dash line & moving packet!
                        markerId = "arrow-running";
                    }} else if (fromState.status === "running") {{
                        edgeState = "transferring";
                        markerId = "arrow-running";
                    }}

                    // Bezier Curve
                    const controlY = (y1 + y2) / 2;
                    const pathData = `M ${{x1}} ${{y1}} C ${{x1}} ${{controlY}}, ${{x2}} ${{controlY}}, ${{x2}} ${{y2}}`;

                    const pathEl = document.createElementNS("http://www.w3.org/2000/svg", "path");
                    pathEl.setAttribute("d", pathData);
                    pathEl.setAttribute("class", `connector-path connector-path-${{edgeState}}`);
                    pathEl.setAttribute("marker-end", `url(#${{markerId}})`);
                    connGroup.appendChild(pathEl);

                    // HARD POSITIONING CONSTRAINT: Packet Card anchored STRICTLY BELOW source card / API card
                    let labelY = srcBottomRel + 14; // Minimum 14px below source bottom
                    let labelX = (x1 + x2) / 2;

                    // Ensure label never overlaps target node top
                    const targetTopRel = toRect.top - viewport.top;
                    if (labelY + 24 > targetTopRel - 10) {{
                        labelY = Math.max(srcBottomRel + 10, targetTopRel - 32);
                    }}

                    // Offset horizontally for split rows (Railway vs Flight)
                    if (conn.from === "railway_agent") {{
                        labelX = x1 - 10;
                    }} else if (conn.from === "flight_agent") {{
                        labelX = x1 + 10;
                    }}

                    // If Transferring: Animated Moving SVG File Icon along path
                    if (edgeState === "transferring") {{
                        const packetG = document.createElementNS("http://www.w3.org/2000/svg", "g");
                        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                        circle.setAttribute("r", "5");
                        circle.setAttribute("fill", "#F59E0B");

                        const anim = document.createElementNS("http://www.w3.org/2000/svg", "animateMotion");
                        anim.setAttribute("path", pathData);
                        anim.setAttribute("dur", "1.8s");
                        anim.setAttribute("repeatCount", "indefinite");

                        packetG.appendChild(circle);
                        packetG.appendChild(anim);
                        packetsGroup.appendChild(packetG);
                    }} 
                    // If Completed: Fixed Static Circle Icon at Midpoint (NO MOTION!)
                    else if (edgeState === "completed") {{
                        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                        circle.setAttribute("cx", labelX);
                        circle.setAttribute("cy", labelY + 12);
                        circle.setAttribute("r", "4");
                        circle.setAttribute("fill", "#10B981");
                        packetsGroup.appendChild(circle);
                    }}
                    // If Failed: Warning Red Dot
                    else if (edgeState === "failed") {{
                        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                        circle.setAttribute("cx", labelX);
                        circle.setAttribute("cy", labelY + 12);
                        circle.setAttribute("r", "5");
                        circle.setAttribute("fill", "#EF4444");
                        packetsGroup.appendChild(circle);
                    }}

                    // HTML Data Packet Preview Card (Positioned STRICTLY BELOW Source Card)
                    const cardDiv = document.createElement("div");
                    cardDiv.className = `packet-card packet-card-${{edgeState}}`;
                    cardDiv.style.left = `${{labelX}}px`;
                    cardDiv.style.top = `${{labelY}}px`;
                    cardDiv.onclick = (e) => {{
                        e.stopPropagation();
                        openDataInspector(conn.label, conn.from, conn.to, conn.summary, conn.raw_data, edgeState);
                    }};

                    const cardIcon = edgeState === "failed" ? "⚠️" : "📄";
                    const displaySummary = edgeState === "failed" ? "Data transfer unavailable" : conn.summary;

                    cardDiv.innerHTML = `
                        <span>${{cardIcon}}</span>
                        <strong style="color:#F3F4F6;">${{conn.label}}</strong>
                        <span class="packet-summary-text">${{displaySummary}}</span>
                    `;
                    cardsLayer.appendChild(cardDiv);
                }});
            }}

            // Data Modal Inspector Handler with Human Summary & Optional Raw JSON Toggle
            function openDataInspector(label, fromNode, toNode, humanSummary, rawData, state) {{
                const modal = document.getElementById("dataModal");
                document.getElementById("modalTitle").innerHTML = `📄 Data Flow: ${{label}}`;
                document.getElementById("modalHumanSummary").innerHTML = `
                    <div style="font-weight:700; color:#F3F4F6; margin-bottom:4px;">Flow: ${{catalog[fromNode]?.name || fromNode}} → ${{catalog[toNode]?.name || toNode}} [${{state.toUpperCase()}}]</div>
                    <div>${{humanSummary}}</div>
                `;
                document.getElementById("modalJson").textContent = JSON.stringify(rawData, null, 2);
                document.getElementById("modalJson").style.display = "none";
                modal.style.display = "flex";
            }}

            function openNodeInspector(nodeId) {{
                const nodeData = nodesState[nodeId] || {{}};
                const meta = catalog[nodeId] || {{ name: nodeId }};
                const modal = document.getElementById("dataModal");

                const inspectorPayload = {{
                    "node_name": meta.name,
                    "category": meta.category,
                    "status": nodeData.status || "waiting",
                    "duration": nodeData.duration || "N/A",
                    "summary": meta.description,
                    "runtime_state": {{ "source": src, "destination": dest, "budget": bgt, "dates": dts }}
                }};

                document.getElementById("modalTitle").innerHTML = `${{meta.icon || "⚙️"}} Agent: ${{meta.name}}`;
                document.getElementById("modalHumanSummary").innerHTML = `
                    <div style="font-weight:700; color:#F3F4F6; margin-bottom:4px;">${{meta.name}} [${{(nodeData.status || "waiting").toUpperCase()}}]</div>
                    <div>${{meta.description}}</div>
                `;
                document.getElementById("modalJson").textContent = JSON.stringify(inspectorPayload, null, 2);
                document.getElementById("modalJson").style.display = "none";
                modal.style.display = "flex";
            }}

            function openApiInspector(agentId) {{
                const apiInfo = apiNodes[agentId];
                if (!apiInfo) return;
                const modal = document.getElementById("dataModal");

                document.getElementById("modalTitle").innerHTML = `${{apiInfo.icon}} Search Service: ${{apiInfo.name}}`;
                document.getElementById("modalHumanSummary").innerHTML = `
                    <div style="font-weight:700; color:#F3F4F6; margin-bottom:4px;">${{apiInfo.name}}</div>
                    <div style="color:#F59E0B; margin-bottom:2px;">• Search Input: ${{src}} → ${{dest}}</div>
                    <div style="color:#10B981;">• Results Found: Options matched</div>
                `;

                const rawObj = {{
                    "service": apiInfo.name,
                    "search_route": `${{src}} → ${{dest}}`,
                    "status": (nodesState[agentId] || {{}}).status || "completed"
                }};
                document.getElementById("modalJson").textContent = JSON.stringify(rawObj, null, 2);
                document.getElementById("modalJson").style.display = "none";
                modal.style.display = "flex";
            }}

            function toggleJsonView() {{
                const el = document.getElementById("modalJson");
                el.style.display = el.style.display === "none" ? "block" : "none";
            }}

            function closeDataModal() {{
                document.getElementById("dataModal").style.display = "none";
            }}

            function copyModalJson() {{
                const txt = document.getElementById("modalJson").textContent;
                navigator.clipboard.writeText(txt);
                alert("Copied to clipboard!");
            }}

            // Initial Render
            applyNodeStates(nodesState, executedNodes);
            renderTimeline(timelineData);
            setTimeout(() => renderConnectors(nodesState, executedNodes), 100);

            window.addEventListener("resize", () => renderConnectors(nodesState, executedNodes));

            // Replay Mode Simulation
            let isReplaying = false;
            function startReplay() {{
                if (isReplaying) return;
                isReplaying = true;

                const replayStates = {{}};
                Object.keys(catalog).forEach(id => {{
                    replayStates[id] = {{ status: "waiting" }};
                }});

                applyNodeStates(replayStates, executedNodes);
                renderConnectors(replayStates, executedNodes);

                const activeSequence = ["user_query", "planner_agent", "transport_decision", "railway_agent", "flight_agent", "hotel_agent", "weather_agent", "itinerary_agent", "final_response", "download_builder"].filter(id => executedNodes.includes(id));

                let stepIndex = 0;
                const interval = setInterval(() => {{
                    if (stepIndex > 0) {{
                        const prevId = activeSequence[stepIndex - 1];
                        replayStates[prevId] = {{ status: "completed", duration: catalog[prevId].duration || "<1s" }};
                    }}

                    if (stepIndex < activeSequence.length) {{
                        const currentId = activeSequence[stepIndex];
                        replayStates[currentId] = {{ status: "running" }};
                        applyNodeStates(replayStates, executedNodes);
                        renderConnectors(replayStates, executedNodes);
                        stepIndex++;
                    }} else {{
                        clearInterval(interval);
                        applyNodeStates(nodesState, executedNodes);
                        renderConnectors(nodesState, executedNodes);
                        isReplaying = false;
                    }}
                }}, 750);
            }}
        </script>
    </body>
    </html>
    """
    return html_content


def render_agent_visualizer(target=None, visible: bool = True, snapshot_data=None, mode: str = "live"):
    """
    Renders the Inline Full-Width System Execution Panel.
    Supports Live execution streaming as well as Replay mode from saved snapshots.
    """
    if not visible or not st.session_state.get("show_visualizer", False):
        if target is not None:
            target.empty()
        return

    html_code = generate_execution_panel_html(snapshot_data=snapshot_data, mode=mode)

    if target is not None:
        with target.container():
            components.html(html_code, height=680, scrolling=True)
    else:
        components.html(html_code, height=680, scrolling=True)


# Backward Compatibility Alias
generate_phone_modal_html = generate_execution_panel_html
