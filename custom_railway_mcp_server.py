from mcp.server.fastmcp import FastMCP
import requests
import os
import re
import json
import urllib.parse
from dotenv import load_dotenv

load_dotenv(override=True)

mcp = FastMCP("Railway Server")

RAILRADAR_API_KEY = os.getenv("RAILRADAR_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

STATION_MAP = {
    "MUMBAI": [("BCT", "Mumbai Central"), ("CSMT", "Chhatrapati Shivaji Maharaj Terminus"), ("LTT", "Lokmanya Tilak Terminus")],
    "DELHI": [("NDLS", "New Delhi"), ("DLI", "Old Delhi"), ("NZM", "Hazrat Nizamuddin")],
    "NEW DELHI": [("NDLS", "New Delhi")],
    "JAIPUR": [("JP", "Jaipur Junction")],
    "GOA": [("MAO", "Madgaon Junction"), ("THVM", "Thivim"), ("KRMI", "Karmali")],
    "BANGALORE": [("SBC", "KSR Bengaluru"), ("YPR", "Yesvantpur Junction")],
    "BENGALURU": [("SBC", "KSR Bengaluru"), ("YPR", "Yesvantpur Junction")],
    "CHENNAI": [("MAS", "Chennai Central"), ("MS", "Chennai Egmore")],
    "KOLKATA": [("HWH", "Howrah Junction"), ("SDAH", "Sealdah")],
    "HYDERABAD": [("SC", "Secunderabad Junction"), ("HYB", "Hyderabad Deccan")],
    "PUNE": [("PUNE", "Pune Junction")],
    "AHMEDABAD": [("ADI", "Ahmedabad Junction")],
    "AGRA": [("AGC", "Agra Cantt")],
    "VARANASI": [("BSB", "Varanasi Junction")],
    "AMRITSAR": [("ASR", "Amritsar Junction")],
    "UDAIPUR": [("UDZ", "Udaipur City")],
    "KOCHI": [("ERS", "Ernakulam Junction")],
    "COCHIN": [("ERS", "Ernakulam Junction")],
    "LUCKNOW": [("LKO", "Lucknow Charbagh")],
    "CHANDIGARH": [("CDG", "Chandigarh Junction")],
    "BHOPAL": [("BPL", "Bhopal Junction"), ("RKMP", "Rani Kamlapati")],
    "PATNA": [("PNBE", "Patna Junction")],
    "NAGPUR": [("NGP", "Nagpur Junction")],
    "SURAT": [("ST", "Surat")],
    "VADODARA": [("BRC", "Vadodara Junction")],
    "SHIMLA": [("SML", "Shimla")],
    "DEHRADUN": [("DDN", "Dehradun")],
    "GUWAHATI": [("GHY", "Guwahati")],
    "JODHPUR": [("JU", "Jodhpur Junction")],
    "JAISALMER": [("JSM", "Jaisalmer")],
    "RISHIKESH": [("YNRK", "Yog Nagari Rishikesh")],
    "HARIDWAR": [("HW", "Haridwar Junction")],
    "KERALA": [("ERS", "Ernakulam Junction (Kochi)"), ("TVC", "Thiruvananthapuram Central")],
    "KERELA": [("ERS", "Ernakulam Junction (Kochi)")],
}

def resolve_station(city_or_station: str):
    clean = (city_or_station or "").strip().upper()
    clean_city = clean.split(",")[0].strip()
    if clean_city in STATION_MAP:
        return STATION_MAP[clean_city][0]
    for k, v in STATION_MAP.items():
        if k in clean_city or clean_city in k:
            return v[0]
    return (clean_city[:4], clean_city.capitalize())

def tavily_train_fallback(src_name: str, src_code: str, dst_name: str, dst_code: str):
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return None
    try:
        query = f"popular direct trains from {src_name} ({src_code}) to {dst_name} ({dst_code}) schedule timing fare vande bharat express"
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "search_depth": "basic", "max_results": 5},
            timeout=8
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

@mcp.tool()
def search_trains(source: str, destination: str):
    """
    Search train options between source and destination stations or cities using RailRadar API with station mapping & web search fallback.
    """
    src_code, src_full = resolve_station(source)
    dst_code, dst_full = resolve_station(destination)

    api_key = os.getenv("RAILRADAR_API_KEY", "")
    headers = {}
    if api_key and api_key != "your_api_key_here":
        headers["Authorization"] = f"Bearer {api_key}"

    # Try official RailRadar API first
    try:
        url = f"https://api.railradar.in/v1/trains/between/{src_code}/{dst_code}"
        response = requests.get(url, headers=headers, params={"byCity": "true"}, timeout=6)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and data.get("data"):
                return data
    except Exception:
        pass

    # Direct Web Search Fallback for real Indian Railways schedules
    tavily_res = tavily_train_fallback(src_full, src_code, dst_full, dst_code)
    web_snippets = ""
    if isinstance(tavily_res, dict) and "results" in tavily_res:
        web_snippets = "\n".join(r.get("content", "") for r in tavily_res["results"])

    confirmtkt_url = f"https://www.confirmtkt.com/rly-search?from={urllib.parse.quote_plus(src_code)}&to={urllib.parse.quote_plus(dst_code)}"

    fallback_response = {
        "status": "success",
        "source": f"{src_full} ({src_code})",
        "destination": f"{dst_full} ({dst_code})",
        "source_code": src_code,
        "destination_code": dst_code,
        "raw_search_notes": web_snippets[:2000],
        "booking_platforms": [
            {"name": "IRCTC", "url": "https://www.irctc.co.in/nget/train-search"},
            {"name": "ConfirmTkt", "url": confirmtkt_url},
            {"name": "RailYatri", "url": "https://www.railyatri.in/train-tickets"}
        ],
        "tip": "Book AC coaches early during peak travel seasons as berths fill quickly."
    }

    return fallback_response

if __name__ == "__main__":
    mcp.run()
