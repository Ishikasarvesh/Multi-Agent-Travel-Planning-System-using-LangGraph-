import re
import urllib.parse
from datetime import datetime, timedelta

def normalize_city(city: str) -> str:
    """Normalize city name by stripping region/country and extra spaces."""
    if not city or not isinstance(city, str):
        return ""
    clean = city.split(",")[0].strip()
    return clean

def is_valid_url(value) -> bool:
    """Check if string is a valid web URL starting with http:// or https://."""
    if not value:
        return False
    try:
        val_str = str(value).strip()
        if val_str.lower() in ["none", "null", "n/a", "undefined", "no data"]:
            return False
        parsed = urllib.parse.urlparse(val_str)
        return (parsed.scheme in {"http", "https"}) and bool(parsed.netloc)
    except (TypeError, ValueError):
        return False

def parse_and_format_date(date_str: str, target_fmt: str = "%Y-%m-%d") -> str:
    """
    Safely parses various date string formats and converts to target_fmt.
    Returns formatted date string or empty string if unparseable/missing.
    """
    if not date_str or not isinstance(date_str, str):
        return ""
    cleaned = date_str.strip()
    formats_to_try = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
        "%d %b", "%d %B"
    ]
    current_year = datetime.now().year
    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(cleaned, fmt)
            if dt.year == 1900:
                dt = dt.replace(year=current_year)
            return dt.strftime(target_fmt)
        except ValueError:
            continue
    return ""

def calculate_return_date(dep_date_str: str, duration) -> str:
    """Calculates return date from departure date and trip duration in days."""
    dep_formatted = parse_and_format_date(dep_date_str, "%Y-%m-%d")
    if not dep_formatted:
        return ""
    try:
        dur_match = re.search(r'\d+', str(duration))
        if not dur_match:
            return ""
        dur_days = int(dur_match.group(0))
        if dur_days <= 0:
            return ""
        dep_dt = datetime.strptime(dep_formatted, "%Y-%m-%d")
        ret_dt = dep_dt + timedelta(days=dur_days)
        return ret_dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def build_flight_platform_links(
    source: str = "",
    destination: str = "",
    dep_date: str = "",
    duration: str = "",
    passengers: int = 1,
    cabin: str = "economy"
) -> list:
    """Generates prefilled deep search links for flight booking platforms."""
    src = normalize_city(source)
    dst = normalize_city(destination)
    if not dst:
        return [
            {"name": "MakeMyTrip", "url": "https://www.makemytrip.com/flights/"},
            {"name": "Google Flights", "url": "https://www.google.com/travel/flights"},
            {"name": "Skyscanner", "url": "https://www.skyscanner.net/"},
            {"name": "EaseMyTrip", "url": "https://www.easemytrip.com/flights.html"},
            {"name": "Booking.com Flights", "url": "https://www.booking.com/flights.html"}
        ]
    
    src_enc = urllib.parse.quote_plus(src)
    dst_enc = urllib.parse.quote_plus(dst)
    
    dep_fmt = parse_and_format_date(dep_date, "%Y-%m-%d") if dep_date else ""
    ret_fmt = calculate_return_date(dep_date, duration) if (dep_date and duration) else ""
    
    # 1. Google Flights
    if src and dep_fmt and ret_fmt:
        gflights_q = f"Flights from {src} to {dst} on {dep_fmt} through {ret_fmt}"
    elif src and dep_fmt:
        gflights_q = f"Flights from {src} to {dst} on {dep_fmt}"
    elif src:
        gflights_q = f"Flights from {src} to {dst}"
    else:
        gflights_q = f"Flights to {dst}"
    google_flights_url = f"https://www.google.com/travel/flights?q={urllib.parse.quote_plus(gflights_q)}"

    # 2. MakeMyTrip
    mmt_url = "https://www.makemytrip.com/flights/"

    # 3. Skyscanner
    skyscanner_url = f"https://www.skyscanner.net/g/referrals/v1/flights/day-single?origin={src_enc}&destination={dst_enc}" if src else "https://www.skyscanner.net/"

    # 4. EaseMyTrip
    easemytrip_url = f"https://www.easemytrip.com/flight-search/{src_enc}-{dst_enc}-{dep_fmt}" if (src and dep_fmt) else "https://www.easemytrip.com/flights.html"

    # 5. Booking.com Flights
    booking_flights_url = f"https://www.booking.com/flights/search.html?from={src_enc}&to={dst_enc}" if src else "https://www.booking.com/flights.html"

    return [
        {"name": "MakeMyTrip", "url": mmt_url},
        {"name": "Google Flights", "url": google_flights_url},
        {"name": "Skyscanner", "url": skyscanner_url},
        {"name": "EaseMyTrip", "url": easemytrip_url},
        {"name": "Booking.com Flights", "url": booking_flights_url},
    ]

def build_railway_platform_links(
    source: str = "",
    destination: str = "",
    dep_date: str = ""
) -> list:
    """Generates railway search links for platforms (using official parameter formats)."""
    src = normalize_city(source)
    dst = normalize_city(destination)
    
    irctc_url = "https://www.irctc.co.in/nget/train-search"
    
    if src and dst:
        confirmtkt_url = f"https://www.confirmtkt.com/rly-search?from={urllib.parse.quote_plus(src)}&to={urllib.parse.quote_plus(dst)}"
        railyatri_url = "https://www.railyatri.in/train-tickets"
    else:
        confirmtkt_url = "https://www.confirmtkt.com/"
        railyatri_url = "https://www.railyatri.in/"

    return [
        {"name": "IRCTC", "url": irctc_url},
        {"name": "ConfirmTkt", "url": confirmtkt_url},
        {"name": "RailYatri", "url": railyatri_url},
    ]

def build_hotel_platform_links(
    destination: str = "",
    checkin_date: str = "",
    duration: str = "",
    guests: int = 2
) -> list:
    """Generates prefilled hotel search links for hotel booking platforms."""
    dst = normalize_city(destination)
    if not dst:
        return [
            {"name": "Booking.com", "url": "https://www.booking.com/"},
            {"name": "MakeMyTrip Hotels", "url": "https://www.makemytrip.com/hotels/"},
            {"name": "Agoda", "url": "https://www.agoda.com/"},
            {"name": "Expedia", "url": "https://www.expedia.com/Hotels"},
            {"name": "Hotels.com", "url": "https://www.hotels.com/"},
            {"name": "Goibibo Hotels", "url": "https://www.goibibo.com/hotels/"},
        ]

    dst_enc = urllib.parse.quote_plus(dst)
    checkin_fmt = parse_and_format_date(checkin_date, "%Y-%m-%d") if checkin_date else ""
    checkout_fmt = calculate_return_date(checkin_date, duration) if (checkin_date and duration) else ""
    
    date_params_bcom = f"&checkin={checkin_fmt}&checkout={checkout_fmt}" if (checkin_fmt and checkout_fmt) else ""
    
    booking_url = f"https://www.booking.com/searchresults.html?ss={dst_enc}{date_params_bcom}"
    mmt_url = f"https://www.makemytrip.com/hotels/hotel-listing/?city={dst_enc}"
    agoda_url = f"https://www.agoda.com/search?city={dst_enc}"
    expedia_url = f"https://www.expedia.com/Hotel-Search?destination={dst_enc}"
    hotels_url = f"https://www.hotels.com/Hotel-Search?destination={dst_enc}"
    goibibo_url = f"https://www.goibibo.com/hotels/hotels-in-{dst_enc}-ct/"
    
    return [
        {"name": "Booking.com", "url": booking_url},
        {"name": "MakeMyTrip Hotels", "url": mmt_url},
        {"name": "Agoda", "url": agoda_url},
        {"name": "Expedia", "url": expedia_url},
        {"name": "Hotels.com", "url": hotels_url},
        {"name": "Goibibo Hotels", "url": goibibo_url},
    ]

def build_google_maps_link(attraction: str, city: str = "") -> str:
    """Generates a prefilled Google Maps location search URL for attractions."""
    attr_clean = attraction.strip() if attraction else ""
    city_clean = normalize_city(city)
    query = f"{attr_clean} {city_clean}".strip()
    return f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(query)}"
