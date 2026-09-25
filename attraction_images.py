import re
from typing import Optional
from urllib.parse import quote_plus
import requests
import streamlit as st
from url_builders import build_google_maps_link

WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"

def convert_markdown_links_to_target_blank(text: str) -> str:
    """
    Converts markdown links [Label](URL) in text to HTML anchor tags with target="_blank".
    """
    if not text or not isinstance(text, str):
        return ""
    
    def replacer(match):
        label = match.group(1).strip()
        url = match.group(2).strip()
        if url.startswith("http://") or url.startswith("https://"):
            return f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="color: #4ea8f0; text-decoration: underline;">{label}</a>'
        return label

    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replacer, text)

# Words that should not be treated as tourist attractions.
EXCLUDED_WORDS = {
    "airport",
    "hotel",
    "hostel",
    "restaurant",
    "breakfast",
    "lunch",
    "dinner",
    "transportation",
    "metro",
    "train",
    "bus",
    "flight",
    "check-in",
    "check-out",
    "arrival",
    "departure",
    "free time",
    "shopping",
}

def clean_text(text: str) -> str:
    """
    Remove Markdown formatting and unnecessary characters.
    """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"^[\-\*\•]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_valid_attraction(name: str) -> bool:
    """
    Check whether the extracted text looks like an attraction based on strict rules.
    """
    if not name:
        return False
        
    name_lower = name.lower().strip()
    
    # Reject short or too long text
    if len(name_lower) < 3 or len(name_lower) > 60:
        return False

    # Reject text containing forbidden words/currency/budget/cost related text
    forbidden_keywords = [
        "₹", "$", "€", "£", "lakhs", "lakh", "crore", "budget", "cost", "price", "total",
        "per day", "per night", "estimated", "flight", "hotel", "airport", "train", "metro",
        "bus", "transport", "breakfast", "lunch", "dinner", "check-in", "check-out",
        "arrival", "departure", "weather", "temperature", "humidity", "booking", "advice",
        "miscellaneous", "miscellanous", "free time"
    ]
    if any(keyword in name_lower for keyword in forbidden_keywords):
        return False

    # Reject text that starts with a number
    if re.match(r'^\d', name_lower):
        return False

    # Reject text containing currency amounts or numbers resembling budget/currency
    if re.search(r'\b\d+\s*(?:lakh|crore|k|usd|inr|eur|gbp)\b', name_lower):
        return False

    # Reject text containing percentage values
    if "%" in name_lower or "percent" in name_lower:
        return False

    # Reject text containing date or time values
    if name_lower.startswith("day ") or re.search(r'\b(?:day|mon|tue|wed|thu|fri|sat|sun)\b', name_lower):
        return False
    if re.search(r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b', name_lower):
        return False

    # Reject text that contains mostly numbers
    digits_count = sum(c.isdigit() for c in name_lower)
    if digits_count / len(name_lower) > 0.4:
        return False

    # Reject generic activities
    generic_activities = {"shopping", "sightseeing", "free time", "leisure", "relax", "relaxation", "rest"}
    if name_lower in generic_activities:
        return False

    # Reject long sentences or complete instruction sentences (usually > 6 words)
    words = name_lower.split()
    if len(words) > 6:
        return False
    if any(char in name_lower for char in ["?", "!"]):
        return False

    # Reject instruction sentences starting with action verbs
    action_verbs = {"visit", "explore", "see", "tour", "discover", "go", "spend", "enjoy", "walk", "check", "arrive", "depart", "book", "take", "catch", "ride", "fly", "travel"}
    if words and words[0] in action_verbs:
        return False

    return True

def extract_attractions_from_itinerary(
    itinerary: str,
    maximum: int = 10,
) -> list[str]:
    """
    Extract likely tourist attractions from the itinerary using label patterns.
    """
    if not itinerary:
        return []

    attractions: list[str] = []
    
    # Label pattern for sightseeing/attractions/visit
    labels_pattern = re.compile(
        r'(?:Sightseeing|Attractions|Places to visit|Visit|Explore|Morning|Afternoon|Evening)\s*:\s*(.*)', 
        re.IGNORECASE
    )
    
    # Action verbs pattern
    action_verbs_pattern = re.compile(
        r'\b(?:visit|explore|see|tour|discover|stop at|walk around)\b\s+(.*)',
        re.IGNORECASE
    )

    lines = itinerary.splitlines()

    for original_line in lines:
        line = clean_text(original_line)
        if not line:
            continue

        label_match = labels_pattern.search(line)
        extracted_parts = []
        
        if label_match:
            content = label_match.group(1).strip()
            # Split content by commas, semicolons, or "and"
            parts = re.split(r',|;|\band\b|\bfollowed by\b', content, flags=re.IGNORECASE)
            extracted_parts.extend(parts)
        else:
            verb_match = action_verbs_pattern.search(line)
            if verb_match:
                content = verb_match.group(1).strip()
                parts = re.split(r',|;|\band\b|\bfollowed by\b', content, flags=re.IGNORECASE)
                extracted_parts.extend(parts)

        for part in extracted_parts:
            part = clean_text(part)
            # Remove nested label prefixes ending with a colon
            part = re.sub(r'^(?:Sightseeing|Attractions|Places to visit|Visit|Explore|Morning|Afternoon|Evening|Day \d+)\s*:\s*', '', part, flags=re.IGNORECASE)
            # Remove leading articles and trailing text
            part = re.sub(r'^(?:the|a|an)\s+', '', part, flags=re.IGNORECASE)
            part = re.sub(r'^\s*(?:visit|explore|see|tour|discover|stop at|walk around)\s+', '', part, flags=re.IGNORECASE)
            part = re.split(r'\b(?:and try|and enjoy|where you can|to experience|for lunch|for dinner|using public|by train|by bus)\b', part, flags=re.IGNORECASE)[0]
            part = part.strip(" .:-*\"'")

            if is_valid_attraction(part):
                # Avoid duplicates
                if part.lower() not in {item.lower() for item in attractions}:
                    attractions.append(part)

            if len(attractions) >= maximum:
                return attractions

    return attractions

def extract_city_and_country(itinerary: str, user_query: str = "") -> tuple[str, str]:
    """
    Extract the destination city and country from the itinerary or user query using the LLM.
    """
    from mcp_client import llm
    from langchain_core.messages import SystemMessage, HumanMessage
    
    prompt = f"""
    Analyze the following travel user query and itinerary. Extract the primary destination city and country that the user is visiting.
    
    User Query: {user_query}
    Itinerary snippet: {itinerary[:1500]}
    
    Return ONLY a valid JSON object matching the format below. Do not include any markdown fences, explanation, or code blocks.
    {{
        "city": "Name of the primary city (e.g. Tokyo, Paris, Dubai)",
        "country": "Name of the country (e.g. Japan, France, UAE)"
    }}
    """
    try:
        response = llm.invoke([
            SystemMessage(content="You are a precise JSON extractor. Output only raw JSON."),
            HumanMessage(content=prompt)
        ])
        text = response.content.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            import json
            data = json.loads(json_match.group(0))
            return data.get("city", "").strip(), data.get("country", "").strip()
    except Exception as e:
        print(f"Error extracting city/country: {e}")
    
    return "", ""

def is_meaningful_match(attraction_name: str, title: str) -> bool:
    """
    Check if at least one meaningful word from the attraction name appears in the title.
    """
    title_lower = title.lower()
    if ":" in title_lower:
        title_lower = title_lower.split(":", 1)[1]
        
    words = re.findall(r'\b\w+\b', attraction_name.lower())
    
    stop_words = {
        "the", "of", "in", "on", "at", "for", "with", "a", "an", "and", "to", "by", "from",
        "temple", "shrine", "park", "castle", "lake", "river", "mountain", "street", "district",
        "palace", "tower", "bridge", "garden", "market", "monument", "museum", "road", "plaza",
        "square", "beach", "hotel", "airport", "station", "landmark", "tourism", "tourist"
    }
    
    meaningful_words = [w for w in words if len(w) >= 3 and w not in stop_words]
    
    # Exclude known generic locations if other, more specific words are present in the query
    known_locations = {
        "tokyo", "osaka", "kyoto", "nara", "hiroshima", "himeji", "hakone", "okinawa",
        "paris", "dubai", "london", "rome", "milan", "venice", "florence", "new york", "ny",
        "japan", "france", "uae", "italy", "uk", "usa", "spain", "india", "delhi", "mumbai"
    }
    
    meaningful_no_loc = [w for w in meaningful_words if w not in known_locations]
    if meaningful_no_loc:
        meaningful_words = meaningful_no_loc
    
    if not meaningful_words:
        meaningful_words = [w for w in words if w not in {"the", "of", "in", "and", "a", "an", "to"}]
        
    if not meaningful_words:
        return False
        
    return any(w in title_lower for w in meaningful_words)

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def search_wikimedia_image(
    attraction_name: str,
    city: str = "",
    country: str = "",
) -> Optional[dict]:
    """
    Search Wikimedia Commons for an attraction image with prioritized query matching.
    """
    search_queries = []
    
    # Priority 1: attraction name + city + country + landmark
    if city and country:
        search_queries.append(f"{attraction_name} {city} {country} landmark")
    # Priority 2: attraction name + city + tourism
    if city:
        search_queries.append(f"{attraction_name} {city} tourism")
    # Priority 3: attraction name + country
    if country:
        search_queries.append(f"{attraction_name} {country}")
        
    # Include raw phrase as fallback
    search_queries.append(attraction_name)
    
    # Deduplicate queries while maintaining order
    seen_queries = set()
    final_queries = []
    for q in search_queries:
        if q not in seen_queries:
            final_queries.append(q)
            seen_queries.add(q)
            
    headers = {
        "User-Agent": (
            "MultiAgentTravelPlanner/1.0 "
            "(educational Streamlit project)"
        )
    }
    
    for search_term in final_queries:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": search_term,
            "gsrnamespace": 6,
            "gsrlimit": 8,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 900,
            "origin": "*",
        }
        
        try:
            response = requests.get(
                WIKIMEDIA_API_URL,
                params=params,
                headers=headers,
                timeout=12,
            )
            response.raise_for_status()
            data = response.json()
            pages = (
                data.get("query", {})
                .get("pages", {})
                .values()
            )
            
            for page in pages:
                image_info_list = page.get("imageinfo", [])
                if not image_info_list:
                    continue
                    
                image_info = image_info_list[0]
                image_url = (
                    image_info.get("thumburl")
                    or image_info.get("url")
                )
                source_url = (
                    image_info.get("descriptionurl")
                    or image_info.get("descriptionshorturl")
                )
                if not image_url:
                    continue
                    
                title = page.get("title", "")
                title_lower = title.lower()
                
                # Reject non-photos: SVG, PDF, TIFF, icons, logos, diagrams, maps, screenshots, and documents
                # Only accept JPG, JPEG, PNG, and WebP
                if not title_lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                    
                # Ignore generic images
                generic_terms = [
                    "newspaper", "currency", "finance", "budget", "map icon", "logo",
                    "advertisement", "book cover", "document", "screenshot", "chart",
                    "graph", "poster", "icon", "diagram", "map"
                ]
                if any(term in title_lower for term in generic_terms):
                    continue
                    
                # Validate that at least one meaningful word matches
                if not is_meaningful_match(attraction_name, title):
                    continue
                    
                return {
                    "image_url": image_url,
                    "source_url": source_url,
                    "title": attraction_name,
                }
        except Exception:
            continue
            
    return None

def split_itinerary_by_day(itinerary: str) -> list[tuple[str, str]]:
    """
    Split the itinerary text into days.
    Returns a list of tuples: (day_heading, day_content)
    """
    if not itinerary:
        return []

    lines = itinerary.splitlines()
    days = []
    current_heading = "Introduction"
    current_lines = []

    # Pattern to find day boundaries (e.g. Day 1, **Day 2**, ### Day 3, etc.)
    day_pattern = re.compile(r'^(?:#+\s*|\*+\s*|-+\s*)?(Day\s+\d+[:\-\s\w]*)', re.IGNORECASE)

    for line in lines:
        match = day_pattern.match(line.strip())
        if match:
            # Save the accumulated content of the previous day
            if current_lines or current_heading != "Introduction":
                days.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Append the last day
    if current_lines or current_heading != "Introduction":
        days.append((current_heading, "\n".join(current_lines).strip()))

    return days

def render_itinerary_with_inline_images(
    itinerary: str,
    user_query: str = "",
) -> None:
    """
    Parse the itinerary day-by-day, render text line-by-line, extract specific attractions
    on a per-line basis, and display matching ATM-card-style images directly below that line.
    """
    if not itinerary:
        st.markdown("_No itinerary generated._")
        return

    # Extract destination city and country once to optimize lookup speed
    city, country = extract_city_and_country(itinerary, user_query)

    days = split_itinerary_by_day(itinerary)
    
    # If the itinerary couldn't be parsed day-by-day, render it as normal text
    if not days:
        st.markdown(itinerary)
        return

    for day_heading, day_content in days:
        # Render the heading cleanly
        if not day_heading.startswith("#") and not day_heading.startswith("**"):
            st.markdown(f"#### {day_heading}")
        else:
            st.markdown(day_heading)

        # Split the content of the day line-by-line
        lines = day_content.splitlines()
        for line in lines:
            # Render the text line with clickable target_blank links if present
            if "[" in line and "](" in line:
                st.markdown(convert_markdown_links_to_target_blank(line), unsafe_allow_html=True)
            else:
                st.markdown(line)
            
            # Detect attraction in this single line
            stripped_line = line.strip()
            if stripped_line:
                line_attractions = extract_attractions_from_itinerary(
                    itinerary=line,
                    maximum=1,
                )
                if line_attractions:
                    attraction = line_attractions[0]
                    img_data = search_wikimedia_image(
                        attraction_name=attraction,
                        city=city,
                        country=country
                    )
                    if img_data:
                        # Generate Google Maps search URL for attraction
                        maps_url = build_google_maps_link(img_data['title'], city=city)

                        # Render fixed small ATM-card design with Google Maps link
                        card_html = f"""
                        <div class="attraction-card">
                            <img src="{img_data['image_url']}" />
                            <div class="attraction-card-body">
                                <div class="attraction-name">{img_data['title']}</div>
                                <div style="display: flex; gap: 0.8rem; margin-top: 0.3rem;">
                                    <a href="{img_data['source_url']}" target="_blank" rel="noopener noreferrer">View source</a>
                                    <a href="{maps_url}" target="_blank" rel="noopener noreferrer" style="color: #4ea8f0;">📍 Google Maps</a>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
        # Add spacing between days
        st.markdown("")