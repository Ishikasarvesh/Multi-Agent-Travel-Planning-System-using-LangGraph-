# ── Chat history helpers and title generation ──

import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(override=True)

# Initialize a ChatGroq LLM for title generation
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

def generate_chat_title(first_message: str) -> str:
    """Generates a concise 3 to 6 word title for a travel planning chat."""
    prompt = f"""
    Create a short, concise, and catchy title for a travel planning chat based on this first user message:
    "{first_message}"
    
    The title must be between 3 to 6 words. Do not include quotes, markdown formatting, or any extra words in the output. Just return the raw title.
    """
    try:
        response = llm.invoke(prompt)
        title = response.content.strip().strip('"').strip("'").strip()
        # Clean any trailing punctuation
        title = re.sub(r'[.\-_?!\'"“”‘’]+$', '', title)
        # Check word count
        words = title.split()
        if 2 <= len(words) <= 8:
            return title
        else:
            return " ".join(first_message.split()[:4]) + " Plan"
    except Exception as e:
        print(f"Error generating chat title: {e}")
        # Fallback: take first few words
        return " ".join(first_message.split()[:4]) + " Plan"

def parse_assistant_message(content: str) -> dict:
    """Decodes structured JSON content for assistant messages or treats it as plain text."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and ("itinerary" in data or "flight_results" in data or "hotel_results" in data):
            return data
    except Exception:
        pass
    return {
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "itinerary": "",
        "final_response": content
    }
