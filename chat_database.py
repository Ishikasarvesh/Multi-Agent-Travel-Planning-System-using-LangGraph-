# ── Chat history database management ──

import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def create_chat_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS travel_chats (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS travel_chat_messages (
                    id SERIAL PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES travel_chats(chat_id) ON DELETE CASCADE
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS travel_chat_state (
                    chat_id TEXT PRIMARY KEY,
                    current_plan JSONB DEFAULT '{}'::jsonb,
                    travel_results JSONB DEFAULT '{}'::jsonb,
                    generated_itinerary TEXT DEFAULT '',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES travel_chats(chat_id) ON DELETE CASCADE
                );
            """)
            conn.commit()

def create_chat(chat_id: str, title: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO travel_chats (chat_id, title) VALUES (%s, %s) ON CONFLICT (chat_id) DO NOTHING",
                (chat_id, title)
            )
            conn.commit()

def list_chats():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chat_id, title, created_at, updated_at FROM travel_chats ORDER BY updated_at DESC")
            return cur.fetchall()

def rename_chat(chat_id: str, new_title: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE travel_chats SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE chat_id = %s",
                (new_title, chat_id)
            )
            conn.commit()

def delete_chat(chat_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM travel_chats WHERE chat_id = %s", (chat_id,))
            conn.commit()

def save_message(chat_id: str, role: str, content: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO travel_chat_messages (chat_id, role, content) VALUES (%s, %s, %s)",
                (chat_id, role, content)
            )
            cur.execute(
                "UPDATE travel_chats SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = %s",
                (chat_id,)
            )
            conn.commit()

def load_messages(chat_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, created_at FROM travel_chat_messages WHERE chat_id = %s ORDER BY id ASC",
                (chat_id,)
            )
            return cur.fetchall()

def delete_last_message(chat_id: str):
    """Deletes the last message in a chat (typically the assistant's response)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM travel_chat_messages WHERE id = (SELECT id FROM travel_chat_messages WHERE chat_id = %s ORDER BY id DESC LIMIT 1)",
                (chat_id,)
            )
            conn.commit()

def update_last_user_message(chat_id: str, new_content: str):
    """Updates the content of the last user message in a chat."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE travel_chat_messages SET content = %s WHERE id = (SELECT id FROM travel_chat_messages WHERE chat_id = %s AND role = 'user' ORDER BY id DESC LIMIT 1)",
                (new_content, chat_id)
            )
            conn.commit()

def save_chat_state(chat_id: str, current_plan: dict, travel_results: dict, generated_itinerary: str):
    import json
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO travel_chat_state (chat_id, current_plan, travel_results, generated_itinerary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chat_id) DO UPDATE
                SET current_plan = EXCLUDED.current_plan,
                    travel_results = EXCLUDED.travel_results,
                    generated_itinerary = EXCLUDED.generated_itinerary,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                chat_id,
                json.dumps(current_plan),
                json.dumps(travel_results),
                generated_itinerary
            ))
            conn.commit()

def load_chat_state(chat_id: str):
    import json
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_plan, travel_results, generated_itinerary FROM travel_chat_state WHERE chat_id = %s", (chat_id,))
            row = cur.fetchone()
            if row:
                try:
                    c_plan = row["current_plan"] if isinstance(row["current_plan"], dict) else json.loads(row["current_plan"])
                except Exception:
                    c_plan = {}
                try:
                    t_res = row["travel_results"] if isinstance(row["travel_results"], dict) else json.loads(row["travel_results"])
                except Exception:
                    t_res = {}
                return {
                    "current_plan": c_plan,
                    "travel_results": t_res,
                    "generated_itinerary": row["generated_itinerary"] or ""
                }
            return {
                "current_plan": {},
                "travel_results": {},
                "generated_itinerary": ""
            }
