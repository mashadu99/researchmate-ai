"""
utils.py — פונקציות עזר משותפות לכל הפרויקט.
"""

import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()


def get_groq_api_key() -> str:
    """
    קורא את מפתח ה-API של Groq.
    מנסה קודם Streamlit Secrets (לענן), אחר כך .env (מקומי).
    """
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "לא נמצא GROQ_API_KEY. "
            "ודאי שיש לך קובץ .env עם המפתח מוגדר בו."
        )
    return key


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """סופר כמה טוקנים מחרוזת משתמשת."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def truncate_text(text: str, max_tokens: int = 3000, model: str = "gpt-3.5-turbo") -> str:
    """מקצר טקסט כדי להישאר בתוך מגבלת טוקנים."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def format_sources(source_chunks: list) -> str:
    """מעצב חתיכות מסמך שאוחזרו לבלוק קריא."""
    if not source_chunks:
        return "לא נמצאו קטעי מקור."

    formatted = []
    for i, chunk in enumerate(source_chunks, 1):
        content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        page = chunk.metadata.get("page", "?") if hasattr(chunk, "metadata") else "?"
        formatted.append(f"[מקור {i} — עמוד {page}]\n{content.strip()}")

    return "\n\n---\n\n".join(formatted)
