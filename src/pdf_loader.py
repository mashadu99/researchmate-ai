"""
pdf_loader.py — מטפל בקריאת קבצי PDF וחילוץ הטקסט שלהם.

PDF הוא פורמט בינארי — לא ניתן לפתוח אותו כמו קובץ טקסט רגיל.
המודול הזה משתמש ב-pypdf לחלץ טקסט עמוד-עמוד ולעטוף אותו
באובייקטי Document של LangChain שנושאים metadata (שם קובץ, מספר עמוד).
"""

import os
import io
from pypdf import PdfReader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """
    קורא קובץ PDF מנתיב בדיסק ומחזיר רשימת Documents.

    מה זה Document של LangChain?
        מיכל עם page_content (הטקסט) ו-metadata (מאיפה הגיע).
        משתמשים בו במקום מחרוזת רגילה כדי לשמור על מקור הטקסט.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF לא נמצא בנתיב: {file_path}")
    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"הקובץ חייב להיות PDF. קיבלתי: {file_path}")

    reader = PdfReader(file_path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue

        cleaned_text = " ".join(text.split())
        doc = Document(
            page_content=cleaned_text,
            metadata={
                "source": os.path.basename(file_path),
                "page": page_num + 1,
                "total_pages": len(reader.pages),
            },
        )
        documents.append(doc)

    if not documents:
        raise ValueError(
            "לא נמצא טקסט קריא ב-PDF הזה. "
            "ייתכן שמדובר במסמך סרוק (PDF של תמונות בלבד)."
        )
    return documents


def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> list[Document]:
    """
    קורא PDF מבייטים גולמיים — משמש כש-Streamlit מעלה קובץ לזיכרון.
    Streamlit נותן לנו bytes, לא נתיב לקובץ, ולכן צריך את הגרסה הזו.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue

        cleaned_text = " ".join(text.split())
        doc = Document(
            page_content=cleaned_text,
            metadata={
                "source": filename,
                "page": page_num + 1,
                "total_pages": len(reader.pages),
            },
        )
        documents.append(doc)

    if not documents:
        raise ValueError(
            "לא נמצא טקסט קריא ב-PDF הזה. "
            "ייתכן שמדובר במסמך סרוק (PDF של תמונות בלבד)."
        )
    return documents


def get_full_text(documents: list[Document]) -> str:
    """מאחד את כל עמודי המסמך למחרוזת אחת גדולה."""
    return "\n\n".join(doc.page_content for doc in documents)
