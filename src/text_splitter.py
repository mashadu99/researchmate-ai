"""
text_splitter.py — מחלק מסמכים ארוכים לחתיכות חופפות עבור RAG.

למה לחלק לחתיכות?
    1. מגבלות טוקנים: ל-GPT יש חלון הקשר מקסימלי — PDF גדול לא יכנס.
    2. דיוק: חתיכות קטנות = אחזור ממוקד. ה-AI מקבל בדיוק הפסקה הרלוונטית.
    3. עלות: פחות טוקנים שנשלחים = עלות API נמוכה יותר.

למה חפיפה (overlap)?
    אם משפט חשוב נחתך בגבול בין שתי חתיכות, החפיפה מבטיחה
    שהוא יופיע מלא לפחות בחתיכה אחת.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    מחלק רשימת Documents לחתיכות קטנות וחופפות.

    RecursiveCharacterTextSplitter מנסה לפצל על פסקאות קודם,
    אחר כך משפטים, אחר כך מילים — מעדיף גבולות טבעיים.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source_name: str = "לא ידוע",
) -> list[Document]:
    """
    מחלק מחרוזת רגילה לחתיכות Document.
    שימושי כשיש לנו טקסט גולמי ולא רשימת Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    return [
        Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_index": i},
        )
        for i, chunk in enumerate(raw_chunks)
    ]


def get_chunk_stats(chunks: list[Document]) -> dict:
    """
    מחזיר סטטיסטיקות על החתיכות — שימושי לדיבאג ולממשק.
    """
    if not chunks:
        return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(chunk.page_content) for chunk in chunks]

    return {
        "count": len(chunks),
        "avg_length": int(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_characters": sum(lengths),
    }
