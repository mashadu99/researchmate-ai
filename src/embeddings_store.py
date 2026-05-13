"""
embeddings_store.py — אחסון וקטורים עם HuggingFace (חינמי, רץ מקומית).

במקום OpenAI Embeddings, משתמשים ב-sentence-transformers שרצים
ישירות על המחשב — ללא API וללא עלות.
המודל all-MiniLM-L6-v2 קטן, מהיר, ומדויק מאוד.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

INDEX_SAVE_PATH = ".faiss_index"


def get_embeddings():
    """יוצר מודל embeddings מקומי של HuggingFace — ללא API."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: list[Document]) -> FAISS:
    """בונה אינדקס FAISS מחתיכות המסמך עם embeddings מקומיים."""
    if not chunks:
        raise ValueError("לא ניתן לבנות מאגר וקטורים מרשימת חתיכות ריקה.")

    embeddings = get_embeddings()
    return FAISS.from_documents(documents=chunks, embedding=embeddings)


def save_vector_store(vector_store: FAISS, path: str = INDEX_SAVE_PATH) -> None:
    """שומר את אינדקס FAISS לדיסק."""
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)


def load_vector_store(path: str = INDEX_SAVE_PATH) -> FAISS:
    """טוען אינדקס FAISS שמור מהדיסק."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"לא נמצא אינדקס FAISS שמור ב-'{path}'. העלי ועבדי מסמך קודם."
        )
    return FAISS.load_local(
        path,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def similarity_search(vector_store: FAISS, query: str, k: int = 4) -> list[Document]:
    """מוצא את k החתיכות הכי דומות סמנטית לשאילתה."""
    return vector_store.similarity_search(query, k=k)


def vector_store_exists(path: str = INDEX_SAVE_PATH) -> bool:
    """בודק אם אינדקס FAISS שמור קיים בדיסק."""
    return os.path.exists(path) and os.path.exists(os.path.join(path, "index.faiss"))
