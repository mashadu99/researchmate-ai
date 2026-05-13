"""
rag_pipeline.py — ה-pipeline המרכזי של ה-AI עם Groq.

Groq הוא שירות AI חינמי ומהיר מאוד שמריץ מודלים כמו Llama 3.
משתמשים ב-LCEL של LangChain: chain = prompt | llm
"""

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from src.utils import get_groq_api_key, truncate_text, format_sources
from src.prompts import (
    RAG_QA_PROMPT,
    SUMMARIZATION_PROMPT,
    INSIGHTS_PROMPT,
    LINKEDIN_POST_PROMPT,
    FOLLOWUP_QUESTIONS_PROMPT,
)
from src.embeddings_store import similarity_search

# המודל החינמי של Groq — מהיר ומדויק
GROQ_MODEL = "llama-3.1-8b-instant"


def get_llm(temperature: float = 0.3) -> ChatGroq:
    """יוצר מופע ChatGroq."""
    return ChatGroq(
        api_key=get_groq_api_key(),
        model_name=GROQ_MODEL,
        temperature=temperature,
    )


def _run_chain(prompt, llm, inputs: dict) -> str:
    """מריץ prompt | llm ומחזיר טקסט."""
    chain = prompt | llm
    result = chain.invoke(inputs)
    return result.content


def answer_question(
    question: str,
    vector_store: FAISS,
    k: int = 4,
    model: str = GROQ_MODEL,
) -> dict:
    """פונקציית ה-RAG המרכזית: אחזר חתיכות רלוונטיות ואז צור תשובה."""
    relevant_chunks = similarity_search(vector_store, question, k=k)

    if not relevant_chunks:
        return {
            "answer": "לא נמצא תוכן רלוונטי במסמך לשאלתך.",
            "sources": [],
            "source_text": "",
        }

    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)
    llm = get_llm(temperature=0.2)
    answer = _run_chain(RAG_QA_PROMPT, llm, {"context": context, "question": question})

    return {
        "answer": answer.strip(),
        "sources": relevant_chunks,
        "source_text": format_sources(relevant_chunks),
    }


def summarize_document(full_text: str, model: str = GROQ_MODEL, max_tokens: int = 3000) -> str:
    """יוצר סיכום מובנה של המסמך."""
    truncated = truncate_text(full_text, max_tokens=max_tokens)
    llm = get_llm(temperature=0.3)
    return _run_chain(SUMMARIZATION_PROMPT, llm, {"text": truncated}).strip()


def extract_insights(full_text: str, model: str = GROQ_MODEL, max_tokens: int = 3000) -> str:
    """מחלץ 5-8 תובנות מפתח מהמסמך."""
    truncated = truncate_text(full_text, max_tokens=max_tokens)
    llm = get_llm(temperature=0.3)
    return _run_chain(INSIGHTS_PROMPT, llm, {"text": truncated}).strip()


def generate_linkedin_post(summary: str, insights: str, model: str = GROQ_MODEL) -> str:
    """יוצר פוסט LinkedIn מהסיכום והתובנות."""
    llm = get_llm(temperature=0.7)
    return _run_chain(LINKEDIN_POST_PROMPT, llm, {"summary": summary, "insights": insights}).strip()


def suggest_followup_questions(question: str, answer: str, model: str = GROQ_MODEL) -> str:
    """מציע 3 שאלות המשך לחקירת המסמך לעומק."""
    llm = get_llm(temperature=0.5)
    return _run_chain(FOLLOWUP_QUESTIONS_PROMPT, llm, {"question": question, "answer": answer}).strip()
