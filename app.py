"""
app.py — ResearchMate AI: אפליקציית Streamlit הראשית
עיצוב RTL מלא עם פונט Heebo מותאם לעברית.
"""

import streamlit as st

from src.pdf_loader import load_pdf_from_bytes, get_full_text
from src.text_splitter import split_documents, get_chunk_stats
from src.embeddings_store import build_vector_store, save_vector_store
from src.rag_pipeline import (
    answer_question,
    summarize_document,
    extract_insights,
    generate_linkedin_post,
    suggest_followup_questions,
)

st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@300;400;500;700;900&family=Rubik:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>

    /* ── פונט בסיסי לכל האפליקציה ── */
    html, body, [class*="css"] {
        font-family: 'Rubik', sans-serif !important;
    }

    /* ── RTL לתוכן הראשי בלבד ── */
    .main .block-container {
        direction: rtl !important;
        max-width: 860px !important;
        padding-top: 2rem !important;
    }

    /* ── RTL לסרגל הצד ── */
    [data-testid="stSidebar"] > div {
        direction: rtl !important;
    }
    [data-testid="stSidebar"] {
        background: #f7f8fa !important;
    }

    /* ── תיקון file uploader — הבאג של Uploadpload ── */
    [data-testid="stFileUploaderDropzone"] {
        direction: ltr !important;
        unicode-bidi: normal !important;
    }
    [data-testid="stFileUploaderDropzone"] button span {
        direction: ltr !important;
    }

    /* ── כותרת ── */
    .main-header {
        font-family: 'Frank Ruhl Libre', serif !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        text-align: center !important;
        letter-spacing: -1px !important;
        padding: 0.5rem 0 0.2rem !important;
        direction: ltr !important;
    }
    .sub-header {
        font-family: 'Rubik', sans-serif !important;
        font-size: 1rem !important;
        color: #6b7280 !important;
        text-align: center !important;
        font-weight: 300 !important;
        margin-bottom: 1.5rem !important;
        direction: rtl !important;
    }

    /* ── טאבים ── */
    .stTabs [data-baseweb="tab-list"] {
        flex-direction: row-reverse !important;
        gap: 2px !important;
        border-bottom: 2px solid #e5e7eb !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Rubik', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    /* ── כפתורים ── */
    .stButton > button {
        font-family: 'Rubik', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }

    /* ── שדות קלט ── */
    input, textarea {
        font-family: 'Rubik', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
        border-radius: 10px !important;
    }

    /* ── קופסת מקורות ── */
    .source-box {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-right: 4px solid #4f46e5 !important;
        border-radius: 8px !important;
        padding: 1rem 1.1rem !important;
        font-size: 0.85rem !important;
        line-height: 1.9 !important;
        direction: rtl !important;
    }

    /* ── תחתית ── */
    .footer-text {
        text-align: center !important;
        color: #9ca3af !important;
        font-size: 0.78rem !important;
        font-weight: 300 !important;
        letter-spacing: 0.5px !important;
    }

    /* ── metric ── */
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] div {
        text-align: right !important;
    }

    /* ── alerts ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }

</style>
""", unsafe_allow_html=True)


# ── אתחול Session State ──────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "documents": None,
        "chunks": None,
        "vector_store": None,
        "full_text": None,
        "summary": None,
        "insights": None,
        "linkedin_post": None,
        "qa_history": [],
        "filename": None,
        "processed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

model_choice = "llama-3.1-8b-instant"

# ── כותרת ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">ResearchMate AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">העלו מאמרי מחקר ופתחו תובנות מופעמות בינה מלאכותית</div>',
    unsafe_allow_html=True,
)
st.divider()


# ── סרגל צד ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ניהול מסמך")

    uploaded_file = st.file_uploader(
        "העלו קובץ PDF",
        type=["pdf"],
        help="מאמר מחקר, דוח, או כל מסמך PDF אחר.",
    )

    if uploaded_file is not None:
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.info(f"**קובץ:** {uploaded_file.name}\n\n**גודל:** {file_size_kb:.1f} KB")

        if st.button("עבד מסמך", type="primary", use_container_width=True):

            with st.spinner("קורא ומחלק את המסמך..."):
                try:
                    docs = load_pdf_from_bytes(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                    )
                    st.session_state.documents = docs
                    st.session_state.full_text = get_full_text(docs)
                    st.session_state.filename = uploaded_file.name
                    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
                    st.session_state.chunks = chunks
                except Exception as e:
                    st.error(f"שגיאה בקריאת ה-PDF: {e}")
                    st.stop()

            with st.spinner("בונה אינדקס וקטורי..."):
                try:
                    vector_store = build_vector_store(chunks)
                    save_vector_store(vector_store)
                    st.session_state.vector_store = vector_store
                    st.session_state.processed = True
                    st.session_state.summary = None
                    st.session_state.insights = None
                    st.session_state.linkedin_post = None
                    st.session_state.qa_history = []
                except Exception as e:
                    st.error(f"שגיאה בבניית אינדקס: {e}")
                    st.stop()

            st.success("המסמך עובד בהצלחה!")

    if st.session_state.processed and st.session_state.chunks:
        stats = get_chunk_stats(st.session_state.chunks)
        st.divider()
        st.subheader("נתוני מסמך")
        col1, col2 = st.columns(2)
        col1.metric("עמודים", len(st.session_state.documents))
        col2.metric("חתיכות", stats["count"])
        col1.metric("ממוצע", f"{stats['avg_length']} תווים")
        col2.metric("סה״כ", f"{stats['total_characters']:,}")

    st.divider()
    st.subheader("מודל AI")
    st.success("Groq — Llama 3.1 (חינמי)")
    st.caption("מופעם על ידי Groq.")


# ── תוכן ראשי ────────────────────────────────────────────────────────────────
if not st.session_state.processed:
    st.markdown("""
    ### ברוכים הבאים ל-ResearchMate AI

    **כיצד להתחיל:**
    1. העלו קובץ PDF דרך סרגל הצד
    2. לחצו על **עבד מסמך**
    3. השתמשו בטאבים למטה לחקור את המסמך
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**שאלות ותשובות**\nשאלו כל שאלה וקבלו תשובות מבוססות על המסמך בלבד.")
    with col2:
        st.info("**סיכום חכם**\nה-AI קורא את המאמר ומחלץ את הנקודות החשובות.")
    with col3:
        st.info("**פוסט LinkedIn**\nהפכו מחקר מורכב לתוכן מרתק לרשתות חברתיות.")

else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "שאלות ותשובות",
        "סיכום",
        "תובנות מפתח",
        "פוסט LinkedIn",
    ])

    # ── טאב 1: שאלות ותשובות ─────────────────────────────────────────────
    with tab1:
        st.subheader("שאלו שאלות על המסמך")
        st.caption("התשובות מבוססות אך ורק על תוכן המסמך שהעליתם.")

        question = st.text_input(
            "השאלה שלכם:",
            placeholder="מהם הממצאים העיקריים של המחקר?",
            key="question_input",
        )

        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_clicked = st.button("קבל תשובה", type="primary", use_container_width=True)
        with col_clear:
            if st.button("נקה היסטוריה", use_container_width=True):
                st.session_state.qa_history = []
                st.rerun()

        if ask_clicked and question.strip():
            with st.spinner("מחפש במסמך ויוצר תשובה..."):
                try:
                    result = answer_question(
                        question=question,
                        vector_store=st.session_state.vector_store,
                        k=4,
                        model=model_choice,
                    )
                    followups = suggest_followup_questions(
                        question=question,
                        answer=result["answer"],
                        model=model_choice,
                    )
                    st.session_state.qa_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["source_text"],
                        "followups": followups,
                    })
                except Exception as e:
                    st.error(f"שגיאה ביצירת תשובה: {e}")

        elif ask_clicked and not question.strip():
            st.warning("אנא הכניסו שאלה קודם.")

        if st.session_state.qa_history:
            for i, item in enumerate(reversed(st.session_state.qa_history)):
                with st.container():
                    st.markdown(f"**שאלה: {item['question']}**")
                    st.markdown(item["answer"])
                    with st.expander("הצג קטעי מקור מהמסמך"):
                        st.markdown(
                            f'<div class="source-box">{item["sources"]}</div>',
                            unsafe_allow_html=True,
                        )
                    if item.get("followups"):
                        with st.expander("שאלות המשך מוצעות"):
                            st.markdown(item["followups"])
                    if i < len(st.session_state.qa_history) - 1:
                        st.divider()

    # ── טאב 2: סיכום ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("סיכום מסמך")

        if st.session_state.summary is None:
            if st.button("צור סיכום", type="primary"):
                with st.spinner("קורא מסמך ויוצר סיכום..."):
                    try:
                        st.session_state.summary = summarize_document(
                            st.session_state.full_text, model=model_choice
                        )
                    except Exception as e:
                        st.error(f"שגיאה: {e}")
        else:
            if st.button("צור סיכום מחדש"):
                with st.spinner("מחדש סיכום..."):
                    try:
                        st.session_state.summary = summarize_document(
                            st.session_state.full_text, model=model_choice
                        )
                    except Exception as e:
                        st.error(f"שגיאה: {e}")

        if st.session_state.summary:
            st.markdown(st.session_state.summary)
            st.divider()
            st.download_button(
                "הורד סיכום",
                data=st.session_state.summary,
                file_name=f"summary_{st.session_state.filename}.txt",
                mime="text/plain",
            )

    # ── טאב 3: תובנות ────────────────────────────────────────────────────
    with tab3:
        st.subheader("תובנות וממצאים מפתח")

        if st.session_state.insights is None:
            if st.button("חלץ תובנות", type="primary"):
                with st.spinner("מחלץ תובנות מפתח..."):
                    try:
                        st.session_state.insights = extract_insights(
                            st.session_state.full_text, model=model_choice
                        )
                    except Exception as e:
                        st.error(f"שגיאה: {e}")
        else:
            if st.button("חלץ מחדש"):
                with st.spinner("מחדש חילוץ תובנות..."):
                    try:
                        st.session_state.insights = extract_insights(
                            st.session_state.full_text, model=model_choice
                        )
                    except Exception as e:
                        st.error(f"שגיאה: {e}")

        if st.session_state.insights:
            st.markdown(st.session_state.insights)
            st.divider()
            st.download_button(
                "הורד תובנות",
                data=st.session_state.insights,
                file_name=f"insights_{st.session_state.filename}.txt",
                mime="text/plain",
            )

    # ── טאב 4: LinkedIn ───────────────────────────────────────────────────
    with tab4:
        st.subheader("יצירת פוסט LinkedIn")
        st.caption("נדרשים סיכום ותובנות לפני יצירת הפוסט.")

        has_summary = st.session_state.summary is not None
        has_insights = st.session_state.insights is not None

        if not has_summary or not has_insights:
            st.warning("אנא צרו סיכום ותובנות קודם (בטאבים למעלה).")
            if not has_summary:
                st.markdown("- [ ] צור סיכום (טאב 2)")
            if not has_insights:
                st.markdown("- [ ] חלץ תובנות (טאב 3)")
        else:
            if st.session_state.linkedin_post is None:
                if st.button("צור פוסט LinkedIn", type="primary"):
                    with st.spinner("מנסח את הפוסט..."):
                        try:
                            st.session_state.linkedin_post = generate_linkedin_post(
                                summary=st.session_state.summary,
                                insights=st.session_state.insights,
                                model=model_choice,
                            )
                        except Exception as e:
                            st.error(f"שגיאה: {e}")
            else:
                if st.button("צור פוסט מחדש"):
                    with st.spinner("מחדש..."):
                        try:
                            st.session_state.linkedin_post = generate_linkedin_post(
                                summary=st.session_state.summary,
                                insights=st.session_state.insights,
                                model=model_choice,
                            )
                        except Exception as e:
                            st.error(f"שגיאה: {e}")

            if st.session_state.linkedin_post:
                st.text_area(
                    "הפוסט שלכם:",
                    value=st.session_state.linkedin_post,
                    height=300,
                    key="linkedin_textarea",
                )
                st.download_button(
                    "הורד פוסט",
                    data=st.session_state.linkedin_post,
                    file_name=f"linkedin_{st.session_state.filename}.txt",
                    mime="text/plain",
                )


# ── תחתית ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div class="footer-text">ResearchMate AI · מופעם על ידי Groq + LangChain + FAISS</div>',
    unsafe_allow_html=True,
)
