# ResearchMate AI 🔬

עוזר מחקר מופעם בינה מלאכותית — העלו מסמכי PDF וקבלו תובנות מיידיות.

## תכונות
- סיכום מסמך מובנה
- שאלות ותשובות מבוססות RAG (ללא הזיות)
- חילוץ תובנות מפתח
- יצירת פוסט LinkedIn

## טכנולוגיות
- **Streamlit** — ממשק משתמש
- **OpenAI** — GPT-3.5/GPT-4 + Embeddings
- **LangChain** — תזמור AI
- **FAISS** — חיפוש וקטורי

## התקנה

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

צרו קובץ `.env`:
```
OPENAI_API_KEY=המפתח_שלכם_כאן
```

## הרצה

```bash
streamlit run app.py
```

פתחו דפדפן בכתובת: `http://localhost:8501`

## ארכיטקטורה

```
העלאת PDF → חילוץ טקסט → חלוקה לחתיכות → Embeddings → אינדקס FAISS
                                                              ↓
שאלה → Embedding → חיפוש דמיון → הקשר → GPT → תשובה
```
