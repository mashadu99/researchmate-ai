"""
prompts.py — כל תבניות ה-Prompt של ResearchMate AI.

למה למרכז prompts?
    Prompts הם ה"תכנות" של LLM — שינוי prompt = שינוי התנהגות ה-AI.
    קובץ אחד = קל לכוון, ללא לוגיקת prompt כפולה.

PromptTemplate = מחרוזת עם {placeholders} שמתמלאים בזמן ריצה.
"""

from langchain_core.prompts import PromptTemplate


# ── 1. שאלות ותשובות (RAG) ───────────────────────────────────────────────────
# עונה רק מתוך ההקשר שחולץ מהמסמך — מונע הזיות של ה-AI.

RAG_QA_TEMPLATE = """You are an expert research assistant helping a user understand a document.

Use ONLY the following context extracted from the document to answer the question.
If the answer cannot be found in the context, say: "לא מצאתי מידע על כך במסמך."
Do not make up information or use knowledge outside the provided context.

Context from document:
{context}

Question: {question}

Provide a clear, accurate, and well-structured answer based solely on the context above.
Answer in the same language the question was asked in.
If relevant, mention which part of the document your answer comes from.

Answer:"""

RAG_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_QA_TEMPLATE,
)


# ── 2. סיכום מסמך ────────────────────────────────────────────────────────────
# יוצר סיכום מובנה עם כותרות קבועות לעקביות.

SUMMARIZATION_TEMPLATE = """You are a research assistant specializing in summarizing academic papers and technical documents.

Read the following document text carefully and provide a comprehensive summary in Hebrew.

Document text:
{text}

Structure your summary with these Hebrew headings:

**סקירה כללית**
(2-3 משפטים על מה המסמך עוסק)

**המטרה המרכזית**
(איזו בעיה או שאלה העבודה מטפלת?)

**שיטות וגישה**
(כיצד גישו המחברים לבעיה?)

**ממצאים עיקריים**
(מה היו התגליות או המסקנות המרכזיות?)

**משמעות ותרומה**
(למה זה חשוב? מה ההשפעה?)

ענה בעברית. שמור על כל סעיף תמציתי אך אינפורמטיבי."""

SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=SUMMARIZATION_TEMPLATE,
)


# ── 3. חילוץ תובנות מפתח ─────────────────────────────────────────────────────

INSIGHTS_TEMPLATE = """You are an expert at extracting actionable insights from research documents.

Based on the following document content, extract the most important insights and findings.

Document content:
{text}

Return exactly 5-8 key insights as a numbered list in Hebrew. Each insight should:
- Be a complete, self-contained statement in Hebrew
- Be specific with numbers, percentages, or facts when available
- Be understandable without reading the full document

תובנות מפתח:"""

INSIGHTS_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=INSIGHTS_TEMPLATE,
)


# ── 4. פוסט LinkedIn ─────────────────────────────────────────────────────────

LINKEDIN_POST_TEMPLATE = """You are a professional content writer who makes research accessible on LinkedIn.

Based on the following research document summary and insights, write a compelling LinkedIn post in Hebrew.

Document summary:
{summary}

Key insights:
{insights}

Write a LinkedIn post in Hebrew that:
- Starts with a compelling hook (question or bold statement)
- Explains the research and why it matters in simple language
- Highlights 3 key findings as bullet points using emojis
- Ends with a thought-provoking question to drive engagement
- Is between 150-250 words
- Includes 5-7 relevant Hebrew hashtags at the end

פוסט LinkedIn:"""

LINKEDIN_POST_PROMPT = PromptTemplate(
    input_variables=["summary", "insights"],
    template=LINKEDIN_POST_TEMPLATE,
)


# ── 5. שאלות המשך ────────────────────────────────────────────────────────────

FOLLOWUP_QUESTIONS_TEMPLATE = """Based on this answer about a research document, suggest 3 natural follow-up questions.

Answer that was just given:
{answer}

Original question:
{question}

Generate 3 follow-up questions in the same language as the original question.
Each question should dig deeper or explore a related aspect not covered in the answer.

Format as a numbered list:
1.
2.
3."""

FOLLOWUP_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["answer", "question"],
    template=FOLLOWUP_QUESTIONS_TEMPLATE,
)
