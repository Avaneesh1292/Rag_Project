import time
from typing import Callable, Optional

from observability import observe  # version-safe, no-op fallback included
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from config import MAX_RETRIES, RETRY_BACKOFF_SECONDS

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    streaming=True,  # enables token streaming
)

# {history} is injected at runtime; empty string when no prior turns exist.
prompt = ChatPromptTemplate.from_template("""
You are a careful question-answering assistant.

You must answer using ONLY the retrieved context from the PDF.
Treat the retrieved context as the only source of truth.

Rules:
1) Do not use outside knowledge.
2) Do not guess, infer unsupported facts, or fill in missing details.
3) If the retrieved context is empty, weak, or not clearly relevant to the question, reply exactly: "I don't know."
4) If the answer is partially supported, provide only the supported part and then say "I don't know." for the rest.
5) Keep the answer concise and factual.
6) Use prior conversation context to resolve pronouns or follow-up references.
{history}
Context:
{context}

Question:
{question}

Answer:
""")


def _format_history(chat_history: list) -> str:
    """Format list of (question, answer) tuples into a prompt-friendly string."""
    if not chat_history:
        return ""
    lines = ["\nPrevious conversation:"]
    for q, a in chat_history:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines) + "\n"


@observe(name="rag_run")
def run_rag(
    question: str,
    retriever,
    chat_history: Optional[list] = None,
    on_token: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run the full RAG pipeline.

    Args:
        question:     The user's question.
        retriever:    HybridRetriever instance.
        chat_history: List of (question, answer) tuples for multi-turn memory.
                      Pass None or [] for no history.
        on_token:     Optional callback called with each streamed token string.
                      Use to print tokens as they arrive.

    Returns:
        dict with keys: "answer" (str), "docs" (list), "sources" (list[str]).
    """
    if chat_history is None:
        chat_history = []

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            docs = retriever.retrieve(question)
            reranked = retriever.rerank(question, docs)

            context = "\n\n".join(d.page_content for d in reranked)
            history_text = _format_history(chat_history)

            chain = (
                {
                    "context": lambda _: context,
                    "question": RunnablePassthrough(),
                    "history": lambda _: history_text,
                }
                | prompt
                | llm
            )

            # Stream tokens and collect full answer
            full_answer = ""
            for chunk in chain.stream(question):
                token = chunk.content
                full_answer += token
                if on_token and token:
                    on_token(token)

            sources = []
            for i, d in enumerate(reranked, start=1):
                page = d.metadata.get("page", "?")
                src = d.metadata.get("source", "unknown")
                sources.append(f"{i}. {src} (page {page})")

            return {
                "answer": full_answer,
                "docs": reranked,
                "sources": sources,
            }
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise RuntimeError(f"RAG pipeline failed after retries: {last_error}")
