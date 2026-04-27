import time

from observability import observe  # version-safe, no-op fallback included
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tools import ask_for_clarification
from config import MAX_RETRIES, RETRY_BACKOFF_SECONDS

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


@observe(name="clarification_decision")
def decide_next_step(question: str):
    system_prompt = (
        "You are the gateway to a RAG (Retrieval-Augmented Generation) system that answers questions about a specific PDF document.\n"
        "Decide whether the question is clear enough to attempt a search in the document.\n"
        "\n"
        "DO NOT ask for clarification for:\n"
        "1. General questions about the document (e.g., 'What is this about?', 'Summarize this', 'What's in the knowledge base?').\n"
        "2. Questions about your capabilities (e.g., 'What can you do?', 'What files can you read?').\n"
        "3. Clear, direct questions, even if they are broad.\n"
        "\n"
        "ONLY call `ask_for_clarification` if the question is truly ambiguous, nonsensical, or requires missing specific details that the document couldn't possibly provide (e.g., 'What did he say?' where 'he' is undefined).\n"
        "\n"
        "If the question is ready for search, respond with NO_CLARIFICATION_NEEDED."
    )

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=question),
                ],
                tools=[ask_for_clarification],
            )
            return response
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise RuntimeError(f"Clarification decision failed after retries: {last_error}")
