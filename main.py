import sys

import config
from observability import observe  # noqa: F401 — initialises Langfuse client

from load_data import load_and_chunk_pdf
from embeddings_store import build_embeddings, build_vector_store
from retriever import HybridRetriever
from decision_model import decide_next_step
from rag_pipeline import run_rag


# ── Terminal colour helpers ──────────────────────────────────────────

def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def bold(text):
    return f"\033[1m{text}\033[0m"

def cyan(text):
    return color(text, 36)

def yellow(text):
    return color(text, 33)

def green(text):
    return color(text, 32)

def red(text):
    return color(text, 31)

def dim(text):
    return f"\033[2m{text}\033[0m"


# ── Initialisation ───────────────────────────────────────────────────

config.validate_config()

print(bold(cyan("\n⟳  Loading embedding model...")))
embeddings = build_embeddings()

print(bold(cyan("⟳  Loading and chunking PDF (semantic)...")))
chunks = load_and_chunk_pdf(embeddings)  # passes embeddings for SemanticChunker

print(bold(cyan("⟳  Building vector store...")))
store = build_vector_store(chunks, embeddings)

print(bold(cyan("⟳  Initialising retriever & cross-encoder...")))
retriever = HybridRetriever(store, embeddings, chunks)

print(bold(green("\n==============================")))
print(bold(green("   RAG PDF QA Terminal Ready!   ")))
print(bold(green("==============================")))
print(dim("  Type 'exit' to quit | 'history' to view conversation\n"))


# ── Conversation memory ──────────────────────────────────────────────
# Stores (question, answer) tuples — capped at last 6 turns
MAX_HISTORY_TURNS = 6
chat_history: list = []


# ── Main loop ────────────────────────────────────────────────────────

while True:
    print(bold(cyan("────────────────────────────────────────────")))
    user_q = input(bold(yellow("Ask a question (or type 'exit'): ")))
    stripped = user_q.strip()

    if not stripped:
        continue

    if stripped.lower() == "exit":
        print(bold(green("\nGoodbye!")))
        sys.exit(0)

    # Special command: show conversation history
    if stripped.lower() == "history":
        if not chat_history:
            print(dim("  (No conversation history yet)\n"))
        else:
            print(bold(cyan("\n── Conversation History ──")))
            for i, (q, a) in enumerate(chat_history, 1):
                print(bold(f"  [{i}] Q: {q}"))
                print(dim(f"       A: {a[:120]}{'...' if len(a) > 120 else ''}"))
            print()
        continue

    # Step 1 — Clarification check
    try:
        decision = decide_next_step(user_q)
    except Exception as e:
        print(red(f"\nUnable to run clarification decision: {e}"))
        continue

    if decision.tool_calls:
        tool_call = decision.tool_calls[0]
        reason = tool_call.get("args", {}).get("reason", "Need more context.")
        print(bold(yellow("\nClarification needed:")))
        print(reason)
        extra = input(bold(yellow("Additional context: ")))
        user_q = user_q + " " + extra

    # Step 2 — RAG answer (streaming)
    print(bold(green("\nANSWER:")))
    try:
        output = run_rag(
            user_q,
            retriever,
            chat_history=chat_history,
            on_token=lambda t: print(t, end="", flush=True),
        )
        print("\n")  # newline after streamed answer
    except Exception as e:
        print(red(f"\nUnable to generate answer: {e}"))
        continue

    answer_text = output["answer"].strip()
    normalized = answer_text.lower().replace(".", "").strip()

    # Step 3 — Handle "I don't know" with optional follow-up
    if normalized in ("i don't know", "i dont know"):
        print(bold(red("I couldn't find enough relevant context in the PDF.")))
        extra = input(bold(yellow("Add more context (or press Enter to skip): "))).strip()
        if extra:
            user_q = f"{user_q} {extra}"
            print(bold(green("\nANSWER:")))
            try:
                output = run_rag(
                    user_q,
                    retriever,
                    chat_history=chat_history,
                    on_token=lambda t: print(t, end="", flush=True),
                )
                print("\n")
                answer_text = output["answer"].strip()
            except Exception as e:
                print(red(f"\nUnable to generate answer after clarification: {e}"))
                continue

    # Step 4 — Display sources
    if output["sources"]:
        print(bold(cyan("Sources:")))
        for src in output["sources"]:
            print(cyan(f"  - {src}"))
        print()

    # Update conversation history (cap at MAX_HISTORY_TURNS)
    chat_history.append((user_q, answer_text))
    if len(chat_history) > MAX_HISTORY_TURNS:
        chat_history = chat_history[-MAX_HISTORY_TURNS:]
