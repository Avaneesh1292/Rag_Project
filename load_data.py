import warnings
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfStreamError
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def load_and_chunk_pdf(embeddings=None):
    """
    Load the PDF from PDF_PATH and split it into chunks.

    If `embeddings` is provided, uses SemanticChunker for meaning-aware splits.
    Falls back to RecursiveCharacterTextSplitter if langchain_experimental is
    not installed or semantic chunking fails.
    """
    if not PDF_PATH:
        raise ValueError("PDF_PATH is empty. Set PDF_PATH in your .env or shell environment.")

    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    with pdf_file.open("rb") as f:
        header = f.read(5)

    if not header.startswith(b"%PDF"):
        if header.startswith(b"PK"):
            raise ValueError(
                "The file at PDF_PATH is not a valid PDF (it starts with 'PK', "
                "typically a ZIP/DOCX/XLSX file). Point PDF_PATH to a real .pdf file."
            )
        raise ValueError(
            "The file at PDF_PATH is not a valid PDF (missing '%PDF' header). "
            "Point PDF_PATH to a real .pdf file."
        )

    loader = PyPDFLoader(PDF_PATH)
    try:
        docs = loader.load()
    except PdfStreamError as e:
        raise ValueError(
            "Failed to read PDF. The file may be corrupted, incomplete, or not actually a PDF. "
            "Please verify PDF_PATH points to a valid PDF."
        ) from e

    # Feature 5 — Semantic Chunking
    if embeddings is not None:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            splitter = SemanticChunker(embeddings)
            chunks = splitter.split_documents(docs)
            print(f"  ✓ Semantic chunking: {len(chunks)} chunks")
            return chunks
        except ImportError:
            warnings.warn(
                "langchain-experimental not installed — falling back to RecursiveCharacterTextSplitter. "
                "Run: pip install langchain-experimental"
            )
        except Exception as e:
            warnings.warn(f"SemanticChunker failed ({e}) — falling back to RecursiveCharacterTextSplitter.")

    # Fallback — character-based chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"  ✓ Character chunking: {len(chunks)} chunks")
    return chunks
