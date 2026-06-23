from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


def split_text_into_chunks(
    text: str,
    chunk_size: int = 50,
    chunk_overlap: int = 10
) -> dict:
    """
    Split a given text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text: The input text to split.
        chunk_size: Maximum size of each chunk (default: 50).
        chunk_overlap: Number of overlapping characters between chunks (default: 10).

    Returns:
        A dict containing the chunks list and total count.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks: List[str] = splitter.split_text(text)

    return {
        "chunks": chunks,
        "total_chunks": len(chunks),
        "chunk_size_used": chunk_size,
        "chunk_overlap_used": chunk_overlap,
    }
