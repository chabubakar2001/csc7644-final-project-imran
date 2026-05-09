"""
Text chunking utilities for the RAG research paper assistant.
"""


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.

    Args:
        text: Full paper text.
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        List of chunk dictionaries with IDs and text.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    words = text.split()
    chunks = []
    start = 0
    chunk_id = 1

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_value = " ".join(chunk_words)

        if chunk_text_value.strip():
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text_value,
                }
            )
            chunk_id += 1

        start += chunk_size - overlap

    return chunks