# src/chunker.py
from typing import List

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks