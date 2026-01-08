# src/retriever.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class VectorRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализирует ретривер с локальной моделью эмбеддингов.
        :param model_name: имя модели из Hugging Face (например, 'all-MiniLM-L6-v2')
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.eval()  # отключаем dropout, если используется
        self.index = None
        self.chunks = []

    def embed(self, texts: List[str]) -> np.ndarray:
        """Генерирует эмбеддинги для списка текстов."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def build_index(self, chunks: List[str]):
        """Строит FAISS-индекс по чанкам."""
        self.chunks = chunks
        embeddings = self.embed(chunks)
        dimension = embeddings.shape[1]
        
        # Используем IndexFlatIP (косинусное сходство), т.к. эмбеддинги нормализованы
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 3) -> List[str]:
        """Ищет k наиболее релевантных чанков по запросу."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.embed([query])
        scores, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]