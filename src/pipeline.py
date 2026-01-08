# src/pipeline.py
from src.document_loader import load_document
from src.chunker import chunk_text
from src.retriever import VectorRetriever  # ← локальный, без OpenAI
from src.generator import Generator        # ← пока с OpenAI (для генерации)
from typing import Optional

class RAGPipeline:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        :param openai_api_key: нужен ТОЛЬКО для генерации ответов (Generator)
        :param embedding_model: модель для локальных эмбеддингов
        """
        self.retriever = VectorRetriever(model_name=embedding_model)
        self.generator = Generator(openai_api_key) if openai_api_key else None

    def ingest(self, file_path: str, chunk_size: int = 512, overlap: int = 64):
        print(f"Loading document: {file_path}")
        text = load_document(file_path)
        print(f"Total characters: {len(text)}")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"Created {len(chunks)} chunks.")
        self.retriever.build_index(chunks)
        print("Index built successfully.")

    def query(self, question: str, k: int = 3) -> str:
        if self.generator is None:
            raise ValueError("OpenAI API key not provided. Cannot generate answers.")
        context = self.retriever.search(question, k=k)
        answer = self.generator.generate(question, context)
        return answer