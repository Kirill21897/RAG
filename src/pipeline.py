from src.document_loader import load_document
from src.chunker import chunk_text
from src.retriever import VectorRetriever
from src.generator import Generator
import os

class RAGPipeline:
    def __init__(self, model_name: str = "qwen3-vl:8b", embedding_model: str = "all-MiniLM-L6-v2"):
        # Ретривер (векторная база) работает локально на твоем ПК
        self.retriever = VectorRetriever(model_name=embedding_model)
        # Генератор стучится на твой сервер 192.168.88.21
        self.generator = Generator(model_name=model_name)

    def ingest(self, file_path: str):
        print(f"Индексация файла: {file_path}")
        text = load_document(file_path)
        chunks = chunk_text(text)
        self.retriever.build_index(chunks)
        print("Индекс успешно построен.")

    def query(self, question: str, k: int = 3) -> dict:
        # 1. Ищем релевантные куски текста локально
        context = self.retriever.search(question, k=k)
        
        # 2. Отправляем вопрос и контекст на удаленный сервер
        answer = self.generator.generate(question, context)
        
        return {
            "answer": answer,
            "contexts": context # Для Ragas
        }