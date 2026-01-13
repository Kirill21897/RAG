import sys
import os
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
# Импортируем обертки для работы через API
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- 1. НАСТРОЙКА ИМПОРТА ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.pipeline import RAGPipeline
    
    rag_obj = RAGPipeline(model_name="gpt-oss:20b") 
    
    doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/sample.pdf"))
    
    if not os.path.exists(doc_path):
        print(f"Критическая ошибка: Файл не найден по пути {doc_path}")
        exit()

    print(f"Индексируем базу знаний: {doc_path}...")
    rag_obj.ingest(doc_path)
    
    def query_rag(question: str):
        return rag_obj.query(question)

except ImportError as e:
    print(f"Ошибка импорта: {e}")
    exit()

# --- 2. ТЕСТОВЫЕ ДАННЫЕ ---
test_questions = pd.read_json("../data/golden_dataset.json")
print(test_questions)

# --- 3. ЗАПУСК ПРОВЕРКИ ---
def run():
    print("Запускаем сбор ответов от локального RAG...")
    
    data_samples = {
        "question": [],
        "answer": [],
        "retrieved_contexts": [],
        "ground_truth": []
    }

    for _, item in test_questions.iterrows():
        q = item["question"]
        print(f"Обработка вопроса: {q}")
        
        response = query_rag(q) 
        
        # Убираем ожидание API, так как локальный сервер не забанит по RPM
        # но можно оставить 1 сек, чтобы сервер не перегрелся
        time.sleep(1)
        
        if isinstance(response, str):
            ans = response
            ctx = ["Context not found"]
        else:
            ans = response["answer"]
            # Важно: берем ключ 'contexts', который мы поправили в pipeline.py
            ctx = response["contexts"]

        data_samples["question"].append(q)
        data_samples["answer"].append(ans)
        data_samples["retrieved_contexts"].append(ctx)
        data_samples["ground_truth"].append(item["ground_truth"])

    # Создаем Dataset для Ragas
    dataset = Dataset.from_dict(data_samples)

    # --- НАСТРОЙКА ЛОКАЛЬНОГО СУДЬИ (Ragas) ---
    # Мы создаем "фейковые" объекты OpenAI, которые на самом деле шлют запросы на твой сервер
    local_server_url = "http://192.168.88.21:91/v1" # Большинство серверов поддерживают /v1
    
    judge_llm = ChatOpenAI(
        model="gpt-oss:20b", 
        base_url=local_server_url, 
        api_key="none"
    )
    
    # Эмбеддинги для оценки (нужны для метрики answer_relevancy)
    judge_embeddings = OpenAIEmbeddings(
        model="gpt-oss:20b", 
        base_url=local_server_url, 
        api_key="none"
    )

    print("Считаем метрики через локальный сервер (это может занять время)...")
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,           # Передаем локального судью
        embeddings=judge_embeddings, # Передаем локальные эмбеддинги
        run_config=RunConfig(
            max_retries=3, 
            max_workers=1 # На локалке лучше считать по очереди
        )
    )

    # Сохраняем и показываем
    df = results.to_pandas()
    output_file = os.path.join(os.path.dirname(__file__), "rag_report.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\nОтчет сохранен в: {output_file}")
    print(df[['question', 'faithfulness', 'answer_relevancy', 'context_precision']])

if __name__ == "__main__":
    run()