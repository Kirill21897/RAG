# src/generator.py
from openai import OpenAI
from typing import List
import os

class Generator:
    api_key = os.getenv("OPENAI_API_KEY")
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str, context: List[str]) -> str:
        context_str = "\n\n".join(context)
        prompt = f"""Answer the question based on the context below.
        
Context:
{context_str}

Question: {question}
Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256
        )
        return response.choices[0].message.content.strip()