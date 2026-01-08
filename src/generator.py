# src/generator.py
from openai import OpenAI
from typing import List

class Generator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
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