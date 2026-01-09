# src/document_loader.py
from docling.document_converter import DocumentConverter
import os
from pathlib import Path

def load_document(file_path: str, save_intermediate: bool = True) -> str:
    """
    Загружает документ, опционально сохраняет промежуточный файл в формате .md
    и возвращает текст.
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
    
    if save_intermediate:
        # Сохраняем рядом с исходным файлом
        converter_path = f"{os.path.dirname(file_path)}" # убираем расширение
        save_path = f"{converter_path}_converted.txt"
        md_content = result.document.export_to_text()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Saved intermediate file: {save_path}")
    
    return result.document.export_to_text()