# Baseline RAG

Minimal, modular RAG baseline for experimentation.

- Python
- Pluggable components: preprocessing (chunking), embeddings, vector store, retriever, generator
- Default: OpenAI for generator/embeddings (configurable)

Quickstart
1. Create virtualenv and install requirements: `pip install -r requirements.txt`
2. Put documents into `data/` and run `python scripts/index.py --data-dir data/` to index
3. Run `python scripts/query.py --index-dir ./index --query "your question"`

See `baseline_demo.ipynb` for a demo.
