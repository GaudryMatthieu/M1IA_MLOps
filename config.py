import os

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
TOP_K = int(os.getenv("TOP_K", 3))
CORPUS_PATH = os.getenv("CORPUS_PATH", "corpus.txt")