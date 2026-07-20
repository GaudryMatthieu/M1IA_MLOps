import os
from dotenv import load_dotenv

# Charge les variables du fichier .env si présent
load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
TOP_K = int(os.getenv("TOP_K", 3))
CORPUS_PATH = os.getenv("CORPUS_PATH", "corpus.txt")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")