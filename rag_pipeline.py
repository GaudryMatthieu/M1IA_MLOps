import os
import time
import chromadb
from ollama import Client
from config import MODEL_NAME, CHUNK_SIZE, TOP_K, OLLAMA_HOST

# Initialisation de la base vectorielle locale en mémoire
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="corpus_docs")

# Initialisation du client Ollama avec l'URL dynamique
ollama_client = Client(host=OLLAMA_HOST)

def charger_et_decouper(file_path, chunk_size):
    """Charge le fichier texte et le découpe en segments."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return []
    
    # Découpage par blocs de caractères
    segments = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return segments

def initialiser_base(segments):
    """Calcule les hébergements (embeddings) et stocke les segments dans ChromaDB."""
    if not segments:
        return
    
    documents = segments
    ids = [f"doc_{i}" for i in range(len(segments))]
    
    # ChromaDB s'occupe des embeddings par défaut
    collection.add(
        documents=documents,
        ids=ids
    )

def interroger_rag(question):
    """Effectue la recherche vectorielle et génère la réponse via le LLM."""
    start_time = time.time()
    
    # 1. Recherche des segments pertinents
    try:
        results = collection.query(
            query_texts=[question],
            n_results=TOP_K
        )
        passages = results['documents'][0] if results['documents'] else []
    except Exception as e:
        passages = []
        print(f"Erreur lors de la recherche vectorielle : {e}")

    contexte = "\n---\n".join(passages)
    
    prompt = f"""Tu es un assistant documentaire. 
Utilise UNIQUEMENT le contexte fourni pour répondre à la question. 
Ne pas inventer d'information. 
Si la réponse n'est pas dans le contexte, dis exactement : "L'information n'est pas présente dans le corpus."

Contexte :
{contexte}

Question : {question}
Réponse :"""

    try:
        response = ollama_client.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt}
        ])
        reponse_texte = response['message']['content']
        statut = "Succès"
    except Exception as e:
        print(f"Erreur de connexion Ollama sur {OLLAMA_HOST} : {e}")
        reponse_texte = f"Erreur : Impossible de joindre le modèle Ollama."
        statut = "Échec de connexion"

    temps_traitement = round(time.time() - start_time, 2)
    
    return {
        "reponse": reponse_texte,
        "passages": passages,
        "temps": temps_traitement,
        "statut": statut
    }