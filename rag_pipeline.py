import time
import chromadb
import ollama
from config import MODEL_NAME, CHUNK_SIZE, TOP_K

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="corpus_docs")

def charger_et_decouper(file_path, chunk_size):
    """Charge le texte et le découpe en segments."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        return []
    
    segments = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return segments

def initialiser_base(segments):
    """Calcule les embeddings et stocke les segments dans ChromaDB."""
    if not segments:
        return
    
    documents = segments
    ids = [f"doc_{i}" for i in range(len(segments))]
    
    collection.add(
        documents=documents,
        ids=ids
    )

def interroger_rag(question):
    """Recherche les segments et interroge le LLM via Ollama."""
    start_time = time.time()
    
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K
    )
    
    passages = results['documents'][0] if results['documents'] else []
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
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt}
        ])
        reponse_texte = response['message']['content']
        statut = "Succès"
    except Exception as e:
        reponse_texte = "Erreur : Impossible de joindre Ollama."
        statut = "Échec de connexion"

    temps_traitement = round(time.time() - start_time, 2)
    
    return {
        "reponse": reponse_texte,
        "passages": passages,
        "temps": temps_traitement,
        "statut": statut
    }