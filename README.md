# Examen Pratique MLOps : Assistant Documentaire Lite RAG

## Description
Application web Streamlit permettant d'interroger un corpus documentaire via un pipeline RAG local. 
Elle utilise ChromaDB pour la recherche vectorielle et Ollama pour la génération de texte.

## Prérequis
- Docker
- Ollama (installé sur l'environnement hôte/Codespace) avec le modèle `qwen2.5:0.5b` téléchargé.

## Paramètres (Configurables dans `.env`)
- `OLLAMA_MODEL` : Modèle utilisé (ex: qwen2.5:0.5b).
- `CHUNK_SIZE` : Taille des segments de découpage du texte (ex: 500).
- `TOP_K` : Nombre de segments récupérés pour le contexte (ex: 3).

## Installation et Lancement

1. Lancer Ollama en arrière-plan :
   ```bash
   ollama serve