import streamlit as st
from config import MODEL_NAME, CHUNK_SIZE, TOP_K, CORPUS_PATH
from rag_pipeline import charger_et_decouper, initialiser_base, interroger_rag
st.set_page_config(page_title="Assistant RAG Lite", layout="wide")

# --- Initialisation automatique du RAG au démarrage ---
@st.cache_resource
def init_application():
    segments = charger_et_decouper(CORPUS_PATH, CHUNK_SIZE)
    if segments:
        initialiser_base(segments)
        return True
    return False

rag_pret = init_application()

# --- Interface Utilisateur Streamlit ---

st.title("Assistant Documentaire Lite RAG 📚")
st.write("Posez vos questions en langage naturel basées exclusivement sur le corpus documentaire fourni.")

if not rag_pret:
    st.error(f"Le fichier `{CORPUS_PATH}` est manquant à la racine. Veuillez le rajouter pour initialiser l'application.")

# Formulaire de saisie
question = st.text_input("Quelle est votre question ?")

if st.button("Lancer la recherche", type="primary"):
    # Gestion d'une question vide
    if not question.strip():
        st.warning("Veuillez saisir une question avant de lancer la recherche.")
    
    elif not rag_pret:
        st.error("Impossible de répondre : le corpus n'est pas chargé.")
        
    else:
        with st.spinner("Analyse du document et traitement par l'IA en cours..."):
            # Appel du Pipeline
            resultat = interroger_rag(question)
            
            # Affichage de la réponse
            st.markdown("### 🤖 Réponse générée")
            if "Erreur :" in resultat["reponse"]:
                st.error(resultat["reponse"])
            else:
                st.info(resultat["reponse"])
            
            # Affichage des métriques (Suivi MLOps simplifié)
            st.markdown("### 📊 Métriques & Suivi MLOps")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Modèle utilisé", MODEL_NAME)
            col2.metric("Temps de traitement", f"{resultat['temps']} s")
            col3.metric("Segments demandés / reçus", f"{TOP_K} / {len(resultat['passages'])}")
            col4.metric("Statut de la requête", resultat["statut"])
            
            # Affichage des passages utilisés
            st.markdown("### 📄 Passages du corpus consultés")
            with st.expander("Cliquez pour visualiser les segments de texte extraits"):
                if resultat["passages"]:
                    for i, passage in enumerate(resultat["passages"]):
                        st.subheader(f"Segment n°{i+1}")
                        st.code(passage, language="text")
                else:
                    st.write("Aucun segment n'a pu être extrait pour cette requête.")