import streamlit as st
from config import MODEL_NAME, CHUNK_SIZE, TOP_K, CORPUS_PATH
from rag_pipeline import charger_et_decouper, initialiser_base, interroger_rag

@st.cache_resource
def init_rag():
    segments = charger_et_decouper(CORPUS_PATH, CHUNK_SIZE)
    initialiser_base(segments)
    return True

init_rag()

st.title("Assistant Documentaire Lite RAG")
st.write(f"Posez une question sur le document. (Modèle: `{MODEL_NAME}` | Segments récupérés: `{TOP_K}`)")

question = st.text_input("Votre question :")

if st.button("Rechercher"):
    if not question.strip():
        st.warning("Veuillez entrer une question valide.")
    else:
        with st.spinner("Recherche dans le document et génération de la réponse..."):
            resultat = interroger_rag(question)
            
            st.markdown("### Réponse de l'IA")
            if "Erreur" in resultat["reponse"]:
                st.error(resultat["reponse"])
            else:
                st.success(resultat["reponse"])
            
            st.markdown("### Métriques de traitement")
            col1, col2, col3 = st.columns(3)
            col1.metric("Temps de traitement", f"{resultat['temps']} s")
            col2.metric("Segments récupérés", len(resultat["passages"]))
            col3.metric("Statut", resultat["statut"])
            
            with st.expander("Voir les passages utilisés (Contexte)"):
                if resultat["passages"]:
                    for i, passage in enumerate(resultat["passages"]):
                        st.info(f"**Extrait {i+1} :**\n{passage}")
                else:
                    st.write("Aucun passage pertinent trouvé.")