from __future__ import annotations

import time

import streamlit as st

from app.config import get_settings
from app.models import DocumentType, QueryRequest
from app.rag.pipeline import build_pipeline
from app.utils.openrouter_client import OpenRouterError

st.set_page_config(
    page_title="Tanger Med Knowledge Assistant",
    page_icon="⚓",
    layout="wide",
)

DOC_TYPE_LABELS = {
    None: "Tous les rapports",
    DocumentType.ANNUAL_REPORT: "Rapport Annuel",
    DocumentType.RSE_REPORT: "Rapport RSE",
}


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return build_pipeline()


def _available_years(pipeline) -> list[int]:
    docs = pipeline.list_documents()
    return sorted({d.year for d in docs})


def main() -> None:
    settings = get_settings()

    st.title("⚓ Tanger Med Knowledge Assistant")
    st.caption(
        "Système RAG multi-années pour l'exploration des Rapports Annuels et RSE "
        "du Groupe Tanger Med ."
    )

    if not settings.has_api_key():
        st.warning(
            "⚠️ `OPENROUTER_API_KEY` n'est pas configurée. Renseignez-la dans un "
            "fichier `.env` (voir `.env.example`) pour pouvoir interroger le modèle.",
            icon="⚠️",
        )

    try:
        pipeline = get_pipeline()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Impossible d'initialiser le pipeline RAG : {exc}")
        st.stop()

    indexed_chunks = pipeline.vector_store.count()
    if indexed_chunks == 0:
        st.error(
            "Aucun document n'est indexé. Placez des PDF dans `data/annual_reports/` "
            "et/ou `data/rse_reports/`, puis lancez :\n\n"
            "```\npython -m app.ingestion\n```"
        )
        st.stop()

    years = _available_years(pipeline)

    with st.sidebar:
        st.subheader("Filtres")
        year_options = ["Toutes"] + [str(y) for y in years]
        selected_year_label = st.selectbox("Années", year_options, index=0)
        selected_years = None if selected_year_label == "Toutes" else [int(selected_year_label)]

        doc_type_label = st.selectbox(
            "Type de rapport",
            list(DOC_TYPE_LABELS.values()),
            index=0,
        )
        selected_doc_type = next(k for k, v in DOC_TYPE_LABELS.items() if v == doc_type_label)

        show_debug = st.toggle("Afficher le panneau Debug / Retrieval", value=False)

        st.divider()
        st.subheader("Corpus indexé")
        docs = pipeline.list_documents()
        for d in docs:
            label = "RSE" if d.document_type == DocumentType.RSE_REPORT else "Annuel"
            st.caption(f"📄 {label} {d.year} — {d.chunk_count} chunks")

    st.subheader("Poser une question")
    example_questions = [
        "Quel a été le tonnage global traité par Tanger Med ?",
        "Quelles sont les principales actions environnementales décrites dans le rapport RSE ?",
        "Quels sont les objectifs ESG du Groupe en matière de gouvernance ?",
        "Quel était le chiffre d'affaires du Groupe en 2027 ?",  # volontairement hors corpus
    ]
    picked_example = st.selectbox(
        "Exemples de questions (optionnel)", ["—"] + example_questions, index=0
    )
    default_text = "" if picked_example == "—" else picked_example

    question = st.text_area("Question", value=default_text, height=90, placeholder=(
        "Ex : Comparez le trafic conteneurs et le chiffre d'affaires entre 2024 et 2025."
    ))

    ask = st.button("Poser la question", type="primary")

    if ask and question.strip():
        request = QueryRequest(
            question=question.strip(),
            years=selected_years,
            document_type=selected_doc_type,
        )

        with st.spinner("Recherche dans le corpus et génération de la réponse..."):
            start = time.perf_counter()
            try:
                response = pipeline.answer(request)
            except OpenRouterError as exc:
                st.error(f"Erreur du fournisseur LLM (OpenRouter) : {exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur inattendue : {exc}")
                st.stop()
            elapsed = time.perf_counter() - start

        st.markdown("## Réponse")
        if response.abstained:
            st.info(response.answer, icon="🤷")
        else:
            st.markdown(response.answer)

        if response.sources:
            st.markdown("## Sources")
            for src in response.sources:
                type_label = "RSE" if src.document_type == DocumentType.RSE_REPORT else "Annuel"
                st.markdown(f"**{src.document}** ({type_label}) — {src.page}")

        if show_debug:
            # Le détail (sections, scores, extraits bruts) est ici, pas dans la
            # réponse principale qui reste volontairement épurée.
            st.markdown("##  Debug / Retrieval")
            st.json(
                {
                    "temps_total_s": round(elapsed, 2),
                    "timings_ms": response.timings_ms,
                    "analyse_requete": response.query_analysis.model_dump(),
                    "nb_chunks_retrouves": response.retrieved_count,
                    "abstention": response.abstained,
                    "sources_detaillees": [s.model_dump(mode="json") for s in response.sources],
                }
            )

    elif ask:
        st.warning("Veuillez saisir une question.")


if __name__ == "__main__":
    main()
