"""
Streamlit demo: agentic movie discovery over the TMDB 5000 dataset.

Run: streamlit run app.py
Requires GEMINI_API_KEY in the environment (see .env.example).
"""
from __future__ import annotations

import os

# Must be set before sentence-transformers/torch is imported anywhere,
# to avoid the local OpenMP runtime conflict some environments hit.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from src.agent import ConversationState, run_agent_turn
from src.data_preprocessing import load_processed
from src.fuzzy_search import FuzzyIndex
from src.semantic_search import SemanticIndex

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

st.set_page_config(page_title="TMDB Movie Agent", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading and preprocessing the TMDB dataset...")
def _load_df() -> pd.DataFrame:
    return load_processed()


@st.cache_resource(show_spinner="Building fuzzy title index...")
def _load_fuzzy_index(_df: pd.DataFrame) -> FuzzyIndex:
    return FuzzyIndex(_df)


@st.cache_resource(show_spinner="Loading semantic embeddings (first run may take a while)...")
def _load_semantic_index(_df: pd.DataFrame) -> SemanticIndex:
    return SemanticIndex(_df)


def _init_state():
    if "conv_state" not in st.session_state:
        st.session_state.conv_state = ConversationState()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Gemini `contents` list, for multi-turn
    if "ui_history" not in st.session_state:
        st.session_state.ui_history = []  # [(role, text, trace)] for display


def render_trace(trace: list[dict]):
    """Visible tool-selection trace: which tool ran, with what args, and what it returned."""
    for step in trace:
        tool = step["tool"]
        result = step.get("result", {})
        with st.expander(f"🔧 {tool}", expanded=False):
            st.markdown("**Arguments:**")
            st.json(step.get("args", {}))
            st.markdown("**Result:**")
            if isinstance(result, dict) and "error" in result:
                st.error(result["error"])
            elif isinstance(result, dict) and result.get("status") in ("ambiguous", "not_found"):
                st.warning(f"Status: {result['status']}")
                if result.get("candidates"):
                    st.write("Candidates:", result["candidates"])
            elif isinstance(result, dict) and result.get("results") and tool == "structured_search":
                st.dataframe(pd.DataFrame(result["results"]), use_container_width=True)
                if "total_matching" in result:
                    st.caption(f"{result['total_matching']} total matching rows (showing {result.get('shown')})")
                if "count" in result:
                    st.caption(f"Count: {result['count']}")
            elif isinstance(result, dict) and result.get("results") and tool == "semantic_search":
                st.dataframe(pd.DataFrame(result["results"]), use_container_width=True)
                if result.get("status") == "low_confidence":
                    st.warning("Low-confidence retrieval — results may not be a strong semantic match.")
            elif isinstance(result, dict) and result.get("answer"):
                st.markdown("**Retrieved context grounding this answer:**")
                st.write(result.get("grounded_on", []))
            else:
                st.json(result)


def main():
    _init_state()
    df = _load_df()
    fuzzy_index = _load_fuzzy_index(df)
    semantic_index = _load_semantic_index(df)

    st.title("🎬 TMDB Movie Discovery Agent")
    st.caption(
        f"{len(df):,} movies · structured search + fuzzy matching + semantic RAG · "
        "agentic tool routing with multi-turn memory"
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.warning(
            "GEMINI_API_KEY is not set. Add it to a `.env` file (see `.env.example`) "
            "or your shell environment, then restart the app.",
            icon="⚠️",
        )
        st.stop()

    client = genai.Client(api_key=api_key)

    with st.sidebar:
        st.subheader("About this demo")
        st.markdown(
            "- **structured_search** — deterministic filters/aggregates\n"
            "- **fuzzy_movie_search** — typo-tolerant title resolution\n"
            "- **semantic_search** — thematic/plot similarity (RAG retrieval)\n"
            "- **movie_details** — full record lookup\n"
            "- **rag_answer** — grounded natural-language generation\n"
        )
        st.subheader("Try asking")
        examples = [
            "What are the top 10 movies by revenue?",
            "How many movies are there in each genre?",
            "Show me the 10 highest-rated science fiction movies with at least 1000 votes",
            "Tell me about Intersteler",
            "I want a movie about someone trying to survive alone on another planet",
            "Find movies similar to Inception",
            "Show me science fiction movies from after 2010",
            "Only show ones rated above 7.5",
            "Tell me more about the first one",
            "What movies are similar to lord of the rings?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
                st.session_state["pending_query"] = ex
        if st.button("🔄 Reset conversation", use_container_width=True):
            st.session_state.conv_state = ConversationState()
            st.session_state.chat_history = []
            st.session_state.ui_history = []
            st.rerun()

    for role, text, trace in st.session_state.ui_history:
        with st.chat_message(role):
            st.markdown(text)
            if trace:
                render_trace(trace)

    query = st.chat_input("Ask about movies...")
    if not query and "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")

    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result, new_history = run_agent_turn(
                    query, df, fuzzy_index, semantic_index, client, MODEL_NAME,
                    st.session_state.conv_state, st.session_state.chat_history,
                )
            st.markdown(result.text_response)
            if result.trace:
                render_trace(result.trace)

        st.session_state.chat_history = new_history
        st.session_state.ui_history.append(("user", query, None))
        st.session_state.ui_history.append(("assistant", result.text_response, result.trace))


if __name__ == "__main__":
    main()