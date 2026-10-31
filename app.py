"""Streamlit chat UI: upload documents, then ask questions about them."""

import streamlit as st

from rag import NOT_FOUND, RAGPipeline, VectorStore, chunk_pages, load_document
from rag.config import load_settings
from rag.llm import GroqChat

settings = load_settings()

st.set_page_config(page_title="RAG Document Chat", page_icon="📄", layout="wide")


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore(persist_dir=settings.persist_dir, collection=settings.collection)


def get_pipeline(top_k: int) -> RAGPipeline:
    llm = GroqChat(api_key=settings.groq_api_key, model=settings.model)
    return RAGPipeline(get_store(), llm, top_k=top_k, min_score=settings.min_score)


store = get_store()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("PDF, TXT or Markdown", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Index files", disabled=not uploads, use_container_width=True):
        with st.spinner("Chunking and embedding..."):
            total = 0
            for upload in uploads:
                pages = load_document(upload.name, upload.getvalue())
                total += store.add(chunk_pages(pages, settings.chunk_size, settings.chunk_overlap))
        st.success(f"Indexed {total} chunks from {len(uploads)} file(s).")

    sources = store.sources()
    st.caption(f"{store.count()} chunks from {len(sources)} file(s)")
    for source in sources:
        cols = st.columns([5, 1])
        cols[0].write(source)
        if cols[1].button("✕", key=f"del-{source}", help=f"Remove {source}"):
            store.delete_source(source)
            st.rerun()

    st.divider()
    top_k = st.slider("Passages to retrieve", 1, 10, settings.top_k)
    st.caption(f"Model: `{settings.model}`")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("📄 RAG Document Chat")
st.caption("Answers come only from your documents, with a citation for every claim.")

if not settings.groq_api_key:
    st.error("Set `GROQ_API_KEY` in a `.env` file to start chatting. Get a free key at https://console.groq.com/keys")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your documents"):
    if store.count() == 0:
        st.warning("Upload and index at least one document first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and answering..."):
            history = st.session_state.messages[-4:]
            answer = get_pipeline(top_k).answer(question, history=history)
        st.markdown(answer.text)
        if answer.sources and answer.text != NOT_FOUND:
            with st.expander(f"Sources ({len(answer.sources)})"):
                if answer.query != question:
                    st.caption(f"Searched for: _{answer.query}_")
                for i, hit in enumerate(answer.sources, start=1):
                    st.markdown(f"**{i}. {hit.label}** · similarity {hit.score:.2f}")
                    st.caption(hit.text[:500] + ("..." if len(hit.text) > 500 else ""))

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer.text})
