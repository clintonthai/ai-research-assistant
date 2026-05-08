import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import tempfile

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Research Assistant")
st.markdown("Upload any PDF — earnings reports, market research, 10-Ks — and ask it anything.")
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com"
    )

    st.markdown("---")
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and groq_api_key:
        if st.button("📥 Process Document", use_container_width=True):
            with st.spinner("Reading and indexing your document..."):
                try:
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Load and split PDF
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150
                    )
                    chunks = splitter.split_documents(documents)

                    # Embeddings (free, runs locally)
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )

                    # Vector store
                    vectorstore = Chroma.from_documents(chunks, embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                    # LLM via Groq (free)
                    llm = ChatGroq(
                        api_key=groq_api_key,
                        model_name="llama3-8b-8192",
                        temperature=0.2
                    )

                    # RAG chain
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True
                    )
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.chat_history = []

                    os.unlink(tmp_path)
                    st.success(f"✅ Ready! Ask anything about **{uploaded_file.name}**")

                except Exception as e:
                    st.error(f"Error processing document: {e}")

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF\n2. Doc gets chunked + embedded\n3. Your question retrieves relevant chunks\n4. LLM answers using only your document")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
if not groq_api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started. Get one free at [console.groq.com](https://console.groq.com)")

elif not st.session_state.qa_chain:
    st.info("👈 Upload a PDF and click **Process Document** to get started.")

else:
    st.markdown(f"**Active document:** `{st.session_state.doc_name}`")
    st.markdown(" ")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask anything about your document..."):
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": question})
                    answer = result["result"]
                    sources = result.get("source_documents", [])

                    st.markdown(answer)

                    # Show source pages
                    if sources:
                        pages = sorted(set(
                            doc.metadata.get("page", "?") + 1
                            for doc in sources
                            if doc.metadata.get("page") is not None
                        ))
                        st.caption(f"📖 Sources: Page(s) {', '.join(str(p) for p in pages)}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    err = f"Something went wrong: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": err
                    })
