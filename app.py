import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="centered"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ── Groq model fallback system ────────────────────────────────────────────────
def get_llm(api_key):
    models = [
        "llama-3.1-8b-instant",
        "llama3-8b-8192"
    ]

    for m in models:
        try:
            return ChatGroq(
                api_key=api_key,
                model_name=m,
                temperature=0.2
            )
        except Exception:
            continue

    raise Exception("No working Groq model available for this API key")


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 AI Research Assistant")
st.caption("Upload a PDF and chat with it using Groq + LangChain")

st.divider()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file and groq_api_key:
        if st.button("Process Document"):
            with st.spinner("Indexing document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        path = tmp.name

                    loader = PyPDFLoader(path)
                    docs = loader.load()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150
                    )
                    chunks = splitter.split_documents(docs)

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )

                    vectorstore = Chroma.from_documents(chunks, embeddings)

                    st.session_state.retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    )

                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.chat_history = []

                    os.unlink(path)

                    st.success("Document ready!")

                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── Main UI ────────────────────────────────────────────────────────────────────
if not groq_api_key:
    st.info("Enter Groq API key in sidebar to start.")

elif not st.session_state.retriever:
    st.info("Upload and process a PDF to begin chatting.")

else:
    st.subheader(f"📄 {st.session_state.doc_name}")

    # show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # chat input (ALWAYS reachable)
    question = st.chat_input("Ask something about your document...")

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm(groq_api_key)

                    prompt = PromptTemplate.from_template("""
You are a research assistant.
Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer:
""")

                    def format_docs(docs):
                        return "\n\n".join(d.page_content for d in docs)

                    chain = (
                        {
                            "context": st.session_state.retriever | format_docs,
                            "question": RunnablePassthrough()
                        }
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    answer = chain.invoke(question)

                    st.markdown(answer)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(str(e))
