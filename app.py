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

st.set_page_config(page_title="AI Research Assistant", page_icon="🔍", layout="centered")

# ── YOUR ORIGINAL UI (restored + safe) ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #E4E7E4, #ABA3AC, #8D828E);
}

/* floating orbs */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.18;
    pointer-events: none;
    animation: float 8s ease-in-out infinite;
    z-index: 0;
}
.orb1 { width: 340px; height: 340px; background: #a78bfa; top: -80px; left: -80px; }
.orb2 { width: 260px; height: 260px; background: #c4b5fd; bottom: 60px; right: -60px; }
.orb3 { width: 200px; height: 200px; background: #818cf8; top: 40%; left: 50%; }

@keyframes float {
    0%,100% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(-24px) scale(1.04); }
}

/* title */
.title {
    font-size: 2.4rem;
    font-weight: 600;
    text-align: center;
    background: linear-gradient(135deg, #C3C0C2, #D5D3D5, #E4E7E4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 1rem;
}

/* chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #E4E7E4, #C9C2CA, #AD9DB0);
    color: white;
    border-radius: 12px;
    border: none;
}
</style>

<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>✦ AI Research Assistant</div>", unsafe_allow_html=True)
st.divider()

# ── STATE ────────────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")

    groq_api_key = st.text_input("Groq API Key", type="password")
    file = st.file_uploader("Upload PDF", type=["pdf"])

    if file and groq_api_key:
        if st.button("Process"):
            with st.spinner("Indexing..."):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(file.read())

                docs = PyPDFLoader(tmp.name).load()
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=150
                ).split_documents(docs)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                vector = Chroma.from_documents(chunks, embeddings)

                st.session_state.retriever = vector.as_retriever(k=4)
                st.session_state.chat = []

                os.unlink(tmp.name)
                st.success("Ready!")

# ── MODEL ────────────────────────────────────────────────────────────────────
def get_llm(key):
    return ChatGroq(
        api_key=key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

# ── MAIN UI ───────────────────────────────────────────────────────────────────
if not groq_api_key:
    st.info("Enter API key")

elif not st.session_state.retriever:
    st.info("Upload PDF")

else:
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask your document...")

    if q:
        st.session_state.chat.append({"role": "user", "content": q})

        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            llm = get_llm(groq_api_key)

            prompt = PromptTemplate.from_template("""
Use only context.

{context}

Q: {question}
A:
""")

            def fmt(docs):
                return "\n\n".join(d.page_content for d in docs)

            chain = (
                {"context": st.session_state.retriever | fmt,
                 "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )

            ans = chain.invoke(q)
            st.markdown(ans)

            st.session_state.chat.append({"role": "assistant", "content": ans})
