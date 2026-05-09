import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="AI Research Assistant", page_icon="🔍", layout="centered")

# ── CLEAN UI (your aesthetic preserved) ───────
st.markdown("""
<style>
* { font-family: Inter, sans-serif; }

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

.title {
    font-size: 2.4rem;
    text-align: center;
    font-weight: 600;
    background: linear-gradient(135deg, #C3C0C2, #D5D3D5, #E4E7E4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.chat-bubble {
    border-radius: 16px;
    padding: 12px;
}
</style>

<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>✦ AI Research Assistant</div>", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
if "db" not in st.session_state:
    st.session_state.db = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input("Groq API Key", type="password")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf and api_key:
        if st.button("Index Document"):
            with st.spinner("Processing..."):

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(pdf.read())
                tmp.close()

                docs = PyPDFLoader(tmp.name).load()

                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                ).split_documents(docs)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                db = FAISS.from_documents(chunks, embeddings)

                st.session_state.db = db
                st.session_state.chat = []

                os.unlink(tmp.name)

                st.success("Document ready 🚀")

# ─────────────────────────────────────────────
# LLM (with fallback-safe model)
# ─────────────────────────────────────────────
def get_llm(key):
    return ChatGroq(
        api_key=key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────
def retrieve(query):
    docs = st.session_state.db.similarity_search(query, k=4)

    context = "\n\n".join(
        f"(Page {d.metadata.get('page', '?')}) {d.page_content}"
        for d in docs
    )

    return context, docs

# ─────────────────────────────────────────────
# CHAT UI (ALWAYS STABLE)
# ─────────────────────────────────────────────
if not api_key:
    st.info("Enter Groq API key")

elif not st.session_state.db:
    st.info("Upload PDF and index it")

else:
    # render history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask your document...")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            llm = get_llm(api_key)

            context, docs = retrieve(question)

            prompt = PromptTemplate.from_template("""
You are a precise research assistant.

Use ONLY this context:
{context}

Question:
{question}

If not found, say you don't know.

Answer clearly and concisely.
""")

            chain = prompt | llm | StrOutputParser()

            answer = chain.invoke({
                "context": context,
                "question": question
            })

            st.markdown(answer)

            # citations
            pages = sorted({
                d.metadata.get("page", 0) + 1 for d in docs
                if d.metadata.get("page") is not None
            })

            if pages:
                st.caption(f"📖 Sources: pages {pages}")

            st.session_state.chat.append(
                {"role": "assistant", "content": answer}
            )
