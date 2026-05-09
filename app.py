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

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Research Lab V8", page_icon="🧪", layout="centered")

# ─────────────────────────────────────────
# UI (your beige gradient restored)
# ─────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #E4E7E4, #ABA3AC, #8D828E);
}

.title {
    text-align:center;
    font-size:2.2rem;
    font-weight:600;
    background: linear-gradient(135deg,#C3C0C2,#E4E7E4);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.chat-bubble {
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from {opacity:0; transform:translateY(6px);}
    to {opacity:1; transform:translateY(0);}
}
</style>

<div class="title">🧪 Research Organization V8</div>
""", unsafe_allow_html=True)

# floating orbs
st.markdown("""
<div style="position:fixed; width:300px; height:300px; background:#a78bfa;
filter:blur(90px); opacity:0.15; top:-50px; left:-50px; border-radius:50%"></div>

<div style="position:fixed; width:250px; height:250px; background:#c4b5fd;
filter:blur(90px); opacity:0.15; bottom:40px; right:-60px; border-radius:50%"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# ─────────────────────────────────────────
# SIDEBAR (INGESTION)
# ─────────────────────────────────────────
with st.sidebar:
    st.header("🧠 Lab Control Panel")

    api_key = st.text_input("Groq API Key", type="password")
    pdf = st.file_uploader("Upload Research Paper", type=["pdf"])

    if pdf and api_key and st.button("Index Paper"):
        with st.spinner("Building knowledge graph..."):

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

            db = Chroma.from_documents(chunks, embeddings)

            st.session_state.retriever = db.as_retriever(search_kwargs={"k": 4})
            st.session_state.chat = []

            os.unlink(tmp.name)

            st.success("Knowledge base ready.")

# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────
def get_llm(key):
    return ChatGroq(
        api_key=key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

prompt = PromptTemplate.from_template("""
You are a scientific research assistant.

Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
Include citations if possible.
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ─────────────────────────────────────────
# MAIN CHAT LOOP (ALWAYS ON)
# ─────────────────────────────────────────
if not api_key:
    st.info("Enter Groq API key to begin.")

elif not st.session_state.retriever:
    st.info("Upload a PDF and index it.")

else:

    # history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # INPUT ALWAYS VISIBLE
    question = st.chat_input("Ask your research question...")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            llm = get_llm(api_key)

            chain = (
                {
                    "context": st.session_state.retriever | format_docs,
                    "question": lambda x: x
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = chain.invoke(question)

            st.markdown(answer)

            st.session_state.chat.append(
                {"role": "assistant", "content": answer}
            )
