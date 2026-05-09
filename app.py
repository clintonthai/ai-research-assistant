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

load_dotenv()
st.set_page_config(page_title="AI Research Assistant", page_icon="✦", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #1a0533 0%, #2d1060 40%, #1e0a4a 70%, #0f0528 100%);
    min-height: 100vh;
}

.orb {
    position: fixed; border-radius: 50%;
    filter: blur(90px); pointer-events: none; z-index: 0;
    animation: floatOrb var(--spd) ease-in-out infinite;
}
.orb1 { width:400px; height:400px; background:#7c3aed; opacity:0.25; top:-100px; left:-100px; --spd:9s; }
.orb2 { width:300px; height:300px; background:#a855f7; opacity:0.2; bottom:0; right:-80px; --spd:12s; animation-delay:-4s; }
.orb3 { width:220px; height:220px; background:#6366f1; opacity:0.18; top:45%; left:55%; --spd:7s; animation-delay:-2s; }
.orb4 { width:180px; height:180px; background:#ec4899; opacity:0.12; top:20%; right:10%; --spd:10s; animation-delay:-6s; }
@keyframes floatOrb {
    0%,100% { transform: translateY(0) scale(1); }
    50%      { transform: translateY(-30px) scale(1.06); }
}

.stars { position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:0; }
.star {
    position:absolute; border-radius:50%; background:white;
    animation: twinkle var(--dur) ease-in-out infinite;
}
@keyframes twinkle {
    0%,100% { opacity:0.1; transform:scale(1); }
    50%      { opacity:0.9; transform:scale(1.4); }
}

.main-title {
    font-size:2.6rem; font-weight:700; text-align:center;
    background: linear-gradient(135deg, #e879f9, #a78bfa, #818cf8, #c4b5fd);
    background-size:300% 300%;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation: fadeUp 0.7s ease forwards, gradientShift 4s ease infinite;
    margin-bottom:0.3rem;
}
.subtitle {
    text-align:center; color:#c4b5fd; font-size:0.95rem;
    font-weight:300; margin-bottom:1.5rem;
    animation: fadeUp 0.9s ease forwards;
}
@keyframes gradientShift {
    0%,100% { background-position:0% 50%; }
    50%      { background-position:100% 50%; }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
}

[data-testid="stSidebar"] {
    background: rgba(20,5,50,0.85) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(196,181,253,0.15) !important;
}
[data-testid="stSidebar"] * { color: #e2d9f3 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #d8b4fe !important; }

.stMarkdown p, p, li, label,
[data-testid="stMarkdownContainer"] p { color: #f0ebff !important; }
.stTextInput label, .stFileUploader label { color: #d8b4fe !important; }

.stTextInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(196,181,253,0.35) !important;
    border-radius: 10px !important; color: #f0ebff !important;
    transition: all 0.3s ease !important;
}
.stTextInput input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.25) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(168,85,247,0.55) !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(196,181,253,0.4) !important;
    border-radius: 14px !important; transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #a78bfa !important;
    background: rgba(167,139,250,0.08) !important;
}
[data-testid="stFileUploader"] * { color: #d8b4fe !important; }

[data-testid="stChatMessage"] {
    animation: fadeIn 0.4s ease forwards;
    border-radius: 16px !important;
    border: 1px solid rgba(196,181,253,0.15) !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(124,58,237,0.18) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] * { color: #f0ebff !important; }

[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px solid rgba(196,181,253,0.3) !important;
    border-radius: 14px !important; color: #f0ebff !important;
}

[data-testid="stAlert"] {
    background: rgba(124,58,237,0.15) !important;
    border: 1px solid rgba(196,181,253,0.25) !important;
    border-radius: 12px !important; animation: fadeIn 0.4s ease forwards;
}
[data-testid="stAlert"] * { color: #e9d5ff !important; }

hr { border-color: rgba(196,181,253,0.2) !important; }

.doc-badge {
    display:inline-flex; align-items:center; gap:6px;
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(168,85,247,0.2));
    color: #e9d5ff !important; padding:5px 16px; border-radius:20px;
    font-size:0.83rem; font-weight:500;
    border: 1px solid rgba(196,181,253,0.3);
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 12px rgba(124,58,237,0.2);
}

.typing-dots { display:flex; gap:5px; padding:4px 0; }
.typing-dots span {
    width:8px; height:8px; border-radius:50%;
    background: linear-gradient(135deg, #a78bfa, #e879f9);
    animation: bounce 1.1s infinite;
    box-shadow: 0 0 6px rgba(167,139,250,0.6);
}
.typing-dots span:nth-child(2) { animation-delay:0.18s; }
.typing-dots span:nth-child(3) { animation-delay:0.36s; }
@keyframes bounce {
    0%,80%,100% { transform:translateY(0) scale(1); }
    40%          { transform:translateY(-10px) scale(1.1); }
}

.stCaptionContainer *, small, .stCaption { color: #c4b5fd !important; }
</style>

<div class="stars" id="stars"></div>
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
<div class="orb orb4"></div>

<script>
(function(){
    const c = document.getElementById('stars');
    if (!c) return;
    for (let i = 0; i < 120; i++) {
        const s = document.createElement('div');
        s.className = 'star';
        const size = Math.random() * 2.5 + 0.5;
        s.style.cssText = `width:${size}px;height:${size}px;top:${Math.random()*100}%;left:${Math.random()*100}%;--dur:${(Math.random()*3+2).toFixed(1)}s;animation-delay:${(Math.random()*4).toFixed(1)}s;`;
        c.appendChild(s);
    }
})();
</script>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">✦ AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any PDF and ask it anything — powered by Groq + LangChain</div>', unsafe_allow_html=True)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 Control Panel")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.divider()
    st.markdown("### 📄 Upload Document")
    pdf = st.file_uploader("Choose a PDF", type=["pdf"])

    if pdf and api_key and st.button("📥 Index Document", use_container_width=True):
        with st.spinner("✨ Building knowledge base..."):
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(pdf.read())
                tmp.close()
                docs = PyPDFLoader(tmp.name).load()
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = Chroma.from_documents(chunks, embeddings)
                st.session_state.retriever = db.as_retriever(search_kwargs={"k": 4})
                st.session_state.chat = []
                os.unlink(tmp.name)
                st.success(f"✅ Ready! Ask anything about **{pdf.name}**")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.markdown("**How it works**")
    st.markdown("① Upload a PDF  \n② Chunked & embedded  \n③ Question finds relevant chunks  \n④ LLM answers from your doc only")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

# ── LLM + Prompt ──────────────────────────────────────────────────────────────
def get_llm(key):
    return ChatGroq(api_key=key, model_name="llama-3.1-8b-instant", temperature=0.2)

prompt = PromptTemplate.from_template("""You are a research assistant. Answer using ONLY the context below.
If the answer isn't in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ── Main chat ─────────────────────────────────────────────────────────────────
if not api_key:
    st.info("👈 Enter your Groq API key to get started. Free at [console.groq.com](https://console.groq.com)")
elif not st.session_state.retriever:
    st.info("👈 Upload a PDF and click **Index Document** to begin.")
else:
    st.markdown(f'<div style="margin-bottom:1.2rem"><span class="doc-badge">📄 Document loaded</span></div>', unsafe_allow_html=True)

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            thinking = st.empty()
            thinking.markdown('<div class="typing-dots"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
            try:
                chain = (
                    {"context": st.session_state.retriever | format_docs, "question": lambda x: x}
                    | prompt | get_llm(api_key) | StrOutputParser()
                )
                answer = chain.invoke(question)
                thinking.empty()
                st.markdown(answer)
                st.session_state.chat.append({"role": "assistant", "content": answer})
            except Exception as e:
                thinking.empty()
                st.error(f"Something went wrong: {e}")
