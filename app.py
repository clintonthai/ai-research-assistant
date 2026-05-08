import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="AI Research Assistant", page_icon="🔍", layout="centered")
st.title("🔍 AI Research Assistant")
st.markdown("Upload any PDF — earnings reports, market research, 10-Ks — and ask it anything.")
st.divider()

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

with st.sidebar:
    st.header("⚙️ Setup")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="Get a free key at console.groq.com")
    st.markdown("---")
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and groq_api_key:
        if st.button("📥 Process Document", use_container_width=True):
            with st.spinner("Reading and indexing your document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.split_documents(documents)
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = Chroma.from_documents(chunks, embeddings)
                    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
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

if not groq_api_key:
    st.info("👈 Enter your Groq API key in the sidebar. Get one free at [console.groq.com](https://console.groq.com)")
elif not st.session_state.retriever:
    st.info("👈 Upload a PDF and click **Process Document** to get started.")
else:
    st.markdown(f"**Active document:** `{st.session_state.doc_name}`")
    st.markdown(" ")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192", temperature=0.2)

                    prompt = PromptTemplate.from_template("""You are a research assistant. Answer the question using only the context below.
If the answer isn't in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:""")

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    chain = (
                        {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    answer = chain.invoke(question)
                    st.markdown(answer)

                    source_docs = st.session_state.retriever.invoke(question)
                    pages = sorted(set(
                        doc.metadata.get("page", 0) + 1
                        for doc in source_docs
                        if doc.metadata.get("page") is not None
                    ))
                    if pages:
                        st.caption(f"📖 Sources: Page(s) {', '.join(str(p) for p in pages)}")

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    err = f"Something went wrong: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err})
