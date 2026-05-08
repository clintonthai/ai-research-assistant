[README.md](https://github.com/user-attachments/files/27532753/README.md)
# 🔍 AI Research Assistant

A RAG (Retrieval-Augmented Generation) app that lets you upload any PDF and ask it questions using natural language. Built with LangChain, Groq, ChromaDB, and Streamlit.

**Use cases:** Earnings reports, market research, 10-Ks, whitepapers, contracts — anything in PDF form.

**[Live Demo](https://your-app-name.streamlit.app)** ← update this after deploying

---

## How It Works

```
PDF → chunks → embeddings → ChromaDB vector store
                                      ↓
          Question → relevant chunks retrieved → Groq LLM → Answer
```

1. Your PDF is split into overlapping chunks (~1000 tokens each)
2. Each chunk is embedded using a local HuggingFace sentence transformer
3. Your question is embedded the same way and matched against chunks via cosine similarity
4. The top 4 most relevant chunks are passed to the LLM with your question
5. The LLM answers using *only* your document — no hallucinated outside knowledge

---

## Stack

| Component | Tool | Cost |
|---|---|---|
| UI | Streamlit | Free |
| LLM | Groq (Llama 3 8B) | Free tier |
| Embeddings | HuggingFace sentence-transformers | Free, runs locally |
| Vector Store | ChromaDB | Free, runs locally |
| Orchestration | LangChain | Free |

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/clintonthai/ai-research-assistant
cd ai-research-assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get a free Groq API key**
Go to [console.groq.com](https://console.groq.com) → sign up → create an API key (takes 2 minutes)

**4. Run the app**
```bash
streamlit run app.py
```

Paste your Groq API key into the sidebar when prompted.

---

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Deploy — done

No secrets needed in Streamlit Cloud — users paste their own Groq key in the sidebar.

---

## Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
└── README.md
```

---

## Features

- 📄 Upload any PDF via drag & drop
- 💬 Chat interface with persistent history
- 📖 Source page citations for every answer
- ⚡ Fast responses via Groq's free LLM API
- 🔒 API key stays in your browser session, never stored

---

Built by [Clinton Thai](https://github.com/clintonthai)
