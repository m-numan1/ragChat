# RagChat

A PDF question-answering app built with **LangGraph**, **Groq (Llama 3.3 70B)**, and **ChromaDB**, with full request tracing via **LangSmith**. Upload PDFs, ask questions, and get answers grounded strictly in the document content — no hallucinated context.

## Features

- 📄 **PDF Upload & Ingestion** — drag-and-drop multiple PDFs directly from the Streamlit UI
- 🔍 **Semantic Search** — local sentence-transformer embeddings (`all-MiniLM-L6-v2`) with ChromaDB vector store
- 🧠 **LLM-Powered Answers** — Groq-hosted Llama 3.3 70B for fast, low-latency inference
- 🕸️ **LangGraph Pipeline** — explicit retrieve → answer graph instead of an opaque chain
- 📊 **LangSmith Tracing** — every retrieval and generation step is logged and inspectable in the LangSmith dashboard
- 🔒 **Context-Grounded Responses** — the model is instructed to answer only from retrieved context, reducing hallucination

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  PDF Upload │ --> │  Chunking +  │ --> │   Chroma    │ --> │  Retriever   │
│ (Streamlit) │     │  Embeddings  │     │ Vector Store│     │   (k=6)      │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Answer    │ <-- │  Groq LLM    │ <-- │   Prompt    │ <-- │  Context     │
│  (Display)  │     │ (Llama 3.3)  │     │  Template   │     │  Assembly    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘

All steps traced to LangSmith
```

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq — Llama 3.3 70B Versatile |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| UI | Streamlit |
| Tracing | LangSmith |
| PDF Loading | LangChain `PyPDFLoader` |

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys) (free tier available)
- A [LangSmith API key](https://smith.langchain.com/) (free tier available)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/m-numan1/ragchat.git
   cd ragchat
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=ragchat
   ```

5. **Create the docs folder**

   ```bash
   mkdir docs
   ```

## Usage

1. **Start the app**

   ```bash
   streamlit run main.py
   ```

2. **Upload PDFs** using the file uploader in the sidebar.

3. **Ask a question** in the text input and click **Get Answer**.

4. **View traces** for each query at [smith.langchain.com](https://smith.langchain.com) under the `ragchat` project — inspect retrieved chunks, prompt construction, and token usage per request.

## Project Structure

```
ragchat/
├── main.py                 # Main Streamlit app + LangGraph pipeline
├── docs/                  # Uploaded PDF storage
├── chroma_db/             # Persisted vector store (auto-generated)
├── requirements.txt
├── .env                    # API keys (not committed)
├── .gitignore
└── README.md
```

## Configuration

Key parameters you can tune in `app.py`:

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `chunk_size` | `RecursiveCharacterTextSplitter` | `800` | Larger keeps related content (e.g. CV sections) together |
| `chunk_overlap` | `RecursiveCharacterTextSplitter` | `100` | Prevents context loss at chunk boundaries |
| `k` (top-k retrieval) | `as_retriever(search_kwargs=...)` | `6` | Increase for documents with information spread across sections |
| `temperature` | `ChatGroq` | `0.2` | Lower = more deterministic, factual answers |

## Roadmap

- [ ] Multi-document source citation in answers
- [ ] Persistent chat history per session
- [ ] Support for `.docx` and `.txt` uploads
- [ ] Streaming token-by-token responses

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Muhammad Numan**
Flutter & Backend Engineer
[GitHub](https://github.com/m-numan1) · [LinkedIn](https://linkedin.com/in/muhammad-numan1)
