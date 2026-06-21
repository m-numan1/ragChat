from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv
import streamlit as st
from langsmith import traceable

load_dotenv()

@st.cache_resource
def build_app():
    loader = DirectoryLoader(r"D:\ragChat\docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    class GraphState(TypedDict):
        question: str
        answer: str
        documents: List[Document]

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    @traceable(name = "retriever")
    def retrieve(state: GraphState) -> GraphState:
        state["documents"] = retriever.invoke(state["question"])
        return state

    prompt = ChatPromptTemplate.from_template("""
You are an assistant. Answer the question using ONLY the context below.
If you don't know, say "I don't have enough information."

Context:
{context}

Question: {question}
""")

    rag_chain = prompt | llm | StrOutputParser()
    @traceable(name = "answer")
    def answer(state: GraphState) -> GraphState:
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        state["answer"] = rag_chain.invoke({"context": context, "question": state["question"]})
        return state

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


# --- Streamlit UI ---
# want to add file uploader in streamlit to upload pdf files and then use those files for RAG chat
paths = []
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(f"./docs/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
    build_app().clear_cache()  # Rebuild the app to include the new documents
    st.success("Files uploaded successfully! You can now ask questions based on the uploaded documents.")
st.title("RAG Chat with LangGraph")

app = build_app()


@traceable(name="rag_pipeline")
def run_query(app, question: str) -> str:
    result = app.invoke({"question": question})
    return result["answer"]

# --- Streamlit UI ---
st.title("RAG Chat with LangGraph")
app = build_app()
question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = run_query(app, question)   # traced ✅
        st.write(answer)
# question = st.text_input("Ask a question:")

# if st.button("Get Answer"):       
#     if question.strip() == "":
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Thinking..."):
#             result = app.invoke({"question": question})
#         st.write(result["answer"])