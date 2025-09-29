# app/rag_backend.py

import os
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# ✅ HuggingFace Embeddings (free & local)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Setup ChromaDB vectorstore
vectorstore = Chroma(
    collection_name="my_corpus",
    embedding_function=embeddings,
    persist_directory="./chroma_store"  # directory where embeddings are saved
)

# ✅ Retriever (for semantic search)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def ingest_file(file_path: str):
    """Load and index a text file into Chroma vector DB."""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    vectorstore.add_documents(docs)
    return f"Ingested {len(docs)} chunks from {file_path}"


def generate_answer(query: str, context: str) -> str:
    """Use Gemini model to generate an answer from context."""
    prompt = f"""
    You are an assistant. Answer the question based on the context below.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    # ✅ Updated Gemini model (choose pro for quality, flash for speed)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)

    return response.text if response and hasattr(response, "text") else "⚠️ No response generated."


def query_rag(query: str) -> str:
    """Search ChromaDB via retriever + Ask Gemini"""
    try:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found."
        return generate_answer(query, context)
    except Exception as e:
        return f"⚠️ Error while querying RAG: {str(e)}"
