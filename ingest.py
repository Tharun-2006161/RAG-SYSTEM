"""
Document Ingestion Pipeline
Reads .txt files from the docs/ folder, splits them into chunks,
and stores them in a persistent ChromaDB vector database.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv

# ── Step 2: Initialize ChromaDB ──────────────────────────────────────────────

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)


# ── Step 3: Load and split the documents ─────────────────────────────────────

def load_and_split_documents(docs_folder: str) -> list:
    """Load all .txt files from a folder and split into chunks."""
    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    docs_path = Path(docs_folder).resolve()

    for file_path in sorted(docs_path.glob("*.txt")):
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"  Loaded {len(chunks)} chunks from {file_path.name}")

    return all_chunks


# ── Step 4: Store chunks in ChromaDB ─────────────────────────────────────────

def ingest_to_chromadb(chunks: list):
    """Store document chunks in ChromaDB."""
    documents = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": chunk.metadata.get("source", "unknown")}
        for chunk in chunks
    ]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"  Stored {len(documents)} chunks in ChromaDB.")


# ── Step 5: Main runner ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting document ingestion...")
    chunks = load_and_split_documents("docs")

    if not chunks:
        print("No .txt files found in docs/ folder.")
    else:
        ingest_to_chromadb(chunks)
        print(f"\nDone! Total chunks stored: {len(chunks)}")
        print("Your knowledge base is ready.")
