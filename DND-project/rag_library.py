#!/usr/bin/env python3
"""
Configurable RAG (Retrieval-Augmented Generation) Library
Using ChromaDB for vector storage, chunking, and Ollama for both embeddings and LLM generation.
"""

import os
import glob
from typing import List, Dict, Any, Optional

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
import hashlib
import json

class RAGConfig:
    """Configuration class for the RAG library."""
    
    def __init__(
        self,
        data_dir: str = "data",
        file_extension: str = "txt",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2:latest",
        instruction: str = "You are a helpful assistant.",
        collection_name: str = "default_collection",
        persistent: bool = False,
        db_path: Optional[str] = None,
        context_limit: int = 200,
        n_results: int = 3
    ):
        self.data_dir = data_dir
        self.file_extension = file_extension
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.instruction = instruction
        self.collection_name = collection_name
        self.persistent = persistent
        self.db_path = db_path
        self.context_limit = context_limit
        self.n_results = n_results


class OllamaEmbeddingFunction:
    """Custom embedding function that uses Ollama for embeddings."""
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Ollama."""
        embeddings = ollama.embed(model=self.model_name, input=input)
        return embeddings.embeddings


def load_documents(config: RAGConfig) -> Dict[str, str]:
    """
    Load text documents from the specified directory.
    """
    documents = {}
    for file_path in glob.glob(os.path.join(config.data_dir, f"*.{config.file_extension}")):
        with open(file_path, 'r') as file:
            content = file.read()
            documents[os.path.basename(file_path)] = content
    
    return documents


def chunk_documents(config: RAGConfig, documents: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Split documents into smaller chunks for embedding.
    """
    chunked_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len
    )
    
    for doc_name, content in documents.items():
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                "id": f"{doc_name}_chunk_{i}",
                "text": chunk,
                "metadata": {"source": doc_name, "chunk": i}
            })
    
    return chunked_documents


def compute_file_hash(file_path):
    """Compute a hash of the file content to detect changes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def compute_config_hash(config):
    """Compute a hash based on config parameters to detect changes."""
    config_dict = {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_model": config.embedding_model,
        "llm_model": config.llm_model,
        "n_results": config.n_results
    }
    return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()

def setup_chroma_db(config, chunks):
    """
    Initialize the ChromaDB collection, resetting data if configuration or embedding dimension changes.
    This version avoids re-embedding chunks that haven't changed.
    """
    if config.persistent:
        client = chromadb.PersistentClient(path=config.db_path)
    else:
        client = chromadb.Client()

    embedding_function = OllamaEmbeddingFunction(model_name=config.embedding_model)

    try:
        # Retrieve the existing collection with the correct embedding function.
        collection = client.get_collection(config.collection_name, embedding_function=embedding_function)
        existing_metadata = collection.get()["metadatas"]
        
        # Get stored embedding dimension from the first chunk in the collection (if any)
        stored_dim = existing_metadata[0].get("embedding_dim") if existing_metadata else None
        test_embedding = ollama.embed(model=config.embedding_model, input=["test"]).embeddings[0]
        current_dim = len(test_embedding)
        
        print(f"🧠 Expected embedding dimension: {stored_dim}, Current model dimension: {current_dim}")
        
        # If the stored dimension does not match, reset the collection.
        if stored_dim != current_dim:
            print(f"⚠️ Embedding dimension changed ({stored_dim} → {current_dim}). Resetting collection...")
            client.delete_collection(config.collection_name)
            collection = client.create_collection(
                name=config.collection_name,
                embedding_function=embedding_function
            )
            stored_dim = current_dim
            reembed_all = True
        else:
            reembed_all = False

    except Exception as e:
        print(f"Creating new collection: {config.collection_name} because: {e}")
        collection = client.create_collection(
            name=config.collection_name,
            embedding_function=embedding_function
        )
        test_embedding = ollama.embed(model=config.embedding_model, input=["test"]).embeddings[0]
        stored_dim = len(test_embedding)
        reembed_all = True

    # Only re-embed chunks if necessary (e.g., first run or detected changes)
    if reembed_all:
        for chunk in chunks:
            chunk_embedding = ollama.embed(model=config.embedding_model, input=[chunk["text"]]).embeddings[0]
            chunk_dim = len(chunk_embedding)
            print(f"🔍 Chunk ID {chunk['id']} has embedding dimension: {chunk_dim}")
            if chunk_dim != stored_dim:
                raise ValueError(f"❌ Mismatched embedding dimension for chunk {chunk['id']}: {chunk_dim} vs expected {stored_dim}")
            chunk["metadata"]["embedding_dim"] = stored_dim

        collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks]
        )
        print(f"✅ Added {len(chunks)} chunks to ChromaDB.")
    else:
        print("✅ Loaded existing collection; no re-embedding necessary.")

    return collection


def retrieve_context(config: RAGConfig, collection: chromadb.Collection, query: str) -> List[str]:
    """
    Retrieve relevant context from ChromaDB based on the query.
    """
    query_embedding = ollama.embed(
        model=config.embedding_model,
        input=[query]
    ).embeddings[0]
    
    results = collection.query(query_embeddings=[query_embedding], n_results=config.n_results)
    
    contexts = [
        metadata["source"] + " - Chunk " + str(metadata["chunk"]) + ": " + text
        for metadata_list, text_list in zip(results["metadatas"], results["documents"])
        for metadata, text in zip(metadata_list, text_list)
    ]
    
    return contexts


def generate_response(config: RAGConfig, query: str, contexts: List[str]) -> str:
    """
    Generate a response using an LLM with the retrieved context.
    """
    context_text = "\n\n".join(contexts)
    prompt = f"""
    {config.instruction}
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer:"""
    
    response = ollama.generate(
        model=config.llm_model,
        prompt=prompt,
    )
    
    return response["response"]


def display_results(config: RAGConfig, query: str, contexts: List[str], response: str) -> None:
    """
    Display the results in a formatted way.
    """
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    print("\nCONTEXT USED:")
    print("-"*80)
    for i, context in enumerate(contexts, 1):
        print(f"Context {i}:")
        print(context[:config.context_limit] + "..." if len(context) > config.context_limit else context)
        print()
    
    print("\nGENERATED RESPONSE:")
    print("-"*80)
    print(response)
    print("="*80 + "\n")
