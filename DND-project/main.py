import ollama

# Set up RAG configuration
from rag_library import (
    RAGConfig,
    load_documents,
    chunk_documents,
    setup_chroma_db,
    retrieve_context,
    generate_response,
    display_results
)

# Step 1: Configure the RAG settings
config = RAGConfig(
    data_dir="rag-documents",  # Directory containing text files
    file_extension="txt",      # File type to process
    chunk_size=500,            # Number of characters per chunk
    chunk_overlap=150,          # Overlapping characters between chunks
    embedding_model="nomic-embed-text",  # Embedding model for vector storage
    llm_model="llama3.2:latest", # Language model for response generation
    instruction="You are a helpful assistant for research.",  # LLM prompt style
    collection_name="my_rag_collection",  # ChromaDB collection name
    persistent=True,            # Whether to persist ChromaDB storage
    db_path="./chroma_db",      # Path for persistent storage
    context_limit=300,          # Max characters to display in context
    n_results=6                 # Number of relevant documents to retrieve
)

test_embedding = ollama.embed(model=config.embedding_model, input=["test"]).embeddings[0]
print(f"🚀 Using embedding model: {config.embedding_model} with dimension: {len(test_embedding)}")

# Step 2: Load and chunk documents
documents = load_documents(config)
chunks = chunk_documents(config, documents)
collection = setup_chroma_db(config, chunks)

# Step 3: Set up ChromaDB
collection = setup_chroma_db(config, chunks)

# Step 4: Run a query
query = "What is a wizard?"
contexts = retrieve_context(config, collection, query)

# Step 5: Generate a response
response = generate_response(config, query, contexts)

# Step 6: Display results
display_results(config, query, contexts, response)
