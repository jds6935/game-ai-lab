# Set up RAG configuration
from rag_library import RAG

rag = RAG(
    data_dir="rag-documents",  # Directory containing text files
    file_extension="txt",      # File type to process
    chunk_size=750,            # Number of characters per chunk
    chunk_overlap=250,          # Overlapping characters between chunks
    embedding_model="nomic-embed-text",  # Embedding model for vector storage
    llm_model="llama3.2:latest", # Language model for response generation
    instruction="You are an assistant that gives me very straight forward answers on DnD",  # LLM prompt style
    collection_name="my_rag_collection",  # ChromaDB collection name
    persistent=True,            # Whether to persist ChromaDB storage
    db_path="./chroma_db",      # Path for persistent storage
    context_limit=300,          # Max characters to display in context
    n_results=12                # Number of relevant documents to retrieve
)

rag.start()

# # Step 2: Run a query
# query = "A 10th-level Wizard is concentrating on the spell Wall of Force, which they cast last turn. On their current turn, they cast Misty Step as a bonus action and then attempt to cast Disintegrate on the Wall of Force. What happens, and why?"
# response = rag.run_query(query)

# print(response)