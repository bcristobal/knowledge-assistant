# embeddings.py

import chromadb
from langchain_ollama import OllamaEmbeddings
from typing import List
from dotenv import load_dotenv  # <-- Move this import up
import os

# --- SOLUTION ---
# Load environment variables at the start of the script.
# This ensures os.getenv() works correctly when function defaults are evaluated.
load_dotenv()
# ----------------

class LocalEmbeddingFunction(chromadb.EmbeddingFunction):
    """ChromaDB-compatible embedding function using Ollama."""

    # The default arguments will now correctly read the loaded environment variables
    def __init__(self, model: str = os.getenv("EMBEDDING_MODEL"), base_url: str = os.getenv("OLLAMA_HOST")):
        # Add a check to provide a more helpful error message
        if not model:
            raise ValueError("Embedding model name not found. Please set EMBEDDING_MODEL in your .env file.")
        
        self.embeddings = OllamaEmbeddings(model=model, base_url=base_url)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(input)

def create_embedding_function() -> LocalEmbeddingFunction:
    """
    Factory function to create a LocalEmbeddingFunction instance.
    The arguments are now handled by the class's __init__ method,
    which reads from the already-loaded environment.
    """
    # No need to pass arguments here, as the __init__ will use the defaults from the environment.
    # Also, no need to call load_dotenv() here anymore.
    return LocalEmbeddingFunction()