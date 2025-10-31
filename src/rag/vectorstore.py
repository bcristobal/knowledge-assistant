import chromadb
import yaml
from pathlib import Path
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from typing import Optional
import uuid
import os
import hashlib

from embeddings import create_embedding_function
from loader import parse_pdf_to_md
from chunker import chunking

class ChromaClient:
    """
    ChromaDB client wrapper.
    Collections are strictly managed through the configuration file.
    Dynamic creation of collections is disabled by design.
    """

    def __init__(self, config_path: str = "config/vectorstore.yaml"):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
            if 'vectorstore' not in self.config or 'settings' not in self.config['vectorstore']:
                raise KeyError("Config file must have 'vectorstore.settings' structure.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        settings = self.config['vectorstore']['settings']
        db_path = settings.get('path', './chroma_db')
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.client = self._create_client(db_path)

        self._collections: dict[str, Collection] = {}
        for col_cfg in settings.get('collections', []):
            col = self._get_or_create_collection(
                name=col_cfg['name'],
                metadata=col_cfg.get('metadata', {})
            )
            self._collections[col_cfg['name']] = col

    def _create_client(self, path: str) -> chromadb.Client:
        return chromadb.PersistentClient(path=path)

    def _get_or_create_collection(self, name: str, metadata: dict = None) -> Collection:
        if metadata is None:
            metadata = {"description": f"Collection for {name}"}
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata,
            embedding_function=create_embedding_function()
        )

    def get_collection(self, name: str) -> Optional[Collection]:
        return self._collections.get(name)

    @property
    def collections(self) -> dict[str, Collection]:
        return self._collections

    def query(self, collection_name: str, query_texts: list[str]) -> dict:
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' is not defined in the configuration file.")
        
        k_results = self.config['vectorstore']['settings'].get('top_k_results', 5)
        
        results = collection.query(
            query_texts=query_texts,
            n_results=k_results
        )
        return results
    
    # --- MODIFIED METHOD ---
    def add_document(self, collection_name: str, file_path: str):
        """Adds a document's chunks to a specific collection."""
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' is not defined in the configuration file.")
        
        print(f"--> Parsing document: {file_path}")
        # parse_pdf_to_md returns a LIST of Document objects.
        documents_from_parser = parse_pdf_to_md(file_path)

        # Add a check in case parsing returns nothing.
        if not documents_from_parser:
            print(f"--> Warning: No content parsed from {file_path}. Skipping document.")
            return

        # THE FIX: Extract the text from each Document object and join them into a single string.
        # Each object in the list has a '.text' or '.page_content' attribute. LlamaIndex uses .text.
        full_text_string = "\n\n".join([doc.text for doc in documents_from_parser])
        
        # Generate a stable, unique ID for the document based on its path
        doc_id = hashlib.sha256(full_text_string.encode()).hexdigest()
        
        chunk_settings = self.config['vectorstore']['settings']
        
        # Now, pass the single 'full_text_string' to your excellent chunking function.
        chunks = chunking(
            document=full_text_string, 
            chunk_size=chunk_settings.get('chunk_size', 500),
            chunk_overlap=chunk_settings.get('chunk_overlap', 100)
        )
        
        # Create a list of metadata dictionaries, one for each chunk
        metadatas = [{"source": os.path.basename(file_path)} for _ in chunks]
        
        # Generate unique IDs for each chunk
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        
        print(f"--> Adding {len(chunks)} chunks to collection '{collection_name}'.")
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        print("--> Document added successfully.")

# --- MODIFIED MAIN BLOCK ---
if __name__ == "__main__":
    load_dotenv()
    
    try:
        chroma_client = ChromaClient()
        
        print("Configured collections:", chroma_client.collections.keys())
        
        doc_collection = chroma_client.get_collection("documents")
        print("\n'documents' collection object:", doc_collection)
        
        non_existent_collection = chroma_client.get_collection("non_existent")
        print("\n'non_existent' collection object:", non_existent_collection)

        if doc_collection:
            # CORRECTED: Use the new path inside the 'data' directory
            document_file_path = "data/raw/Prácticas_beñat_listado_tech.pdf"
            
            chroma_client.add_document(
                collection_name="documents",
                file_path=document_file_path # Use the full path variable
            )
            
            # Example query after adding the document
            print("\n--- Performing test query ---")
            query_results = chroma_client.query(
                collection_name="documents",
                query_texts=["what technologies are mentioned?"]
            )
            print(query_results['documents'])


    except Exception as e:
        print(f"\nAn error occurred: {e}")