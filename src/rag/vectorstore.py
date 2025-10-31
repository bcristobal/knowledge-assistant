import chromadb
import yaml
import logging
from pathlib import Path
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from typing import Optional, List
import os
import hashlib

from embeddings import create_embedding_function
from loader import parse_pdf_to_md, get_doc_id
from chunker import chunking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            
        logger.info(f"Initialized ChromaClient with {len(self._collections)} collections")

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
        """Query a collection with the given texts."""
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' is not defined in the configuration file.")
        
        k_results = self.config['vectorstore']['settings'].get('top_k_results', 5)
        
        logger.info(f"Querying collection '{collection_name}' with {len(query_texts)} queries")
        results = collection.query(
            query_texts=query_texts,
            n_results=k_results
        )
        return results
    
    def _extract_text_from_documents(self, documents: List) -> str:
        """
        Safely extract text from parsed documents.
        Handles multiple possible document structures from LlamaParse.
        """
        if not documents:
            logger.warning("No documents provided for text extraction")
            return ""
        
        text_parts = []
        
        for i, doc in enumerate(documents):
            try:
                # Try different possible attributes
                if hasattr(doc, 'text'):
                    text_parts.append(doc.text)
                elif hasattr(doc, 'text_resource') and hasattr(doc.text_resource, 'text'):
                    text_parts.append(doc.text_resource.text)
                elif hasattr(doc, 'page_content'):
                    text_parts.append(doc.page_content)
                elif isinstance(doc, str):
                    text_parts.append(doc)
                else:
                    logger.warning(f"Document {i} has unrecognized structure: {type(doc)}")
            except Exception as e:
                logger.error(f"Error extracting text from document {i}: {e}")
                continue
        
        if not text_parts:
            logger.error("Could not extract text from any documents")
            return ""
        
        return "\n\n".join(text_parts)
    
    def _check_document_exists(self, collection: Collection, doc_id: str) -> bool:
        """Check if a document with the given ID already exists in the collection."""
        try:
            existing = collection.get(ids=[doc_id])
            return len(existing['ids']) > 0
        except Exception as e:
            logger.warning(f"Error checking document existence: {e}")
            return False
    
    def _get_existing_chunk_ids(self, collection: Collection, doc_id: str) -> set:
        """Get all chunk IDs for a given document."""
        try:
            # Query for chunks that start with the document ID
            all_items = collection.get()
            chunk_ids = {id for id in all_items['ids'] if id.startswith(f"{doc_id}_")}
            return chunk_ids
        except Exception as e:
            logger.warning(f"Error getting existing chunks: {e}")
            return set()

    def add_document(
        self, 
        collection_name: str, 
        file_path: str,
        force_reprocess: bool = False
    ) -> dict:
        """
        Adds a document's chunks to a specific collection.
        
        Args:
            collection_name: Name of the collection to add to
            file_path: Path to the PDF file
            force_reprocess: If True, reprocess even if document exists
            
        Returns:
            dict with status information
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' is not defined in the configuration file.")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")


        # Generate stable document ID
        doc_id = get_doc_id(file_path=file_path)
        
        # Check if document already exists
        existing_chunks = self._get_existing_chunk_ids(collection, doc_id)
        if existing_chunks and not force_reprocess:
            logger.info(f"Document already exists with {len(existing_chunks)} chunks. Use force_reprocess=True to reprocess.")
            return {
                "status": "already_exists",
                "doc_id": doc_id,
                "existing_chunks": len(existing_chunks),
                "file": file_path
            }
        
        # Delete existing chunks if reprocessing
        if existing_chunks and force_reprocess:
            logger.info(f"Deleting {len(existing_chunks)} existing chunks for reprocessing")
            try:
                collection.delete(ids=list(existing_chunks))
            except Exception as e:
                logger.error(f"Failed to delete existing chunks: {e}")
        
        logger.info(f"Processing document: {file_path}")
        
        # Parse the document
        try:
            documents_from_parser = parse_pdf_to_md(file_path)
        except Exception as e:
            logger.error(f"Failed to parse document: {e}")
            raise
        
        if not documents_from_parser:
            logger.warning(f"No content parsed from {file_path}. Skipping document.")
            return {
                "status": "skipped",
                "reason": "no_content",
                "file": file_path
            }
        
        # Extract text using the safe method
        full_text_string = self._extract_text_from_documents(documents_from_parser)
        
        if not full_text_string or not full_text_string.strip():
            logger.warning(f"Extracted text is empty for {file_path}")
            return {
                "status": "skipped",
                "reason": "empty_content",
                "file": file_path
            }
        
        
        # Get chunking settings
        chunk_settings = self.config['vectorstore']['settings']
        
        # Chunk the document
        try:
            chunks = chunking(
                document=full_text_string, 
                chunk_size=chunk_settings.get('chunk_size', 500),
                chunk_overlap=chunk_settings.get('chunk_overlap', 50)
            )
        except Exception as e:
            logger.error(f"Failed to chunk document: {e}")
            raise
        
        if not chunks:
            logger.warning(f"No chunks generated from {file_path}")
            return {
                "status": "skipped",
                "reason": "no_chunks",
                "file": file_path
            }
        
        # Prepare metadata for each chunk
        file_name = os.path.basename(file_path)
        metadatas = [
            {
                "source": file_name,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            } 
            for i in range(len(chunks))
        ]
        
        # Generate unique IDs for each chunk
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Add chunks to collection
        logger.info(f"Adding {len(chunks)} chunks to collection '{collection_name}'")
        try:
            collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            logger.info(f"Successfully added document with {len(chunks)} chunks")
            return {
                "status": "success",
                "doc_id": doc_id,
                "chunks_added": len(chunks),
                "file": file_path
            }
        except Exception as e:
            logger.error(f"Failed to add chunks to collection: {e}")
            raise


def process_directory(
    chroma_client: ChromaClient,
    collection_name: str,
    directory_path: str,
    pattern: str = "*.pdf",
    force_reprocess: bool = False
) -> dict:
    """
    Process all files matching pattern in a directory.
    
    Args:
        chroma_client: ChromaClient instance
        collection_name: Target collection name
        directory_path: Directory to scan
        pattern: File pattern to match (default: "*.pdf")
        force_reprocess: If True, reprocess existing documents
        
    Returns:
        dict with processing results
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    files = list(directory.glob(pattern))
    logger.info(f"Found {len(files)} files matching pattern '{pattern}'")
    
    results = {
        "total_files": len(files),
        "processed": [],
        "failed": [],
        "skipped": []
    }
    
    for file_path in files:
        try:
            result = chroma_client.add_document(
                collection_name=collection_name,
                file_path=str(file_path),
                force_reprocess=force_reprocess
            )
            
            if result["status"] == "success":
                results["processed"].append(result)
            elif result["status"] == "skipped":
                results["skipped"].append(result)
            else:
                results["failed"].append(result)
                
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            results["failed"].append({
                "file": str(file_path),
                "error": str(e)
            })
    
    return results


if __name__ == "__main__":
    load_dotenv()
    
    try:
        chroma_client = ChromaClient()
        
        logger.info(f"Configured collections: {list(chroma_client.collections.keys())}")
        
        doc_collection = chroma_client.get_collection("documents")
        if not doc_collection:
            logger.error("'documents' collection not found")
            exit(1)
        
        # Process a single document
        document_file_path = "data/raw/Prácticas_beñat_listado_tech.pdf"
        
        if os.path.exists(document_file_path):
            result = chroma_client.add_document(
                collection_name="documents",
                file_path=document_file_path,
                force_reprocess=False
            )
            logger.info(f"Processing result: {result}")
            
            # Example query after adding the document
            logger.info("--- Performing test query ---")
            query_results = chroma_client.query(
                collection_name="documents",
                query_texts=["what technologies are mentioned?"]
            )
            
            if query_results['documents']:
                logger.info(f"Found {len(query_results['documents'][0])} relevant chunks")
                for i, doc in enumerate(query_results['documents'][0][:3]):
                    logger.info(f"Chunk {i+1}: {doc[:200]}...")
            else:
                logger.warning("No results found")
        else:
            logger.warning(f"File not found: {document_file_path}")
            
            # Alternative: Process entire directory
            logger.info("--- Processing directory instead ---")
            results = process_directory(
                chroma_client=chroma_client,
                collection_name="documents",
                directory_path="data/raw",
                pattern="*.pdf",
                force_reprocess=False
            )
            logger.info(f"Directory processing results: {results}")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")