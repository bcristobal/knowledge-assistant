# loader.py

import os
import uuid
import hashlib

from llama_cloud_services import LlamaParse
from dotenv import load_dotenv
import pymupdf

def load_document(file_path: str):
    if ".pdf" in file_path:
        return _load_pdf(file_path=file_path)

def _load_pdf(file_path: str):
    doc = pymupdf.open(file_path)
    return doc

def get_doc_id(file_path: str):
    doc = load_document(file_path=file_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def parse_pdf_to_md(file_path: str): # <-- Parameter is the full path
    """
    Parses a PDF file using LlamaCloud and saves the result as a Markdown file.
    """
    print("--> Initializing LlamaParse...")
    parser = LlamaParse(
        verbose=True,
    )

    print(f"--> Sending '{file_path}' to LlamaCloud for parsing. This may take a moment...")
    job_result = parser.parse(file_path)

    print("--> Extracting markdown document from the job result...")
    document = job_result.get_markdown_documents()

    if not document:
        print("--> WARNING: No document was returned from the parser. The output file will not be created.")
        return []
    
    # --- FIXED OUTPUT PATH LOGIC ---
    # Get just the filename from the full path
    file_name_only = os.path.basename(file_path)
    output_file_path = f"data/processed/{file_name_only.replace('.pdf', '')}.md"
    
    print(f"--> Writing {len(document)} document to '{output_file_path}'...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as f:
        for doc in document:
            f.write(doc.text_resource.text)
            f.write("\n\n---\n\n")

    print(f"--> Successfully parsed and wrote content to '{output_file_path}'")
    return document


if __name__ == "__main__":
    load_dotenv()
    
    if "LLAMA_CLOUD_API_KEY" not in os.environ:
        raise ValueError("API key not found in .env file.")

    job_id = uuid.uuid4()
    # The caller provides the full, correct path
    pdf_file_to_parse = "data/raw/Prácticas_beñat_listado_tech.pdf"
    
    print(f"Started parsing the file '{pdf_file_to_parse}' under job_id {job_id}")

    if not os.path.exists(pdf_file_to_parse):
        raise FileNotFoundError(f"The input PDF file was not found at: {pdf_file_to_parse}")

    resultant_docs = parse_pdf_to_md(pdf_file_to_parse) # <-- Pass the full path
    print(f"\n--> Main script finished. Total documents processed: {len(resultant_docs)}")