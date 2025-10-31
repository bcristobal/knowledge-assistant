
# chunker.py
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def chunking(document: str, chunk_size, chunk_overlap):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(document)
    
    if not md_header_splits:
        print("Warning: No content to split.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split the markdown sections into chunks
    chunks = []
    for section in md_header_splits:
        chunks.extend(text_splitter.split_text(section.page_content))
    
    return chunks
