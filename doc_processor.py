"""
Document ingestion pipeline:
  1. Download .md file from Supabase Storage
  2. Parse + chunk by heading (md_extractor)
  3. Embed each chunk (OpenAI)
  4. Save to pgvector (document_repo)
"""
import os
from database import get_supabase
from md_extractor import extract_chunks
from rag_service import embed_text
from document_repo import save_chunk, delete_chunks_by_file


async def ingest_document(file_name: str) -> int:
    """
    Download, process, and store a .md file from Supabase Storage.
    Returns the number of chunks saved.
    """
    db = get_supabase()
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")

    # Download file bytes
    response = db.storage.from_(bucket).download(file_name)
    if not response:
        raise ValueError(f"File not found in storage: {file_name}")

    markdown_text = response.decode("utf-8")

    # Remove old chunks for this file (re-ingestion support)
    delete_chunks_by_file(file_name)

    # Parse into chunks
    chunks = extract_chunks(markdown_text)
    if not chunks:
        raise ValueError(f"No content extracted from {file_name}")

    saved = 0
    for chunk in chunks:
        vector = await embed_text(chunk["chunk_text"])
        save_chunk(
            file_name=file_name,
            section_title=chunk["section_title"],
            chunk_text=chunk["chunk_text"],
            vector=vector,
        )
        saved += 1

    print(f"[DocProcessor] Ingested {saved} chunks from {file_name}")
    return saved
