import asyncio
import os
from openai import AsyncOpenAI
from document_repo import vector_search

_openai: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai


async def embed_text(text: str) -> list[float]:
    """Embed a string using OpenAI text-embedding-3-small."""
    client = _get_openai()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


async def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    """Embed query and find top_k matching document chunks."""
    try:
        vector = await embed_text(query)
        chunks = await asyncio.to_thread(vector_search, vector, top_k)
        print(f"[RAG] vector_search returned {len(chunks)} chunks")
        return chunks
    except Exception as e:
        import traceback
        print(f"[RAG] retrieve_context error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return []


def format_rag_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a prompt-ready string."""
    if not chunks:
        return ""
    parts = ["Relevant knowledge base context:"]
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("section_title", "")
        text = chunk.get("chunk_text", "")
        parts.append(f"\n[{i}] {title}\n{text}")
    return "\n".join(parts)
