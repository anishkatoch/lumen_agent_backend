import asyncio
import os
from openai import AsyncOpenAI

_openai: AsyncOpenAI | None = None

LLM_MODEL = "gpt-4o"
LLM_TIMEOUT = 10  # seconds


def _get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai


async def generate_response(
    user_message: str,
    session_history: list[dict],
    rag_context: str,
    long_term_memory: str | None,
) -> str:
    """
    Call LLM with full context.
    Returns the assistant reply text.
    """
    client = _get_openai()

    system_parts = [
        "You are a helpful voice AI assistant. "
        "Respond concisely — your answers will be spoken aloud. "
        "Avoid markdown, bullet points, or formatting. "
        "Speak naturally as if in a conversation.\n"
        "Follow these rules:\n"
        "1. For questions about the current conversation, the user, or general chat — answer naturally using the conversation history.\n"
        "2. For questions about specific topics, facts, or information — ONLY use the knowledge base context provided below. "
        "If the topic is not in the knowledge base, say: 'I don't have information about that in my knowledge base.' "
        "Do not use your general training knowledge to answer factual questions."
    ]

    if long_term_memory:
        system_parts.append(f"\nWhat you know about this user:\n{long_term_memory}")

    if rag_context:
        system_parts.append(f"\n{rag_context}")
    else:
        system_parts.append("\nNo relevant documents found in the knowledge base for this query.")

    system_prompt = "\n".join(system_parts)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session_history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            ),
            timeout=LLM_TIMEOUT,
        )
        return response.choices[0].message.content.strip()
    except asyncio.TimeoutError:
        return "I'm sorry, I took too long to think. Could you ask again?"
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return "I encountered an error. Please try again."


async def generate_memory_summary(
    old_summary: str | None,
    new_messages: list[dict],
) -> str:
    """Merge old memory summary with new conversation into a concise new summary."""
    client = _get_openai()

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in new_messages
    )

    prompt_parts = ["Summarize what you now know about this user based on:"]
    if old_summary:
        prompt_parts.append(f"Previous summary:\n{old_summary}")
    prompt_parts.append(f"New conversation:\n{history_text}")
    prompt_parts.append(
        "\nWrite a concise paragraph (max 200 words) capturing key facts, "
        "preferences, and past issues. Merge old and new info — no repetition."
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "\n\n".join(prompt_parts)}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Memory summary error: {e}")
        return old_summary or ""
