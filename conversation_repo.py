from database import get_supabase


def create_conversation(user_id: str, title: str = "Voice session") -> dict:
    db = get_supabase()
    result = (
        db.table("conversations")
        .insert({"user_id": user_id, "title": title})
        .execute()
    )
    if not result.data:
        raise RuntimeError("Failed to create conversation")
    return result.data[0]


def get_conversations(user_id: str) -> list[dict]:
    db = get_supabase()
    result = (
        db.table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data
