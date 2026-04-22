import asyncio
import json
import os

import httpx
import jwt
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr
from typing import Optional

load_dotenv()

app = FastAPI(title="Voice Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lumen-agent-phi.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=".*",
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

security = HTTPBearer()

# ─── JWT Verification (supports both HS256 and ES256) ────────────────
_jwks_cache: Optional[dict] = None

async def _get_jwks() -> dict:
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
        _jwks_cache = resp.json()
    return _jwks_cache

async def decode_supabase_token(token: str) -> dict:
    header = jwt.get_unverified_header(token)
    alg = header.get("alg", "HS256")
    kid = header.get("kid")

    if alg != "HS256" and kid:
        jwks = await _get_jwks()
        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == kid:
                if alg.startswith("ES"):
                    public_key = ECAlgorithm.from_jwk(json.dumps(key_data))
                else:
                    public_key = RSAAlgorithm.from_jwk(json.dumps(key_data))
                return jwt.decode(token, public_key, algorithms=[alg], options={"verify_aud": False})
        raise jwt.InvalidTokenError("No matching key found in JWKS")

    return jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})


# ─── Pydantic Models ────────────────────────────────────────────────

class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    user: Optional[dict] = None
    message: Optional[str] = None

class RefreshRequest(BaseModel):
    refresh_token: str

class WebhookPayload(BaseModel):
    name: str  # file name from Supabase Storage webhook


# ─── JWT Dependency ──────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    token = credentials.credentials
    try:
        return await decode_supabase_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


# ─── Auth Routes ─────────────────────────────────────────────────────

@app.post("/auth/signup", response_model=AuthResponse)
async def sign_up(body: SignUpRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/signup",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": body.email, "password": body.password, "data": {"full_name": body.full_name}},
        )
    data = resp.json()
    if resp.status_code != 200 or "error" in data:
        msg = data.get("error_description") or data.get("msg") or "Signup failed"
        raise HTTPException(status_code=400, detail=msg)
    # Email confirmation required — Supabase returns user but no session tokens
    if "access_token" not in data:
        return AuthResponse(message="confirmation_required")
    return AuthResponse(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        user={
            "id": data["user"]["id"],
            "email": data["user"]["email"],
            "full_name": data["user"].get("user_metadata", {}).get("full_name"),
        },
    )


@app.post("/auth/signin", response_model=AuthResponse)
async def sign_in(body: SignInRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": body.email, "password": body.password},
        )
    data = resp.json()
    if resp.status_code != 200 or "error" in data:
        msg = data.get("error_description") or data.get("msg") or "Invalid credentials"
        raise HTTPException(status_code=401, detail=msg)
    return AuthResponse(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        user={
            "id": data["user"]["id"],
            "email": data["user"]["email"],
            "full_name": data["user"].get("user_metadata", {}).get("full_name"),
        },
    )


@app.post("/auth/refresh")
async def refresh_token(body: RefreshRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"refresh_token": body.refresh_token},
        )
    data = resp.json()
    if resp.status_code != 200 or "error" in data:
        raise HTTPException(status_code=401, detail="Refresh token invalid or expired")
    return {"access_token": data["access_token"], "refresh_token": data["refresh_token"]}


@app.post("/auth/signout")
async def sign_out(current_user: dict = Depends(get_current_user)):
    return {"message": "Signed out successfully"}


# ─── Protected REST Routes ────────────────────────────────────────────

@app.get("/api/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user.get("sub"),
        "email": current_user.get("email"),
        "full_name": current_user.get("user_metadata", {}).get("full_name"),
        "role": current_user.get("role", "authenticated"),
    }


@app.get("/api/conversations")
async def list_conversations(current_user: dict = Depends(get_current_user)):
    from conversation_repo import get_conversations
    return get_conversations(current_user["sub"])


@app.get("/api/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
    from message_repo import get_conversation_history
    return get_conversation_history(conversation_id)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/ingest-all-dev")
async def ingest_all_documents_dev():
    """No-auth version for local testing only. Remove before going to production."""
    return await ingest_all_documents_impl()


@app.post("/api/ingest-all")
async def ingest_all_documents(current_user: dict = Depends(get_current_user)):
    return await ingest_all_documents_impl()


async def ingest_all_documents_impl():
    """
    Manually ingest all .md files from the Supabase Storage documents bucket.
    Call this once after uploading files to populate the vector store.
    """
    from doc_processor import ingest_document
    from database import get_supabase

    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")
    db = get_supabase()

    try:
        files = db.storage.from_(bucket).list()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list bucket: {e}")

    md_files = [f["name"] for f in files if f["name"].endswith(".md")]

    if not md_files:
        return {"status": "no .md files found in bucket", "bucket": bucket}

    results = []
    for file_name in md_files:
        try:
            chunks = await ingest_document(file_name)
            results.append({"file": file_name, "chunks": chunks, "status": "ok"})
            print(f"[Ingest] {file_name} → {chunks} chunks")
        except Exception as e:
            results.append({"file": file_name, "status": "error", "error": str(e)})
            print(f"[Ingest] {file_name} → ERROR: {e}")

    return {"ingested": results}


# ─── Webhook: Auto Document Ingestion ────────────────────────────────

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")  # optional — set in Supabase webhook headers

@app.post("/webhook/new-document")
async def new_document_webhook(payload: dict, request: Request):
    """
    Supabase Database Webhook fires on every INSERT to storage.objects.
    Supabase sends payload like:
      { "type": "INSERT", "table": "objects", "schema": "storage",
        "record": { "name": "file.md", "bucket_id": "documents", ... } }
    """
    from doc_processor import ingest_document

    # Optional secret verification
    if WEBHOOK_SECRET:
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {WEBHOOK_SECRET}":
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    # Extract file info from Supabase storage.objects INSERT payload
    record = payload.get("record", {})
    file_name = (
        record.get("name")
        or payload.get("name")
        or payload.get("file_name")
    )
    bucket_id = record.get("bucket_id", "")

    print(f"[Webhook] Received payload: type={payload.get('type')} bucket={bucket_id} file={file_name}")

    if not file_name:
        return {"status": "ignored", "reason": "no file_name in payload"}

    if not file_name.endswith(".md"):
        return {"status": "ignored", "reason": f"not a .md file: {file_name}"}

    # Only process the documents bucket (skip other buckets if webhook is broad)
    expected_bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")
    if bucket_id and bucket_id != expected_bucket:
        return {"status": "ignored", "reason": f"wrong bucket: {bucket_id}"}

    # Run ingestion in background so webhook returns immediately (Supabase has a short timeout)
    asyncio.create_task(ingest_document(file_name))
    print(f"[Webhook] Queued ingestion for: {file_name}")
    return {"status": "processing", "file": file_name}


# ─── WebSocket: Voice Pipeline ────────────────────────────────────────

@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    from websocket_manager import manager, UserSession, AgentState
    from conversation_repo import create_conversation
    from message_repo import save_message, get_session_messages
    from memory_repo import get_user_memory, upsert_user_memory
    from deepgram_service import transcribe_audio
    from elevenlabs_service import text_to_speech_stream
    from rag_service import retrieve_context, format_rag_context
    from llm_service import generate_response, generate_memory_summary
    from vad_service import detect_speech, build_wav_bytes

    # Verify token
    try:
        payload = await decode_supabase_token(token)
        user_id = payload["sub"]
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[WS AUTH FAILED] {type(e).__name__}: {e}")
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Check capacity
    if manager.is_full():
        await websocket.close(code=1008, reason="Server at capacity")
        return

    await websocket.accept()

    # Create conversation + session
    conversation = create_conversation(user_id)
    conversation_id = conversation["id"]

    sensitivity = float(os.getenv("BARGE_IN_SENSITIVITY", "0.2"))
    session = UserSession(
        user_id=user_id,
        websocket=websocket,
        conversation_id=conversation_id,
        sensitivity=sensitivity,
    )
    manager.add_session(session)

    await manager.send_json(websocket, {
        "type": "ready",
        "conversation_id": conversation_id,
    })

    async def run_tts_and_send(text: str):
        """Stream TTS audio chunks back over WebSocket."""
        session.state = AgentState.SPEAKING
        await manager.send_json(websocket, {"type": "state", "state": "SPEAKING"})
        try:
            async for chunk in text_to_speech_stream(text):
                if session.state != AgentState.SPEAKING:
                    break  # barge-in: stop sending
                await manager.send_bytes(websocket, chunk)
        except Exception as e:
            print(f"[WS] TTS stream error: {e}")
        finally:
            await manager.send_json(websocket, {"type": "audio_end"})
            if session.state == AgentState.SPEAKING:
                session.state = AgentState.LISTENING
                await manager.send_json(websocket, {"type": "state", "state": "LISTENING"})

    async def run_pipeline(pcm_bytes: bytes, sr: int = 16000):
        """STT → RAG → LLM → TTS on a complete utterance."""
        try:
            print(f"[Pipeline] Starting — audio={len(pcm_bytes)}B @ {sr}Hz")
            wav_bytes = build_wav_bytes(pcm_bytes, sample_rate=sr)
            print(f"[Pipeline] Sending to Deepgram — wav={len(wav_bytes)}B")
            transcript = await transcribe_audio(wav_bytes)
            print(f"[Pipeline] Deepgram transcript: {repr(transcript)}")

            if not transcript:
                await manager.send_json(websocket, {"type": "state", "state": "LISTENING"})
                session.state = AgentState.LISTENING
                return

            await manager.send_json(websocket, {"type": "transcript", "text": transcript})
            save_message(conversation_id, user_id, "user", transcript)

            rag_chunks = await retrieve_context(transcript)
            print(f"[RAG] Found {len(rag_chunks)} chunks for: {transcript[:60]}")
            rag_context = format_rag_context(rag_chunks)
            long_term_memory = get_user_memory(user_id)
            session_history = get_session_messages(conversation_id)

            reply = await generate_response(
                user_message=transcript,
                session_history=session_history,
                rag_context=rag_context,
                long_term_memory=long_term_memory,
            )

            await manager.send_json(websocket, {"type": "reply", "text": reply})
            save_message(conversation_id, user_id, "assistant", reply)
            session.message_count += 1

            session.tts_task = asyncio.create_task(run_tts_and_send(reply))
            await session.tts_task

            if session.message_count % 10 == 0:
                asyncio.create_task(update_memory(user_id, conversation_id))

        except Exception as e:
            print(f"[WS] Pipeline error: {e}")
            import traceback; traceback.print_exc()
            session.state = AgentState.LISTENING
            await manager.send_json(websocket, {"type": "state", "state": "LISTENING"})

    async def update_memory(uid: str, conv_id: str):
        old_summary = get_user_memory(uid)
        messages = get_session_messages(conv_id)
        new_summary = await generate_memory_summary(old_summary, messages)
        upsert_user_memory(uid, new_summary)

    # Speech accumulation state
    speech_buffer: list[bytes] = []
    silence_chunks = 0
    speech_chunks = 0
    sample_rate = 16000
    SILENCE_LIMIT = 6    # 6 × 250ms = 1.5s silence → end of utterance
    MIN_SPEECH = 2       # need at least 2 chunks of speech before processing

    # Main message loop
    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                audio_bytes = data["bytes"]

                # Barge-in while speaking
                if session.state == AgentState.SPEAKING:
                    has_speech = await detect_speech(audio_bytes, session.sensitivity)
                    if has_speech:
                        if session.tts_task and not session.tts_task.done():
                            session.tts_task.cancel()
                        session.state = AgentState.LISTENING
                        await manager.send_json(websocket, {"type": "stop_audio"})
                        await manager.send_json(websocket, {"type": "state", "state": "LISTENING"})
                        speech_buffer.clear()
                        silence_chunks = 0
                        speech_chunks = 0
                    continue

                if session.state == AgentState.PROCESSING:
                    continue

                # LISTENING: accumulate speech
                if session.state == AgentState.LISTENING:
                    has_speech = await detect_speech(audio_bytes, session.sensitivity)
                    print(f"[VAD] chunk={len(audio_bytes)}B speech={has_speech} speech_chunks={speech_chunks} silence={silence_chunks}")

                    if has_speech:
                        speech_buffer.append(audio_bytes)
                        speech_chunks += 1
                        silence_chunks = 0
                        if speech_chunks == 1:
                            print("[VAD] Speech started")
                            await manager.send_json(websocket, {"type": "state", "state": "PROCESSING"})
                    else:
                        if speech_chunks > 0:
                            silence_chunks += 1
                            speech_buffer.append(audio_bytes)

                            if silence_chunks >= SILENCE_LIMIT:
                                print(f"[VAD] Utterance end — speech={speech_chunks} chunks, launching pipeline")
                                if speech_chunks >= MIN_SPEECH:
                                    full_audio = b"".join(speech_buffer)
                                    session.state = AgentState.PROCESSING
                                    asyncio.create_task(run_pipeline(full_audio, sample_rate))
                                else:
                                    print("[VAD] Too short, discarding")
                                    session.state = AgentState.LISTENING
                                    await manager.send_json(websocket, {"type": "state", "state": "LISTENING"})
                                speech_buffer.clear()
                                silence_chunks = 0
                                speech_chunks = 0

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "set_sensitivity":
                        session.sensitivity = float(msg.get("value", 0.5))
                    elif msg.get("type") == "sample_rate":
                        sample_rate = int(msg.get("value", 16000))
                        print(f"[WS] Client sample rate: {sample_rate}Hz")
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        msg = str(e)
        if "disconnect message" not in msg and "WebSocket" not in msg:
            print(f"[WS] Unexpected error: {e}")
    finally:
        manager.remove_session(user_id)
        asyncio.create_task(update_memory(user_id, conversation_id))
