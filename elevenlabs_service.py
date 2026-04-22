import asyncio
import os
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings


async def text_to_speech_stream(text: str):
    """
    Async generator that yields audio chunks from ElevenLabs TTS.
    Uses streaming for low latency.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # default: George

    client = ElevenLabs(api_key=api_key)

    settings = VoiceSettings(
        stability=0.5,
        similarity_boost=0.75,
        style=0.0,
        use_speaker_boost=True,
    )

    print(f"[ElevenLabs] Requesting TTS for: {text[:60]}")
    try:
        audio_stream = await asyncio.to_thread(
            client.text_to_speech.convert_as_stream,
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings=settings,
            output_format="mp3_44100_128",
        )
        chunk_count = 0
        for chunk in audio_stream:
            if chunk:
                chunk_count += 1
                yield chunk
        print(f"[ElevenLabs] Done — sent {chunk_count} chunks")
    except Exception as e:
        import traceback
        print(f"[ElevenLabs] TTS error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return


async def text_to_speech_bytes(text: str) -> bytes | None:
    """Collect full audio as bytes (fallback for non-streaming use)."""
    chunks = []
    async for chunk in text_to_speech_stream(text):
        chunks.append(chunk)
    return b"".join(chunks) if chunks else None
