import asyncio
import os
from deepgram import DeepgramClient, PrerecordedOptions


async def transcribe_audio(audio_bytes: bytes) -> str | None:
    """
    Send audio bytes to Deepgram and return transcript text.
    Returns None if no speech detected or on timeout.
    """
    timeout_seconds = int(os.getenv("TRANSCRIPT_TIMEOUT_SECONDS", "8"))
    api_key = os.getenv("DEEPGRAM_API_KEY")

    client = DeepgramClient(api_key=api_key)

    options = PrerecordedOptions(
        model="nova-2",
        language="en-US",
        smart_format=True,
        punctuate=True,
    )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.listen.prerecorded.v("1").transcribe_file,
                {"buffer": audio_bytes, "mimetype": "audio/wav"},
                options,
            ),
            timeout=timeout_seconds,
        )
        results = response.results
        if not results or not results.channels:
            return None
        alternatives = results.channels[0].alternatives
        if not alternatives:
            return None
        transcript = alternatives[0].transcript.strip()
        return transcript if transcript else None

    except asyncio.TimeoutError:
        return None
    except Exception as e:
        print(f"[Deepgram] Error: {e}")
        return None
