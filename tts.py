"""Text-to-speech for the bot's final answer via the OpenAI TTS API.

Model and voice match generate_test_audio.py (kept as a small duplicate so the
pipeline does not depend on a test-data script); the instruction differs on purpose:
a brisk customer-service voice here vs. a slower customer voice for test input. Output is WAV.
The synthesized text is never printed or logged here.
"""

import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # optional .env support; OPENAI_API_KEY may also come from the shell

TTS_MODEL = "gpt-4o-mini-tts-2025-12-15"
TTS_VOICE = "marin"
TTS_INSTRUCTIONS = "Sprich natürliches Hochdeutsch, klar, freundlich und in einem zügigen, professionellen Kundenservice-Tempo."

_client = None


def wav_duration(path):
    """Duration from the WAV format fields and the file size.

    OpenAI's streamed WAV header carries a placeholder frame count, so
    getnframes() cannot be trusted (same logic as generate_test_audio.py).
    """
    with wave.open(str(path)) as w:
        bytes_per_second = w.getframerate() * w.getnchannels() * w.getsampwidth()
    return (path.stat().st_size - 44) / bytes_per_second  # 44 = standard WAV header


def synthesize(text: str, output_path: str) -> dict:
    global _client
    if _client is None:
        _client = OpenAI()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    with _client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions=TTS_INSTRUCTIONS,
        response_format="wav",
    ) as response:
        response.stream_to_file(path)
    elapsed = time.perf_counter() - start

    return {
        "output_path": str(path),
        "elapsed_seconds": elapsed,
        "duration_seconds": wav_duration(path),
    }
