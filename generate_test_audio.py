"""Generate synthetic German test audio from audio_samples.txt with OpenAI TTS.

Test-data generation only; standalone on purpose (does not use LLMClient).
Each line of audio_samples.txt is "<file_stem> | <German text>" -> audio/<file_stem>.wav.
Existing files are skipped unless --force is given. Reads OPENAI_API_KEY from the environment.

Usage:  uv run python generate_test_audio.py [--force]
"""

import argparse
import wave
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SAMPLES_PATH = Path("audio_samples.txt")
AUDIO_DIR = Path("audio")

TTS_MODEL = "gpt-4o-mini-tts-2025-12-15"  # pinned snapshot for reproducible test audio
TTS_VOICE = "marin"
TTS_INSTRUCTIONS = "Sprich natürliches Hochdeutsch in ruhigem Tempo, wie ein Kunde am Telefon."


def read_samples():
    samples = []
    for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            stem, text = (part.strip() for part in line.split("|", 1))
            samples.append((stem, text))
    return samples


def wav_duration(path):
    """Duration in seconds from the WAV header's format fields and the actual data size.

    Streamed WAV output may carry a placeholder frame count, so we derive the
    frame count from the file size instead of trusting getnframes().
    """
    with wave.open(str(path)) as w:
        bytes_per_second = w.getframerate() * w.getnchannels() * w.getsampwidth()
    return (path.stat().st_size - 44) / bytes_per_second  # 44 = standard WAV header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()
    AUDIO_DIR.mkdir(exist_ok=True)

    print(f"TTS model: {TTS_MODEL}, voice: {TTS_VOICE}, format: wav")
    for stem, text in read_samples():
        path = AUDIO_DIR / f"{stem}.wav"
        if path.exists() and not args.force:
            status = "skipped"
        else:
            with client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=text,
                instructions=TTS_INSTRUCTIONS,
                response_format="wav",
            ) as response:
                response.stream_to_file(path)
            status = "created"
        print(f"{stem:<20} {str(path):<32} {status:<8} {wav_duration(path):.2f}s")


if __name__ == "__main__":
    main()
