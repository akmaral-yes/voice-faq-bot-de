"""Speech-to-text with faster-whisper.

Prototype configuration: model "small", CPU, int8 quantization, language forced to "de".
- Language detection is disabled because language="de" is forced; the bot targets
  German customer-service speech.
- Swiss German has not been evaluated.
- This is a prototype choice, not a claim that it is the optimal production setup.

The model is loaded lazily on first use and cached, so model-load time is
reported separately from transcription time. Transcripts are never printed or
logged here; they may contain PII.
"""

import time

from faster_whisper import WhisperModel

MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LANGUAGE = "de"

_model = None


def get_model():
    """Returns (model, load_seconds); load_seconds is None if the model was already cached."""
    global _model
    if _model is not None:
        return _model, None
    start = time.perf_counter()
    _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model, time.perf_counter() - start


def transcribe(audio_path: str) -> dict:
    model, load_seconds = get_model()

    start = time.perf_counter()
    segments, _info = model.transcribe(audio_path, language=LANGUAGE)
    # segments is a lazy generator: consuming it inside the timed section is
    # what actually runs the decoding.
    text = " ".join(segment.text.strip() for segment in segments)
    elapsed = time.perf_counter() - start

    return {
        "text": " ".join(text.split()),
        "language": LANGUAGE,
        "language_forced": True,
        "elapsed_seconds": elapsed,
        "model_load_seconds": load_seconds,  # None when the cached model was reused
    }
