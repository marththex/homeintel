"""
ingestion/processors/audio.py — Audio transcription via faster-whisper.

Transcribes audio files using a local Whisper model (GPU-accelerated on CUDA).
The transcript is fed into the standard text embedding pipeline as modality=audio.

Model is lazy-loaded as a singleton on first use. First call will be slow if
the model weights are not already cached in ~/.cache/huggingface/hub/.

WHISPER_MODEL_SIZE controls accuracy vs speed:
  tiny   — fastest, lowest accuracy
  base   — good balance for short clips and clear speech (default)
  small  — better for accents and background noise
  medium — high accuracy, noticeable GPU memory use (~3 GB VRAM)
  large-v3 — best accuracy, ~6 GB VRAM
"""

import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from faster_whisper import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "Loading Whisper model=%s device=%s compute=%s",
            settings.whisper_model_size, device, compute_type,
        )
        _model = WhisperModel(
            settings.whisper_model_size,
            device=device,
            compute_type=compute_type,
        )
    return _model


def transcribe(path: Path) -> tuple[str, str]:
    """
    Transcribe an audio file with faster-whisper.

    Returns (transcript_text, title) where title is derived from the filename.
    Returns ("", "") if the file produces no speech segments.
    """
    model = _get_model()

    logger.debug("Transcribing %s", path.name)
    segments, info = model.transcribe(str(path), beam_size=5)

    parts: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            parts.append(text)

    transcript = " ".join(parts)
    if not transcript:
        logger.warning("No speech detected in %s — skipping", path.name)
        return "", ""

    logger.info(
        "Transcribed %s: %.1fs audio → %d chars (lang=%s)",
        path.name, info.duration, len(transcript), info.language,
    )

    title = path.stem.replace("_", " ").replace("-", " ").title()
    return transcript, title
