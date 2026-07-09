"""
api/transcribe.py — Speech-to-text endpoint for voice dictation.

POST /transcribe   multipart audio blob  ->  {"text": "..."}

Reuses the existing faster-whisper singleton from ingestion.processors.audio.
The recorded blob is written to a temp file, transcribed, and the temp file is
always removed afterward.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from ingestion.processors.audio import transcribe

logger = logging.getLogger(__name__)
router = APIRouter()

# Browser MediaRecorder MIME types -> sensible temp-file extension. faster-whisper
# decodes via PyAV which sniffs the container by content, so the extension is a
# hint only — but a correct one avoids any ambiguity.
_EXT_BY_TYPE = {
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".m4a",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
}


def _ext_for_content_type(content_type: str | None) -> str:
    """Return a temp-file extension for a recorded-audio MIME type."""
    if not content_type:
        return ".bin"
    base = content_type.split(";", 1)[0].strip().lower()
    return _EXT_BY_TYPE.get(base, ".bin")


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe an uploaded audio blob to text via faster-whisper."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    suffix = _ext_for_content_type(file.content_type)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        text, _title = transcribe(tmp_path)
        return {"text": text}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
