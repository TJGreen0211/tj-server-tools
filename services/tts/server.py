import io
import re
import subprocess
import tempfile
import threading
from pathlib import Path

import soundfile as sf
import torch
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

VOICES = Path("/voices")
MAX_CHUNK = 300  # characters per generation call; Chatterbox degrades on long inputs
FORMATS = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "opus": "audio/ogg"}

# Skip MIOpen: it compiles kernels for every new input shape. PyTorch's own ROCm kernels avoid that.
torch.backends.cudnn.enabled = False

model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device("cuda"))
gpu_lock = threading.Lock()
app = FastAPI()


class SpeechRequest(BaseModel):
    input: str
    voice: str = "default"
    language: str = "en"
    response_format: str = "wav"
    model: str = "chatterbox-multilingual"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8


def chunks(text: str) -> list[str]:
    """Split text into sentence groups of at most MAX_CHUNK characters."""
    out = []
    for sentence in re.split(r"(?<=[.!?¿¡;])\s+", text.strip()):
        if out and len(out[-1]) + len(sentence) < MAX_CHUNK:
            out[-1] += " " + sentence
        else:
            out.append(sentence)
    return [c for c in out if c.strip()]


def voice_path(name: str) -> Path | None:
    """Reference clip for a voice name, or None for Chatterbox's built-in voice (also used for unknown names)."""
    matches = sorted(VOICES.glob(f"{name}.*")) if re.fullmatch(r"[\w-]+", name) else []
    return matches[0] if matches else None


def encode(wav: torch.Tensor, fmt: str) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav.squeeze(0).numpy(), model.sr, format="WAV", subtype="PCM_16")
    if fmt == "wav":
        return buf.getvalue()
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "wav", "-i", "-", "-f", "ogg" if fmt == "opus" else fmt, "-"],
        input=buf.getvalue(), capture_output=True, check=True,
    ).stdout


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"language must be one of {sorted(SUPPORTED_LANGUAGES)}")
    if req.response_format not in FORMATS:
        raise HTTPException(400, f"response_format must be one of {list(FORMATS)}")
    prompt = voice_path(req.voice)
    parts = []
    with gpu_lock, torch.inference_mode():
        for text in chunks(req.input):
            parts.append(model.generate(
                text, language_id=req.language, audio_prompt_path=str(prompt) if prompt else None,
                exaggeration=req.exaggeration, cfg_weight=req.cfg_weight, temperature=req.temperature,
            ).cpu())
    if not parts:
        raise HTTPException(400, "input is empty")
    gap = torch.zeros(1, int(model.sr * 0.2))
    wav = torch.cat([p for part in parts for p in (part, gap)][:-1], dim=1)
    return Response(encode(wav, req.response_format), media_type=FORMATS[req.response_format])


@app.get("/v1/voices")
def list_voices():
    return {"voices": ["default"] + sorted(p.stem for p in VOICES.iterdir() if p.is_file())}


@app.post("/v1/voices")
def add_voice(name: str = Form(...), file: UploadFile = File(...)):
    """Save a reference clip (about 10 seconds of clean speech) as a cloneable voice."""
    if not re.fullmatch(r"[\w-]+", name) or name == "default":
        raise HTTPException(400, "name may only contain letters, digits, _ and -")
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        for old in VOICES.glob(f"{name}.*"):
            old.unlink()
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", tmp.name, "-ac", "1", str(VOICES / f"{name}.wav")],
                       check=True)
    return {"voice": name}
