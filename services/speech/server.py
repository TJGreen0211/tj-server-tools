import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from nemo.collections.asr.models import ASRModel
from omegaconf import open_dict

from diarize import diarize

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000

# Skip MIOpen: it compiles kernels for every new audio length (minutes per request) and its
# LSTM fails on first use. PyTorch's own ROCm kernels are about as fast here with neither problem.
torch.backends.cudnn.enabled = False

model = ASRModel.from_pretrained(MODEL_NAME, map_location="cuda").eval()
# Local attention lets Parakeet handle multi-hour audio without running out of VRAM.
model.change_attention_model("rel_pos_local_attn", [256, 256])
model.change_subsampling_conv_chunking_factor(1)
# CUDA graphs are NVIDIA-only; turn them off for ROCm.
decoding_cfg = model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.greedy.use_cuda_graph_decoder = False
model.change_decoding_strategy(decoding_cfg)

gpu_lock = threading.Lock()
app = FastAPI()


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), str(dst)],
        check=True,
    )


def load_audio(uploads: list[UploadFile], work_dir: Path) -> tuple[Path, np.ndarray]:
    """Decode every upload to 16 kHz mono and join them in order (e.g. the parts of a paused recording)."""
    parts = []
    for i, upload in enumerate(uploads):
        src, dst = work_dir / f"input{i}", work_dir / f"part{i}.wav"
        src.write_bytes(upload.file.read())
        to_wav(src, dst)
        parts.append(sf.read(dst, dtype="float32")[0])
    audio = np.concatenate(parts)
    wav = work_dir / "audio.wav"
    sf.write(wav, audio, SAMPLE_RATE)
    return wav, audio


def transcribe_plain(audio: np.ndarray) -> dict:
    """Whole clip in one pass: fast, for short single-speaker audio like dictation."""
    with gpu_lock, torch.inference_mode():
        hyp = model.transcribe([audio], timestamps=True, verbose=False)[0]
    segments = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["segment"]} for s in hyp.timestamp["segment"]]
    words = [{"word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)} for w in hyp.timestamp["word"]]
    return {"text": hyp.text, "segments": segments, "words": words}


def transcribe_diarized(wav: Path, audio: np.ndarray, work_dir: Path, num_speakers: int | None, max_speakers: int) -> dict:
    """Diarize, then transcribe each speaker turn on its own.

    Parakeet picks one language per stretch of audio, so transcribing turns separately
    keeps English and Spanish turns from swallowing each other."""
    with gpu_lock, torch.inference_mode():
        turns = diarize(wav, work_dir, num_speakers, max_speakers)
        clips = [audio[int(t["start"] * SAMPLE_RATE): int(t["end"] * SAMPLE_RATE)] for t in turns]
        hyps = model.transcribe(clips, batch_size=8, timestamps=True, verbose=False) if clips else []

    segments, words = [], []
    for turn, hyp in zip(turns, hyps):
        if not hyp.text.strip():
            continue
        off = turn["start"]
        segments.append({"start": round(off, 2), "end": round(turn["end"], 2), "speaker": turn["speaker"], "text": hyp.text})
        words += [
            {"word": w["word"], "start": round(off + w["start"], 2), "end": round(off + w["end"], 2), "speaker": turn["speaker"]}
            for w in hyp.timestamp["word"]
        ]
    return {"text": " ".join(s["text"] for s in segments), "segments": segments, "words": words}


def ts(t: float, sep: str) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02}{sep}{int((s % 1) * 1000):03}"


def subtitles(segments: list, fmt: str) -> str:
    sep = "," if fmt == "srt" else "."
    out = ["WEBVTT\n"] if fmt == "vtt" else []
    for i, s in enumerate(segments, 1):
        text = f"{s['speaker']}: {s['text']}" if "speaker" in s else s["text"]
        head = f"{i}\n" if fmt == "srt" else ""
        out.append(f"{head}{ts(s['start'], sep)} --> {ts(s['end'], sep)}\n{text}\n")
    return "\n".join(out)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: list[UploadFile] = File(...),
    model_name: str = Form(MODEL_NAME, alias="model"),
    response_format: str = Form("json"),
    diarize: bool = Form(False),
    num_speakers: int | None = Form(None),
    max_speakers: int = Form(8),
):
    with tempfile.TemporaryDirectory() as tmp:
        wav, audio = load_audio(file, Path(tmp))
        if diarize:
            result = transcribe_diarized(wav, audio, Path(tmp), num_speakers, max_speakers)
        else:
            result = transcribe_plain(audio)

    if response_format == "text":
        if diarize:
            return PlainTextResponse("\n".join(f"[{ts(s['start'], '.')[:8]}] {s['speaker']}: {s['text']}" for s in result["segments"]))
        return PlainTextResponse(result["text"])
    if response_format in ("srt", "vtt"):
        return PlainTextResponse(subtitles(result["segments"], response_format))
    if response_format == "verbose_json":
        return JSONResponse(result)
    return JSONResponse({"text": result["text"]})
