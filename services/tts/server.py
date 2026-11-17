"""Text to speech with voice cloning: Qwen3-TTS through faster-qwen3-tts (CUDA graphs, streaming).

POST /v1/audio/speech follows the OpenAI shape. response_format "pcm" (raw 16-bit mono at the
X-Sample-Rate header's rate) is streamed as it is generated, so playback can start after the first
chunk; "wav" streams too when stream=true. mp3 / flac / opus are returned once complete.

The service protects itself rather than trusting clients: text length is capped, every generation
has a deadline, waiting for the GPU is bounded (503 when busy), a disconnect stops generation even
before the first audio, and a watchdog restarts the process if the model itself hangs.
"""
import asyncio
import logging
import os
import struct
import threading
import time

import anyio
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from faster_qwen3_tts import FasterQwen3TTS
from pydantic import BaseModel

from engine import Busy, Engine, RequestError
from tts_utils import FORMATS, VoiceError, VoiceStore, encode, pcm16

log = logging.getLogger("uvicorn.error")

MODEL_NAME = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "default")
CLONE_MODE = os.environ.get("TTS_CLONE_MODE", "auto")          # auto | xvec
STREAM_CHUNK = int(os.environ.get("TTS_STREAM_CHUNK", "8"))     # codec steps per chunk; 12 = 1 s of audio
MAX_INPUT_CHARS = int(os.environ.get("TTS_MAX_INPUT_CHARS", "4000"))
MAX_GEN_SEC = float(os.environ.get("TTS_MAX_GEN_SEC", "300"))   # a generation stops here, client or not
WAIT_SEC = float(os.environ.get("TTS_WAIT_SEC", "60"))          # longest a request waits for the GPU
MAX_WAITING = int(os.environ.get("TTS_MAX_WAITING", "8"))       # requests waiting beyond this get 503


def pick_dtype() -> torch.dtype:
    """bf16 needs Ampere (compute 8.x) or newer; Turing cards like the RTX 2070 run fp16.
    TTS_DTYPE=float32 is the fallback if fp16 output is garbled or silent."""
    wanted = os.environ.get("TTS_DTYPE", "auto").strip().lower()
    if wanted != "auto":
        return getattr(torch, wanted)
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


DTYPE = pick_dtype()
log.info("loading %s (%s) on %s", MODEL_NAME, DTYPE, torch.cuda.get_device_name())
model = FasterQwen3TTS.from_pretrained(MODEL_NAME, device="cuda", dtype=DTYPE, attn_implementation="sdpa")
model.warmup(prefill_len=100)   # capture the CUDA graphs now, not on the first request

voices = VoiceStore("/voices", default=DEFAULT_VOICE)
# the repo ships a synthetic default voice (default_voice/); install it unless one was uploaded
if voices.seed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_voice")):
    log.info("installed the bundled %r voice", DEFAULT_VOICE)
engine = Engine(model, voices, clone_mode=CLONE_MODE, stream_chunk=STREAM_CHUNK,
                max_input_chars=MAX_INPUT_CHARS, max_gen_sec=MAX_GEN_SEC,
                wait_sec=WAIT_SEC, max_waiting=MAX_WAITING)


def _watchdog():
    """Deadlines and stop flags act between chunks; if one generation holds the GPU far past its
    deadline, the model itself is stuck and only a fresh process recovers. Exiting lets Docker's
    restart policy bring the service back instead of leaving it hung."""
    while True:
        time.sleep(15)
        if engine.hung():
            log.critical("a speech generation has held the GPU for %s s; restarting the service",
                         engine.status()["busy_sec"])
            os._exit(1)


threading.Thread(target=_watchdog, daemon=True, name="tts-watchdog").start()


def _warm_generation():
    """First use is much slower than steady state; pay it at startup, and catch fp16 trouble early."""
    if voices.get(DEFAULT_VOICE) is None:
        log.warning("no %r voice yet; upload one: POST /v1/voices (name, file, text)", DEFAULT_VOICE)
        return
    job = engine.prepare("Hello, this is a warm-up. Hola, esto es una prueba.", DEFAULT_VOICE, "auto")
    audio = engine.synthesize(job)
    if not np.isfinite(audio).all() or float(np.abs(audio).max(initial=0.0)) < 1e-3:
        log.error("warm-up produced invalid or silent audio with %s; try TTS_DTYPE=float32", DTYPE)


_warm_generation()
app = FastAPI()


class SpeechRequest(BaseModel):
    input: str
    voice: str = DEFAULT_VOICE
    language: str = "auto"            # ISO code (en, es, ...) or name (English, Spanish, ...) or auto
    response_format: str = "wav"      # pcm | wav | mp3 | flac | opus
    stream: bool | None = None        # pcm always streams; wav streams when true
    model: str = "qwen3-tts"


def wav_stream_header(sample_rate: int) -> bytes:
    """A WAV header for a stream of unknown length (players read until the data ends)."""
    size = 0xFFFFFFFF
    return (b"RIFF" + struct.pack("<I", size) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data" + struct.pack("<I", size))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "dtype": str(DTYPE).replace("torch.", ""),
            "sample_rate": engine.sample_rate, "default_voice": DEFAULT_VOICE,
            "default_voice_ready": voices.get(DEFAULT_VOICE) is not None, **engine.status()}


def _busy(e: Busy) -> HTTPException:
    return HTTPException(503, str(e), headers={"Retry-After": "5"})


async def _watch_disconnect(request: Request, stop: threading.Event):
    """Sets `stop` when the client hangs up, from the start of the request: the web server would
    only notice when it next tried to send audio, which may not have happened yet."""
    while not stop.is_set():
        message = await request.receive()
        if message["type"] == "http.disconnect":
            stop.set()


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest, request: Request):
    fmt = req.response_format.lower()
    if fmt not in FORMATS:
        raise HTTPException(400, f"response_format must be one of {list(FORMATS)}")
    try:
        job = engine.prepare(req.input, req.voice, req.language)
    except RequestError as e:
        raise HTTPException(400, str(e))

    sr = engine.sample_rate
    headers = {"X-Sample-Rate": str(sr), "X-Channels": "1", "X-Sample-Format": "s16le"}
    try:
        engine.reserve()        # waits at most TTS_WAIT_SEC, so a busy GPU is a 503, not an endless queue
    except Busy as e:
        raise _busy(e)

    if fmt == "pcm" or (fmt == "wav" and req.stream):
        # generation runs in its own thread and stops on its own (end of text, deadline) or when
        # `stop` is set: by the disconnect watcher, or when this response ends for any reason
        chunks, stop = engine.stream_in_thread(job, pcm16, held=True)

        async def body():
            watcher = asyncio.create_task(_watch_disconnect(request, stop))
            try:
                if fmt == "wav":
                    yield wav_stream_header(sr)
                while True:
                    item = await anyio.to_thread.run_sync(chunks.get)
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        log.error("speech stream failed: %s", item)
                        break
                    yield item
            finally:
                stop.set()
                watcher.cancel()
        return StreamingResponse(body(), media_type=FORMATS[fmt], headers=headers)

    audio = engine.synthesize(job, held=True)
    return Response(encode(audio, sr, fmt), media_type=FORMATS[fmt], headers=headers)


@app.get("/v1/voices")
def list_voices():
    names = voices.names()
    return {"voices": names, "default": DEFAULT_VOICE,
            "with_transcript": [n for n in names if voices.get(n).transcript]}


@app.post("/v1/voices")
def add_voice(name: str = Form(...), file: UploadFile = File(...), text: str = Form("")):
    """Saves a cloneable voice: about 10 s of clean speech, plus `text` (what is said in it).
    With the transcript the clone is closer to the speaker; without it only the voice's
    embedding is used."""
    try:
        voice = voices.add(name, file.file.read(), text)
    except VoiceError as e:
        raise HTTPException(400, str(e))
    try:
        engine.voices_changed()
    except Busy as e:
        raise _busy(e)
    return {"voice": voice.name, "transcript": bool(voice.transcript)}


@app.delete("/v1/voices/{name}")
def remove_voice(name: str):
    if not voices.remove(name):
        raise HTTPException(404, f"no voice {name!r}")
    try:
        engine.voices_changed()
    except Busy as e:
        raise _busy(e)
    return {"removed": name}
