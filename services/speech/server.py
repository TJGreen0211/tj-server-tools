import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from nemo.collections.asr.models import ASRModel, EncDecSpeakerLabelModel
from omegaconf import open_dict

import meetings
from audio import FFMPEG, SAMPLE_RATE, is_silent, pad_to_whole_seconds, pad_turns, tidy_text
from diarize import diarize

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
SPEAKER_MODEL = "titanet_large"  # the same voice-embedding model the diarizer uses
# a meeting speaker's voice embedding is taken from up to this much of their speech per window
EMBED_MAX_SEC = 30
# clip lengths (whole seconds) run once at startup so live captions never hit a cold length
WARM_SEC = int(os.environ.get("STT_WARM_SEC", "60"))

log = logging.getLogger("uvicorn.error")

# Load and convert on CPU: putting fp32 weights on the GPU first would peak past what is
# left after tts, and fp16 halves the resident size so all three models fit one 8 GB card.
model = ASRModel.from_pretrained(MODEL_NAME, map_location="cpu").eval()
# Local attention lets Parakeet handle multi-hour audio without running out of VRAM.
model.change_attention_model("rel_pos_local_attn", [256, 256])
model.change_subsampling_conv_chunking_factor(1)
# Keep the graph decoder off: audio length varies per request, so graph capture buys little.
decoding_cfg = model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.greedy.use_cuda_graph_decoder = False
model.change_decoding_strategy(decoding_cfg)
model.half().to("cuda")

# fp32 unlike the ASR model: infer_segment feeds float clips straight in, and the diarizer
# is handed this model, so one fp32 copy serves both paths and matches its embeddings.
speaker_model = EncDecSpeakerLabelModel.from_pretrained(SPEAKER_MODEL, map_location="cpu").eval().to("cuda")

gpu_lock = threading.Lock()
app = FastAPI()


def _warm_up():
    """Runs every whole-second length up to WARM_SEC once (see pad_to_whole_seconds), one
    at a time under the GPU lock so real requests are served in between."""
    start = time.monotonic()
    noise = np.random.default_rng(0).normal(0, 0.01, WARM_SEC * SAMPLE_RATE).astype(np.float32)
    for sec in range(1, WARM_SEC + 1):
        with gpu_lock, torch.inference_mode():
            model.transcribe([noise[: sec * SAMPLE_RATE]], timestamps=True, verbose=False)
    log.info("speech warm-up: clip lengths 1-%d s ready in %.0f s", WARM_SEC, time.monotonic() - start)


if WARM_SEC > 0:
    threading.Thread(target=_warm_up, daemon=True, name="stt-warm-up").start()


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        [FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), str(dst)],
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


EMPTY = {"text": "", "segments": [], "words": []}


def transcribe_plain(audio: np.ndarray) -> dict:
    """Whole clip in one pass: fast, for short single-speaker audio like dictation."""
    if is_silent(audio):
        return EMPTY
    with gpu_lock, torch.inference_mode():
        hyp = model.transcribe([pad_to_whole_seconds(audio)], timestamps=True, verbose=False)[0]
    segments = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": tidy_text(s["segment"])}
                for s in hyp.timestamp["segment"]]
    words = [{"word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)} for w in hyp.timestamp["word"]]
    return {"text": tidy_text(hyp.text), "segments": segments, "words": words}


def transcribe_turns(audio: np.ndarray, turns: list[dict]) -> tuple[list, list]:
    """Transcribes each speaker turn on its own (caller holds gpu_lock).

    Parakeet picks one language per stretch of audio, so transcribing turns separately
    keeps English and Spanish turns from swallowing each other. Each turn is cut with a
    little padding so its first and last syllables survive; silent turns are skipped."""
    kept, clips = [], []
    for turn, (lo, hi) in zip(turns, pad_turns(turns, len(audio) / SAMPLE_RATE)):
        clip = audio[int(lo * SAMPLE_RATE): int(hi * SAMPLE_RATE)]
        if not is_silent(clip):
            kept.append((turn, lo))
            clips.append(clip)
    hyps = model.transcribe(clips, batch_size=8, timestamps=True, verbose=False) if clips else []

    segments, words = [], []
    for (turn, off), hyp in zip(kept, hyps):
        if not hyp.text.strip():
            continue
        segments.append({"start": round(turn["start"], 2), "end": round(turn["end"], 2), "speaker": turn["speaker"], "text": tidy_text(hyp.text)})
        words += [
            {"word": w["word"], "start": round(off + w["start"], 2), "end": round(off + w["end"], 2), "speaker": turn["speaker"]}
            for w in hyp.timestamp["word"]
        ]
    return segments, words


def transcribe_diarized(wav: Path, audio: np.ndarray, work_dir: Path, num_speakers: int | None, max_speakers: int) -> dict:
    """Diarize the whole recording, then transcribe each speaker turn on its own."""
    with gpu_lock, torch.inference_mode():
        turns = diarize(wav, work_dir, num_speakers, max_speakers, speaker_model=speaker_model)
        segments, words = transcribe_turns(audio, turns)
    return {"text": " ".join(s["text"] for s in segments), "segments": segments, "words": words}


def speaker_embeddings(audio: np.ndarray, turns: list[dict]) -> dict:
    """{speaker: (voice embedding, seconds of speech)} from each speaker's turns (caller holds gpu_lock)."""
    out = {}
    for spk in dict.fromkeys(t["speaker"] for t in turns):
        mine = [t for t in turns if t["speaker"] == spk]
        clip = np.concatenate([audio[int(t["start"] * SAMPLE_RATE): int(t["end"] * SAMPLE_RATE)] for t in mine])
        if len(clip) < SAMPLE_RATE // 2:  # too little to embed reliably; loop what there is
            clip = np.resize(clip, SAMPLE_RATE // 2) if len(clip) else np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        emb, _ = speaker_model.infer_segment(clip[: EMBED_MAX_SEC * SAMPLE_RATE])
        out[spk] = (emb.squeeze().float().cpu().numpy(), sum(t["end"] - t["start"] for t in mine))
    return out


def process_window(audio: np.ndarray, max_speakers: int, single_speaker: bool) -> dict:
    """One window of a meeting: diarize, transcribe each turn, and embed each speaker's voice."""
    with tempfile.TemporaryDirectory() as tmp, gpu_lock, torch.inference_mode():
        if single_speaker:
            turns = [{"start": 0.0, "end": len(audio) / SAMPLE_RATE, "speaker": "SPEAKER_0"}]
        else:
            wav = Path(tmp) / "window.wav"
            sf.write(wav, audio, SAMPLE_RATE)
            turns = diarize(wav, Path(tmp), None, max_speakers, speaker_model=speaker_model)
        segments, words = transcribe_turns(audio, turns)
        speakers = speaker_embeddings(audio, turns) if turns else {}
    return {"segments": segments, "words": words, "speakers": speakers}


meeting_sessions = meetings.Registry(process_window)


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


# -- meetings: the recording streams in while it happens, so finishing only has the last window left

@app.post("/v1/meetings")
def meeting_start(num_speakers: int | None = Form(None), max_speakers: int = Form(8)):
    return {"id": meeting_sessions.create(num_speakers, max_speakers).id}


def _session(sid: str) -> meetings.Session:
    session = meeting_sessions.get(sid)
    if session is None:
        raise HTTPException(404, "unknown or expired meeting")
    return session


@app.post("/v1/meetings/{sid}/audio")
def meeting_audio(sid: str, file: UploadFile = File(...), part: int = Form(1), seq: int | None = Form(None)):
    """Appends the next piece of recorder part `part`. Pieces must arrive in order; `seq` (0, 1, 2, ...
    per part) lets a retried piece be recognised and ignored instead of appended twice."""
    session = _session(sid)
    try:
        appended = session.append(part, file.file.read(), seq)
    except (meetings.SessionError, RuntimeError) as e:
        raise HTTPException(409, str(e))
    session.schedule(meeting_sessions.executor)
    return {**session.status(), "duplicate": not appended}


@app.get("/v1/meetings/{sid}")
def meeting_status(sid: str):
    return _session(sid).status()


@app.post("/v1/meetings/{sid}/finish")
def meeting_finish(sid: str, response_format: str = Form("verbose_json")):
    """Transcribes whatever is left and returns the whole meeting, labelled by speaker."""
    session = _session(sid)
    try:
        result = session.finish()
    except meetings.SessionError as e:
        raise HTTPException(409, str(e))
    finally:
        meeting_sessions.pop(sid)
    if response_format == "text":
        return PlainTextResponse("\n".join(f"[{ts(s['start'], '.')[:8]}] {s['speaker']}: {s['text']}" for s in result["segments"]))
    if response_format in ("srt", "vtt"):
        return PlainTextResponse(subtitles(result["segments"], response_format))
    return JSONResponse(result)


@app.delete("/v1/meetings/{sid}")
def meeting_discard(sid: str):
    session = meeting_sessions.pop(sid)
    if session:
        session.discard()
    return {"discarded": session is not None}
