"""Helpers for the TTS service: voices, text chunking, language names, audio encoding. No model code here."""
import io
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

FORMATS = {"pcm": "audio/pcm", "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "opus": "audio/ogg"}

# Qwen3-TTS takes language names. Requests may use ISO codes (as openlumara sends) or names.
LANGUAGES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German",
    "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian",
}

# Long inputs are generated a few sentences at a time, so a request never outgrows the model's
# static cache and its first audio doesn't wait on the whole text being prefilled.
MAX_CHUNK = 300

VOICE_NAME = re.compile(r"[\w-]+")
MIN_REF_SEC, MAX_REF_SEC = 3.0, 30.0
VOICE_RATE = 24000


def language_name(value: str) -> str | None:
    """'es' / 'Spanish' / 'spanish' -> 'Spanish'; '' / 'auto' -> 'Auto'; unsupported -> None."""
    v = (value or "").strip().lower()
    if v in ("", "auto"):
        return "Auto"
    if v in LANGUAGES:
        return LANGUAGES[v]
    names = {n.lower(): n for n in LANGUAGES.values()}
    return names.get(v)


def chunks(text: str, max_chars: int = MAX_CHUNK) -> list[str]:
    """Split text into sentence groups of at most max_chars (a single longer sentence stays whole)."""
    out = []
    for sentence in re.split(r"(?<=[.!?¿¡;…])\s+|\n+", text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if out and len(out[-1]) + 1 + len(sentence) <= max_chars:
            out[-1] += " " + sentence
        else:
            out.append(sentence)
    return out


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] -> 16-bit little-endian PCM."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def encode(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """A complete audio file in `fmt` (wav, mp3, flac, opus)."""
    wav = wav_bytes(audio, sample_rate)
    if fmt == "wav":
        return wav
    container = {"mp3": "mp3", "flac": "flac", "opus": "ogg"}[fmt]
    codec = ["-c:a", "libopus"] if fmt == "opus" else []
    return subprocess.run(
        [FFMPEG, "-nostdin", "-loglevel", "error", "-f", "wav", "-i", "-", *codec, "-f", container, "-"],
        input=wav, capture_output=True, check=True,
    ).stdout


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


# ---------------------------------------------------------------------------
# voices: <name>.wav (the reference clip) and an optional <name>.txt (its transcript)
# ---------------------------------------------------------------------------

class VoiceError(Exception):
    pass


@dataclass
class Voice:
    name: str
    audio: Path
    transcript: str

    def xvec_only(self, clone_mode: str) -> bool:
        """ICL cloning (closest to the reference) needs the transcript; without it, use the speaker
        embedding only."""
        return clone_mode == "xvec" or not self.transcript


class VoiceStore:
    def __init__(self, directory: str | Path, default: str = "default"):
        self.dir = Path(directory)
        self.default = default

    def names(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.wav"))

    def get(self, name: str) -> Voice | None:
        if not VOICE_NAME.fullmatch(name or ""):
            return None
        audio = self.dir / f"{name}.wav"
        if not audio.is_file():
            return None
        txt = self.dir / f"{name}.txt"
        return Voice(name, audio, txt.read_text(encoding="utf-8").strip() if txt.is_file() else "")

    def resolve(self, name: str) -> Voice:
        """The named voice, else the default voice. Raises VoiceError when neither exists."""
        voice = self.get(name) or self.get(self.default)
        if voice is None:
            raise VoiceError(
                f"No voice {name or self.default!r} and no default voice {self.default!r}. Upload one first: "
                "POST /v1/voices with name, file (about 10 s of clean speech) and text (what is said in it)."
            )
        return voice

    def add(self, name: str, data: bytes, transcript: str = "") -> Voice:
        """Stores a reference clip (any format ffmpeg reads) as 24 kHz mono WAV, plus its transcript."""
        if not VOICE_NAME.fullmatch(name or ""):
            raise VoiceError("name may only contain letters, digits, _ and -")
        self.dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "upload", Path(tmp) / "voice.wav"
            src.write_bytes(data)
            done = subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                                   "-ac", "1", "-ar", str(VOICE_RATE), str(dst)], capture_output=True)
            if done.returncode != 0:
                raise VoiceError("could not read the uploaded audio")
            seconds = sf.info(str(dst)).duration
            if not MIN_REF_SEC <= seconds <= MAX_REF_SEC:
                raise VoiceError(f"the reference clip is {seconds:.1f} s; use {MIN_REF_SEC:.0f}-{MAX_REF_SEC:.0f} s "
                                 "of clean speech (about 10 s is ideal)")
            (self.dir / f"{name}.wav").write_bytes(dst.read_bytes())
        txt = self.dir / f"{name}.txt"
        if transcript.strip():
            txt.write_text(transcript.strip(), encoding="utf-8")
        elif txt.exists():
            txt.unlink()
        return self.get(name)

    def remove(self, name: str) -> bool:
        voice = self.get(name)
        if voice is None:
            return False
        voice.audio.unlink()
        txt = self.dir / f"{name}.txt"
        if txt.exists():
            txt.unlink()
        return True
