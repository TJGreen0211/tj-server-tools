"""Turns speech requests into audio with a Qwen3-TTS model (faster-qwen3-tts).

The model is passed in, so this module can be tested without a GPU. It needs:
  sample_rate                                   output rate (24 kHz for Qwen3-TTS)
  generate_voice_clone(text, language, **kw)    -> ([audio], sample_rate)
  generate_voice_clone_streaming(..., chunk_size=n) -> yields (audio_chunk, sample_rate, timing)
"""
import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from tts_utils import Voice, VoiceError, VoiceStore, chunks, language_name, silence

# pause between the sentence groups a long input is split into
GAP_SEC = 0.15


class RequestError(ValueError):
    """A request the service can't serve (bad language, empty text, unknown voice)."""


@dataclass
class Job:
    pieces: list[str]
    language: str
    voice: Voice


class Engine:
    def __init__(self, model, voices: VoiceStore, clone_mode: str = "auto", stream_chunk: int = 8):
        self.model = model
        self.voices = voices
        self.clone_mode = clone_mode          # auto: ICL when the voice has a transcript; xvec: embedding only
        self.stream_chunk = stream_chunk      # codec steps per streamed chunk (12 steps = 1 s of audio)
        # CUDA graphs and the static cache serve one generation at a time
        self.lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return int(self.model.sample_rate)

    def prepare(self, text: str, voice: str, language: str) -> Job:
        """Validates a request up front, so errors become HTTP errors before any audio is sent."""
        pieces = chunks(text or "")
        if not pieces:
            raise RequestError("input is empty")
        lang = language_name(language)
        if lang is None:
            raise RequestError(f"unsupported language {language!r}")
        try:
            v = self.voices.resolve(voice)
        except VoiceError as e:
            raise RequestError(str(e))
        return Job(pieces, lang, v)

    def _clone_args(self, job: Job) -> dict:
        return {
            "language": job.language,
            "ref_audio": str(job.voice.audio),
            "ref_text": job.voice.transcript,
            "xvec_only": job.voice.xvec_only(self.clone_mode),
        }

    def stream(self, job: Job) -> Iterator[np.ndarray]:
        """Audio (float32, mono, sample_rate) as it is generated. Holds the GPU until done or
        until the consumer stops iterating (a client that disconnects frees it)."""
        self.lock.acquire()
        try:
            for i, text in enumerate(job.pieces):
                if i:
                    yield silence(GAP_SEC, self.sample_rate)
                for audio, _sr, _timing in self.model.generate_voice_clone_streaming(
                        text=text, chunk_size=self.stream_chunk, **self._clone_args(job)):
                    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                    if audio.size:
                        yield audio
        finally:
            self.lock.release()

    def synthesize(self, job: Job) -> np.ndarray:
        """The whole utterance at once (for formats that can't be streamed)."""
        parts = []
        with self.lock:
            for i, text in enumerate(job.pieces):
                if i:
                    parts.append(silence(GAP_SEC, self.sample_rate))
                audio_list, _sr = self.model.generate_voice_clone(text=text, **self._clone_args(job))
                parts += [np.asarray(a, dtype=np.float32).reshape(-1) for a in audio_list]
        return np.concatenate(parts) if parts else silence(0.1, self.sample_rate)

    def voices_changed(self):
        """A voice file was replaced or removed: drop the model's cached reference prompts
        (faster-qwen3-tts caches them by file path)."""
        cache = getattr(self.model, "_voice_prompt_cache", None)
        if isinstance(cache, dict):
            with self.lock:
                cache.clear()
