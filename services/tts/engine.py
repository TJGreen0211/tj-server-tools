"""Turns speech requests into audio with a Qwen3-TTS model (faster-qwen3-tts).

The model is passed in, so this module can be tested without a GPU. It needs:
  sample_rate                                   output rate (24 kHz for Qwen3-TTS)
  generate_voice_clone(text, language, **kw)    -> ([audio], sample_rate)
  generate_voice_clone_streaming(..., chunk_size=n) -> yields (audio_chunk, sample_rate, timing)

The GPU serves one generation at a time. Nothing here trusts the client to behave: input is
capped, every generation has a deadline, waiting for the GPU is bounded (callers get Busy
instead of queueing forever), and a streamed generation runs in its own thread that stops on a
flag rather than on the client closing anything.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from tempo import MAX_SPEED, MIN_SPEED, TimeStretch
from tts_utils import Voice, VoiceError, VoiceStore, chunks, language_name, silence

log = logging.getLogger("uvicorn.error")

# pause between the sentence groups a long input is split into
GAP_SEC = 0.15


class RequestError(ValueError):
    """A request the service can't serve (bad language, empty or too long text, unknown voice)."""


class Busy(RuntimeError):
    """The GPU didn't become free in time, or too many requests are already waiting for it."""


@dataclass
class Job:
    pieces: list[str]
    language: str
    voice: Voice
    speed: float = 1.0


class Engine:
    def __init__(self, model, voices: VoiceStore, clone_mode: str = "auto", stream_chunk: int = 8,
                 max_input_chars: int = 4000, max_gen_sec: float = 300.0,
                 wait_sec: float = 60.0, max_waiting: int = 8):
        self.model = model
        self.voices = voices
        self.clone_mode = clone_mode          # auto: ICL when the voice has a transcript; xvec: embedding only
        self.stream_chunk = stream_chunk      # codec steps per streamed chunk (12 steps = 1 s of audio)
        self.max_input_chars = max_input_chars
        self.max_gen_sec = max_gen_sec        # a generation stops here, whatever the client does
        self.wait_sec = wait_sec              # how long a request may wait for the GPU
        self.max_waiting = max_waiting        # how many requests may wait at once
        # CUDA graphs and the static cache serve one generation at a time
        self.lock = threading.Lock()
        self._state = threading.Lock()
        self._waiting = 0
        self.busy_since: float | None = None  # monotonic time the current generation took the GPU

    @property
    def sample_rate(self) -> int:
        return int(self.model.sample_rate)

    def status(self) -> dict:
        busy = self.busy_since
        return {"busy_sec": round(time.monotonic() - busy, 1) if busy else None, "waiting": self._waiting}

    def prepare(self, text: str, voice: str, language: str, speed: float = 1.0) -> Job:
        """Validates a request up front, so errors become HTTP errors before any audio is sent."""
        text = text or ""
        if len(text) > self.max_input_chars:
            raise RequestError(f"input is {len(text)} characters; the limit is {self.max_input_chars}")
        pieces = chunks(text)
        if not pieces:
            raise RequestError("input is empty")
        lang = language_name(language)
        if lang is None:
            raise RequestError(f"unsupported language {language!r}")
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise RequestError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        try:
            v = self.voices.resolve(voice)
        except VoiceError as e:
            raise RequestError(str(e))
        return Job(pieces, lang, v, float(speed))

    # -- the GPU ------------------------------------------------------------

    def reserve(self, timeout: float | None = None):
        """Takes the GPU for one generation, waiting at most `timeout` (default wait_sec).
        Raises Busy rather than queueing without end. The caller then passes held=True to
        stream / stream_in_thread / synthesize, which release it when done."""
        with self._state:
            if self._waiting >= self.max_waiting:
                raise Busy(f"{self._waiting} requests are already waiting for the GPU")
            self._waiting += 1
        try:
            if not self.lock.acquire(timeout=self.wait_sec if timeout is None else timeout):
                raise Busy("the GPU stayed busy; try again shortly")
        finally:
            with self._state:
                self._waiting -= 1
        self.busy_since = time.monotonic()

    def _release(self):
        self.busy_since = None
        self.lock.release()

    def _clone_args(self, job: Job) -> dict:
        return {
            "language": job.language,
            "ref_audio": str(job.voice.audio),
            "ref_text": job.voice.transcript,
            "xvec_only": job.voice.xvec_only(self.clone_mode),
        }

    def stream(self, job: Job, stop: threading.Event | None = None, held: bool = False) -> Iterator[np.ndarray]:
        """Audio (float32, mono, sample_rate) as it is generated. Holds the GPU until it is done,
        `stop` is set, the deadline passes (both checked after every chunk), or the generator
        is closed. held=True: the caller already reserved the GPU. A job's speed is applied
        as the audio is generated."""
        if not held:
            self.reserve()
        deadline = time.monotonic() + self.max_gen_sec
        stretch = TimeStretch(job.speed, self.sample_rate) if job.speed != 1 else None
        try:
            for i, text in enumerate(job.pieces):
                if (stop is not None and stop.is_set()) or time.monotonic() > deadline:
                    return
                if i:
                    gap = silence(GAP_SEC, self.sample_rate)
                    yield stretch.process(gap) if stretch else gap
                for audio, _sr, _timing in self.model.generate_voice_clone_streaming(
                        text=text, chunk_size=self.stream_chunk, **self._clone_args(job)):
                    if stop is not None and stop.is_set():
                        return
                    if time.monotonic() > deadline:
                        log.warning("speech generation hit its %.0f s limit; stopping", self.max_gen_sec)
                        return
                    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                    if stretch:
                        audio = stretch.process(audio)
                    if audio.size:
                        yield audio
            if stretch:
                tail = stretch.flush()
                if tail.size:
                    yield tail
        finally:
            self._release()

    def stream_in_thread(self, job: Job, convert=lambda audio: audio, held: bool = False):
        """Runs stream() in its own thread, which alone owns the generator, and returns
        (chunks, stop): `chunks` is a queue of convert(audio) items ending with None (an
        Exception item before it means generation failed), and setting `stop` makes the thread
        finish after its current chunk and free the GPU.

        HTTP streaming must use this: a web framework doesn't reliably close a generator whose
        client disconnected, and the thread finishes on its own (stop, deadline or end of text)
        even if nobody ever reads the queue."""
        out: queue.Queue = queue.Queue()
        stop = threading.Event()
        if not held:
            self.reserve()

        def produce():
            gen = self.stream(job, stop, held=True)
            try:
                for audio in gen:
                    if stop.is_set():
                        break
                    out.put(convert(audio))
            except Exception as e:
                out.put(e)
            finally:
                gen.close()
                out.put(None)

        threading.Thread(target=produce, daemon=True, name="tts-stream").start()
        return out, stop

    def synthesize(self, job: Job, held: bool = False) -> np.ndarray:
        """The whole utterance at once (for formats that can't be streamed)."""
        if not held:
            self.reserve()
        deadline = time.monotonic() + self.max_gen_sec
        parts = []
        try:
            for i, text in enumerate(job.pieces):
                if time.monotonic() > deadline:
                    log.warning("speech generation hit its %.0f s limit; stopping", self.max_gen_sec)
                    break
                if i:
                    parts.append(silence(GAP_SEC, self.sample_rate))
                audio_list, _sr = self.model.generate_voice_clone(text=text, **self._clone_args(job))
                parts += [np.asarray(a, dtype=np.float32).reshape(-1) for a in audio_list]
        finally:
            self._release()
        audio = np.concatenate(parts) if parts else silence(0.1, self.sample_rate)
        if job.speed != 1:
            stretch = TimeStretch(job.speed, self.sample_rate)
            audio = np.concatenate([stretch.process(audio), stretch.flush()])
        return audio

    def voices_changed(self):
        """A voice file was replaced or removed: drop the model's cached reference prompts
        (faster-qwen3-tts caches them by file path)."""
        cache = getattr(self.model, "_voice_prompt_cache", None)
        if isinstance(cache, dict):
            self.reserve()
            try:
                cache.clear()
            finally:
                self._release()

    def hung(self, grace_sec: float = 120.0) -> bool:
        """True if one generation has held the GPU well past its deadline: stop flags and
        deadlines only act between chunks, so this means the model itself is stuck."""
        busy = self.busy_since
        return busy is not None and time.monotonic() - busy > self.max_gen_sec + grace_sec
