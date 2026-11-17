"""Audio helpers shared by the transcription endpoints and meeting sessions. No model code here."""
import os
import re
import subprocess
import threading

import numpy as np

SAMPLE_RATE = 16000
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

# Each speaker turn is widened by this much on both sides (never past the midpoint to the next
# turn), so the diarizer's tight boundaries don't clip the first and last syllables.
TURN_PAD_SEC = 0.25

# A clip counts as silent when even its loud moments stay under this level (RMS, full scale = 1.0).
# Parakeet makes up a word or two on silence; skipping those clips avoids that.
SILENCE_RMS = float(os.environ.get("SILENCE_RMS", "0.005"))
MIN_SPEECH_SEC = 0.3


# Parakeet glues a sentence that opens with an inverted mark onto the one before it
# ("pendientes.¿Quién"); a space goes back in after the closing punctuation.
_GLUED_OPENER = re.compile(r"([.!?…:;,])([¿¡])")


def tidy_text(text: str) -> str:
    """Fixes spacing Parakeet gets wrong in its output text."""
    return _GLUED_OPENER.sub(r"\1 \2", text)


def pad_turns(turns: list[dict], duration: float) -> list[tuple[float, float]]:
    """(start, end) to cut for each turn: padded, but never past the midpoint to a neighbouring turn."""
    bounds = []
    for i, t in enumerate(turns):
        lo, hi = t["start"] - TURN_PAD_SEC, t["end"] + TURN_PAD_SEC
        if i > 0:
            lo = max(lo, (turns[i - 1]["end"] + t["start"]) / 2)
        if i < len(turns) - 1:
            hi = min(hi, (t["end"] + turns[i + 1]["start"]) / 2)
        # overlapping turns put the midpoint inside this one; never cut into the turn itself
        bounds.append((max(0.0, min(lo, t["start"])), min(duration, max(hi, t["end"]))))
    return bounds


def frame_rms(audio: np.ndarray, frame_sec: float = 0.03) -> np.ndarray:
    n = max(1, int(frame_sec * SAMPLE_RATE))
    usable = len(audio) // n * n
    if not usable:
        return np.sqrt(np.mean(np.square(audio), keepdims=True)) if len(audio) else np.zeros(1)
    return np.sqrt(np.mean(np.square(audio[:usable].reshape(-1, n)), axis=1))


def is_silent(audio: np.ndarray) -> bool:
    if len(audio) < MIN_SPEECH_SEC * SAMPLE_RATE:
        return True
    return float(np.percentile(frame_rms(audio), 95)) < SILENCE_RMS


def quietest_point(audio: np.ndarray, start: int, end: int, frame_sec: float = 0.25) -> int:
    """Sample index of the quietest moment in audio[start:end], to cut a window between words."""
    region = audio[start:end]
    n = int(frame_sec * SAMPLE_RATE)
    if len(region) < n:
        return end
    rms = frame_rms(region, frame_sec)
    return start + int(np.argmin(rms)) * n + n // 2


class StreamDecoder:
    """Decodes one growing audio stream (e.g. a MediaRecorder webm) to 16 kHz mono as bytes arrive.

    One ffmpeg process per stream, fed through stdin, so nothing is ever decoded twice no matter
    how long the meeting runs. Samples are kept as int16 to halve memory on long meetings."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [FFMPEG, "-loglevel", "error", "-probesize", "32768", "-analyzeduration", "0",
             "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._buf = np.zeros(SAMPLE_RATE * 60, dtype=np.int16)
        self._len = 0      # samples held in _buf
        self._base = 0     # samples dropped from the front (already processed); indices stay absolute
        self._lock = threading.Lock()
        self._errors = b""
        self._closed = False
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._err_reader = threading.Thread(target=self._read_errors, daemon=True)
        self._err_reader.start()

    def _read(self):
        leftover = b""
        while True:
            data = self.proc.stdout.read1(1 << 16)
            if not data:
                break
            data = leftover + data
            cut = len(data) // 2 * 2
            leftover = data[cut:]
            samples = np.frombuffer(data[:cut], dtype=np.int16)
            with self._lock:
                if self._len + len(samples) > len(self._buf):
                    bigger = np.zeros(max(len(self._buf) * 2, self._len + len(samples)), dtype=np.int16)
                    bigger[:self._len] = self._buf[:self._len]
                    self._buf = bigger
                self._buf[self._len:self._len + len(samples)] = samples
                self._len += len(samples)

    def _read_errors(self):
        for line in self.proc.stderr:
            self._errors = (self._errors + line)[-500:]

    def feed(self, data: bytes):
        if self._closed:
            raise RuntimeError("stream already closed")
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RuntimeError(f"audio decoder stopped: {self._errors.decode(errors='replace').strip()}")

    def close(self, timeout: float = 120):
        """End of stream: let ffmpeg flush its last samples, then wait for it."""
        if self._closed:
            return
        self._closed = True
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:  # a hung decoder: keep what it produced, don't fail the request
            self.proc.kill()
            self.proc.wait(10)
        self._reader.join(timeout)
        self._err_reader.join(timeout)
        self._close_pipes()

    def kill(self):
        self._closed = True
        try:
            self.proc.kill()
            self.proc.wait(10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._reader.join(10)
        self._err_reader.join(10)
        self._close_pipes()

    def _close_pipes(self):
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                pipe.close()
            except OSError:
                pass

    def __len__(self):
        """Samples decoded so far, including any already dropped."""
        return self._base + self._len

    def slice(self, start: int, end: int) -> np.ndarray:
        with self._lock:
            if start < self._base:
                raise ValueError(f"samples before {self._base} were already dropped")
            lo, hi = start - self._base, min(end - self._base, self._len)
            return self._buf[lo:max(lo, hi)].astype(np.float32) / 32768.0

    def drop_before(self, index: int):
        """Frees the samples before `index` (absolute); they can't be sliced afterwards. A long
        meeting then only keeps its unprocessed tail in memory, not the whole recording."""
        with self._lock:
            k = min(index - self._base, self._len)
            if k <= 0:
                return
            keep = self._buf[k:self._len]
            buf = np.zeros(max(len(keep) * 2, SAMPLE_RATE * 60), dtype=np.int16)
            buf[:len(keep)] = keep
            self._buf, self._len, self._base = buf, len(keep), self._base + k
