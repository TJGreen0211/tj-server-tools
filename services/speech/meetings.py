"""Meeting sessions: the recording streams in while the meeting runs and is transcribed in windows,
so finishing a meeting only leaves the last window to process.

Each window is diarized on its own, so its speaker labels are local to it. Every local speaker gets
a voice embedding, and is matched to the meeting's speakers so far by cosine similarity: a close
enough match keeps that speaker's label, otherwise it is a new speaker.

The model work is passed in as `process_window`, so this module can be tested without a GPU."""
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from audio import SAMPLE_RATE, StreamDecoder, is_silent, quietest_point

# Diarize once this much unprocessed audio has arrived. Longer windows give each speaker more
# audio to be recognised by, so labels stay consistent from window to window.
WINDOW_SEC = float(os.environ.get("MEETING_WINDOW_SEC", "480"))
# End each window at the quietest moment in its last stretch, so it never cuts through a word.
CUT_SEARCH_SEC = 60.0
# A final window shorter than this is too short to diarize reliably; it is treated as one turn.
MIN_DIARIZE_SEC = 20.0
# Cosine similarity at which two voice embeddings count as the same person. NeMo's own speaker
# check uses about 0.4; this leans slightly towards keeping people apart.
SAME_SPEAKER = float(os.environ.get("SAME_SPEAKER_SIMILARITY", "0.5"))
# Sessions nobody has touched for this long are dropped (the client fell back or went away).
IDLE_EXPIRY_SEC = 3 * 3600

# process_window(audio, max_speakers, single_speaker) ->
#   {"segments": [{start, end, speaker, text}], "words": [{word, start, end, speaker}],
#    "speakers": {local_label: (embedding, seconds_of_speech)}}
# times relative to the window start, labels local to the window
ProcessWindow = Callable[[np.ndarray, int, bool], dict]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


class SessionError(Exception):
    pass


class Session:
    def __init__(self, process_window: ProcessWindow, num_speakers: int | None, max_speakers: int):
        self.id = secrets.token_urlsafe(12)
        self.process_window = process_window
        self.num_speakers = num_speakers
        self.max_speakers = min(max_speakers, num_speakers) if num_speakers else max_speakers
        self.parts: list[tuple[int, StreamDecoder]] = []   # (part number, decoder), in order
        self.done = 0                                       # samples of the joined audio already transcribed
        self.segments: list[dict] = []
        self.words: list[dict] = []
        self.speakers: list[dict] = []                      # [{"emb": unit vector, "weight": seconds}]
        self.touched = time.time()
        self.closed = False
        self.error: str | None = None
        self._lock = threading.Lock()        # appends vs. reading the audio
        self._process_lock = threading.Lock()
        self._queued = False

    # -- audio --------------------------------------------------------------

    def append(self, part: int, data: bytes):
        """Adds bytes to recorder part `part`. A new part number starts a new stream (a resumed
        recording); parts are joined in the order they first arrive."""
        with self._lock:
            if self.closed:
                raise SessionError("meeting already finished")
            self.touched = time.time()
            if not self.parts or self.parts[-1][0] != part:
                if any(p == part for p, _ in self.parts):
                    raise SessionError(f"part {part} already ended; parts must arrive in order")
                if self.parts:
                    self.parts[-1][1].close()
                self.parts.append((part, StreamDecoder()))
            decoder = self.parts[-1][1]
        decoder.feed(data)

    def _length(self) -> int:
        return sum(len(d) for _, d in self.parts)

    def _slice(self, start: int, end: int) -> np.ndarray:
        out, base = [], 0
        for _, d in self.parts:
            n = len(d)
            lo, hi = max(start, base), min(end, base + n)
            if lo < hi:
                out.append(d.slice(lo - base, hi - base))
            base += n
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    # -- processing ---------------------------------------------------------

    def process(self, final: bool = False):
        """Transcribes every complete window that has arrived; with final=True, the rest as well."""
        with self._process_lock:
            window = int(WINDOW_SEC * SAMPLE_RATE)
            search = min(int(CUT_SEARCH_SEC * SAMPLE_RATE), window // 2)
            while True:
                total = self._length()
                remaining = total - self.done
                if remaining >= window:
                    audio = self._slice(self.done, self.done + window)
                    audio = audio[:quietest_point(audio, window - search, window)]
                elif final and remaining > 0:
                    audio = self._slice(self.done, total)
                else:
                    break
                self._process_audio(audio, self.done)
                self.done += len(audio)

    def _process_audio(self, audio: np.ndarray, start: int):
        if is_silent(audio):
            return
        single = len(audio) < MIN_DIARIZE_SEC * SAMPLE_RATE
        result = self.process_window(audio, self.max_speakers, single)
        names = self._link(result["speakers"])
        off = start / SAMPLE_RATE
        for s in result["segments"]:
            self.segments.append({**s, "start": round(s["start"] + off, 2), "end": round(s["end"] + off, 2),
                                  "speaker": names[s["speaker"]]})
        for w in result["words"]:
            self.words.append({**w, "start": round(w["start"] + off, 2), "end": round(w["end"] + off, 2),
                               "speaker": names[w["speaker"]]})

    def _link(self, local: dict) -> dict:
        """Maps this window's speaker labels onto the meeting's speakers, adding new ones as needed.
        Several local speakers may map to one meeting speaker (the diarizer split one voice)."""
        names = {}
        for label, (emb, secs) in sorted(local.items(), key=lambda kv: -kv[1][1]):
            emb = _unit(emb)
            sims = [float(emb @ s["emb"]) for s in self.speakers]
            best = int(np.argmax(sims)) if sims else None
            full = self.num_speakers is not None and len(self.speakers) >= self.num_speakers
            if best is not None and (sims[best] >= SAME_SPEAKER or full):
                s = self.speakers[best]
                s["emb"] = _unit(s["emb"] * s["weight"] + emb * secs)
                s["weight"] += secs
            else:
                self.speakers.append({"emb": emb, "weight": secs})
                best = len(self.speakers) - 1
            names[label] = f"SPEAKER_{best}"
        return names

    def schedule(self, executor: ThreadPoolExecutor):
        """Queues a background pass, unless one is already waiting."""
        with self._lock:
            if self._queued or self.closed:
                return
            self._queued = True
        executor.submit(self._background)

    def _background(self):
        with self._lock:
            self._queued = False
        try:
            self.process(final=False)
            self.error = None
        except Exception as e:  # the window stays unprocessed, so finish() retries it
            self.error = f"{type(e).__name__}: {e}"

    def finish(self) -> dict:
        with self._lock:
            if self.closed:
                raise SessionError("meeting already finished")
            self.closed = True
            decoders = [d for _, d in self.parts]
        for d in decoders:
            d.close()
        self.process(final=True)
        return self.result()

    def result(self) -> dict:
        return {"text": " ".join(s["text"] for s in self.segments), "segments": self.segments, "words": self.words}

    def status(self) -> dict:
        return {"id": self.id, "received_sec": round(self._length() / SAMPLE_RATE, 2),
                "processed_sec": round(self.done / SAMPLE_RATE, 2), "speakers": len(self.speakers),
                "error": self.error}

    def discard(self):
        with self._lock:
            self.closed = True
            for _, d in self.parts:
                d.kill()


class Registry:
    def __init__(self, process_window: ProcessWindow):
        self.process_window = process_window
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        # one background worker: windows run one at a time, and the GPU lock serializes them
        # with normal requests anyway
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meeting")

    def create(self, num_speakers: int | None, max_speakers: int) -> Session:
        self.expire()
        s = Session(self.process_window, num_speakers, max_speakers)
        with self.lock:
            self.sessions[s.id] = s
        return s

    def get(self, sid: str) -> Session | None:
        with self.lock:
            return self.sessions.get(sid)

    def pop(self, sid: str) -> Session | None:
        with self.lock:
            return self.sessions.pop(sid, None)

    def expire(self):
        now = time.time()
        with self.lock:
            stale = [sid for sid, s in self.sessions.items() if now - s.touched > IDLE_EXPIRY_SEC]
            dropped = [self.sessions.pop(sid) for sid in stale]
        for s in dropped:
            s.discard()
