"""Changes the speaking rate without changing the pitch (WSOLA), a chunk at a time, so streamed
speech can be sped up or slowed down as it is generated.

WSOLA cuts the audio into overlapping 40 ms frames and lays them out closer together (faster) or
further apart (slower). Each frame is nudged by up to 10 ms to where its waveform best continues
the previous one, which keeps voices clean instead of buzzy.
"""
import numpy as np

MIN_SPEED, MAX_SPEED = 0.25, 4.0


class TimeStretch:
    """speed 1.5 plays 1.5x as fast. Feed audio to process() as it arrives and call flush() once
    at the end. The output is the same however the input is split into chunks, and its total
    length is round(input length / speed). About 50 ms of audio is held back between chunks."""

    def __init__(self, speed: float, sample_rate: int, frame_sec: float = 0.04, tolerance_sec: float = 0.01):
        self.speed = float(speed)
        self.n = max(4, int(frame_sec * sample_rate) // 2 * 2)   # frame length
        self.hs = self.n // 2                                     # output step between frames
        self.ha = self.hs * self.speed                            # input step between frames
        self.tol = max(1, int(tolerance_sec * sample_rate))       # how far a frame may be nudged
        # periodic Hann: the halves of overlapping frames add up to exactly 1
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n) / self.n)).astype(np.float32)
        # starts with half a frame of silence, so the first word isn't faded in; the output
        # that silence becomes is dropped again (see _emit)
        self.buf = np.zeros(self.hs, np.float32)
        self.base = 0            # input index of buf[0]
        self.k = 0               # next frame
        self.prev = None         # input position of the previous frame
        self.acc = np.zeros(self.n, np.float32)   # overlap-add in progress; acc[0] is output k * hs
        self.fed = 0             # input samples received
        self.emitted = -self.hs  # output samples returned so far, counting the dropped lead-in

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self.fed += len(audio)
        self.buf = np.concatenate([self.buf, audio])
        return self._emit(self._frames(self.base + len(self.buf)))

    def flush(self) -> np.ndarray:
        """The rest of the audio, padded or trimmed to the exact output length."""
        real_end = self.hs + self.fed
        pad = 2 * self.n + 2 * self.tol + int(self.ha) + 1
        self.buf = np.concatenate([self.buf, np.zeros(pad, np.float32)])
        parts = self._frames(self.base + len(self.buf), stop_at=real_end)
        parts.append(self.acc[: self.hs].copy())
        before = max(self.emitted, 0)
        out = self._emit(parts)
        want = max(0, round(self.fed / self.speed) - before)
        if len(out) >= want:
            return out[:want]
        return np.concatenate([out, np.zeros(want - len(out), np.float32)])

    def _frames(self, end: int, stop_at: int | None = None) -> list:
        """Lays out every frame whose input is available (input indices below `end`)."""
        parts = []
        n, hs, tol = self.n, self.hs, self.tol
        while True:
            c = round(self.k * self.ha)            # where this frame would start with no nudging
            if stop_at is not None and c >= stop_at:
                break
            need = c + tol + n
            if self.prev is not None:
                need = max(need, self.prev + hs + n)
            if need > end:
                break
            if self.prev is None:
                p = c
            else:
                # the audio that naturally follows the previous frame; pick the nearby frame
                # that matches it best
                t0 = self.prev + hs - self.base
                template = self.buf[t0: t0 + n]
                lo = max(c - tol, self.base)
                seg = self.buf[lo - self.base: c + tol + n - self.base]
                corr = np.correlate(seg, template, mode="valid")
                p = lo + int(np.argmax(corr)) if corr.any() else c
            self.acc += self.win * self.buf[p - self.base: p - self.base + n]
            parts.append(self.acc[:hs].copy())
            self.acc = np.concatenate([self.acc[hs:], np.zeros(hs, np.float32)])
            self.prev = p
            self.k += 1
        # drop input that no later frame can use
        keep = min(round(self.k * self.ha) - tol, (self.prev if self.prev is not None else 0) + hs)
        if keep > self.base:
            self.buf = self.buf[keep - self.base:]
            self.base = keep
        return parts

    def _emit(self, parts: list) -> np.ndarray:
        out = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        skip = min(len(out), max(0, -self.emitted))   # the lead-in silence from __init__
        self.emitted += len(out)
        return out[skip:]
