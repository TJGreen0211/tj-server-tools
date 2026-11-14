"""Unit tests for the meeting-session plumbing (services/speech/audio.py and meetings.py).

The GPU models are replaced by a stub, so this needs only numpy and ffmpeg:
    python tests/test_meetings.py                          (ffmpeg on PATH, or FFMPEG=/path/to/ffmpeg)
    make test-unit                                         (inside the speech container)
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# the repo checkout, or /app inside the speech container (where the tests are mounted at /tests)
for _src in (Path(__file__).resolve().parent.parent / "services" / "speech", Path("/app")):
    if (_src / "meetings.py").exists():
        sys.path.insert(0, str(_src))
        break

import audio  # noqa: E402
import meetings  # noqa: E402

SR = audio.SAMPLE_RATE


def tone_track(pattern):
    """pattern: [(seconds, on)] -> 16 kHz float32: a 220 Hz tone (speech stand-in) or silence."""
    out = []
    for secs, on in pattern:
        t = np.arange(int(secs * SR)) / SR
        out.append((0.2 * np.sin(2 * np.pi * 220 * t) if on else np.zeros_like(t)).astype(np.float32))
    return np.concatenate(out)


def to_webm(samples: np.ndarray) -> bytes:
    """Encodes like a browser MediaRecorder would: webm/opus."""
    with tempfile.TemporaryDirectory() as tmp:
        raw, out = Path(tmp) / "in.f32", Path(tmp) / "out.webm"
        samples.astype(np.float32).tofile(raw)
        subprocess.run([audio.FFMPEG, "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(raw),
                        "-c:a", "libopus", "-b:a", "64k", "-f", "webm", str(out)], check=True)
        return out.read_bytes()


def chunks(data: bytes, size: int):
    return [data[i:i + size] for i in range(0, len(data), size)]


class StubModel:
    """Stands in for process_window: one segment per window, one local speaker per window.

    The voice embedding for each window comes from `voices` (a list consumed in order), so tests
    control which windows sound like the same person."""

    def __init__(self, voices=None):
        self.calls = []
        self.voices = list(voices or [])

    def __call__(self, window, max_speakers, single_speaker):
        self.calls.append({"seconds": len(window) / SR, "single": single_speaker})
        emb = self.voices.pop(0) if self.voices else np.ones(4)
        dur = len(window) / SR
        return {
            "segments": [{"start": 0.0, "end": dur, "speaker": "L0", "text": f"window {len(self.calls)}"}],
            "words": [{"word": "w", "start": 0.0, "end": 0.1, "speaker": "L0"}],
            "speakers": {"L0": (emb, dur)},
        }


class TurnPadding(unittest.TestCase):
    def test_pads_both_sides(self):
        self.assertEqual(audio.pad_turns([{"start": 1.0, "end": 2.0}], 10.0), [(0.75, 2.25)])

    def test_never_past_midpoint_to_neighbour(self):
        b = audio.pad_turns([{"start": 1.0, "end": 2.0}, {"start": 2.2, "end": 3.0}], 10.0)
        self.assertAlmostEqual(b[0][1], 2.1)
        self.assertAlmostEqual(b[1][0], 2.1)

    def test_overlapping_turns_keep_their_own_audio(self):
        b = audio.pad_turns([{"start": 1.0, "end": 3.0}, {"start": 2.5, "end": 4.0}], 10.0)
        self.assertLessEqual(b[1][0], 2.5)
        self.assertGreaterEqual(b[0][1], 3.0)

    def test_clamped_to_audio(self):
        self.assertEqual(audio.pad_turns([{"start": 0.1, "end": 4.9}], 5.0), [(0.0, 5.0)])


class Silence(unittest.TestCase):
    def test_silence_and_short_clips(self):
        self.assertTrue(audio.is_silent(np.zeros(SR * 3, dtype=np.float32)))
        self.assertTrue(audio.is_silent(tone_track([(0.1, True)])))
        self.assertTrue(audio.is_silent(np.random.default_rng(0).normal(0, 0.001, SR * 3).astype(np.float32)))

    def test_speech_is_not_silent(self):
        self.assertFalse(audio.is_silent(tone_track([(2, False), (1, True), (2, False)])))

    def test_quietest_point_lands_in_the_gap(self):
        track = tone_track([(5, True), (1, False), (4, True)])
        cut = audio.quietest_point(track, 0, len(track))
        self.assertTrue(5 * SR <= cut <= 6 * SR, cut / SR)


class Decoder(unittest.TestCase):
    def test_streams_while_bytes_arrive(self):
        track = tone_track([(20, True), (10, False), (30, True)])
        data = to_webm(track)
        d = audio.StreamDecoder()
        pieces = chunks(data, 4096)
        half = len(pieces) // 2
        for c in pieces[:half]:
            d.feed(c)
        deadline = time.time() + 10
        while len(d) < 10 * SR and time.time() < deadline:
            time.sleep(0.05)
        self.assertGreater(len(d), 10 * SR, "decoder should produce audio before the stream ends")
        for c in pieces[half:]:
            d.feed(c)
        d.close()
        self.assertAlmostEqual(len(d) / SR, 60.0, delta=0.1)
        decoded = d.slice(0, len(d))
        self.assertFalse(audio.is_silent(decoded[: 20 * SR]))
        self.assertTrue(audio.is_silent(decoded[int(21 * SR): 29 * SR]))


class Sessions(unittest.TestCase):
    def setUp(self):
        self._saved = (meetings.WINDOW_SEC, meetings.CUT_SEARCH_SEC, meetings.MIN_DIARIZE_SEC)
        meetings.WINDOW_SEC, meetings.CUT_SEARCH_SEC, meetings.MIN_DIARIZE_SEC = 20.0, 5.0, 4.0

    def tearDown(self):
        meetings.WINDOW_SEC, meetings.CUT_SEARCH_SEC, meetings.MIN_DIARIZE_SEC = self._saved

    def _speechy(self, seconds):
        # 3 s of "speech" then 1 s of silence, repeated: windows have quiet places to cut at
        return tone_track([(3, True), (1, False)] * int(seconds // 4))

    def test_windows_cover_the_meeting_and_cut_in_silence(self):
        model = StubModel()
        s = meetings.Session(model, None, 8)
        track = self._speechy(64)
        for c in chunks(to_webm(track), 8000):
            s.append(1, c)
        result = s.finish()
        self.assertGreaterEqual(len(model.calls), 3)
        self.assertAlmostEqual(sum(c["seconds"] for c in model.calls), len(track) / SR, delta=0.1)
        # every cut but the last falls in one of the 1 s gaps (3-4 s into each 4 s block)
        for seg in result["segments"][:-1]:
            self.assertTrue(3.0 <= seg["end"] % 4 <= 4.0 or seg["end"] % 4 < 0.05, seg["end"])
        starts = [seg["start"] for seg in result["segments"]]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(starts[0], 0.0)

    def test_background_processes_before_finish(self):
        model = StubModel()
        s = meetings.Session(model, None, 8)
        executor = ThreadPoolExecutor(1)
        for c in chunks(to_webm(self._speechy(48)), 8000):
            s.append(1, c)
            s.schedule(executor)
        deadline = time.time() + 20
        while s.done == 0 and time.time() < deadline:
            time.sleep(0.05)
        self.assertGreater(s.done, 0, "windows should be processed while audio is still arriving")
        before = len(model.calls)
        s.finish()
        self.assertGreater(len(model.calls), before)
        executor.shutdown()

    def test_paused_meeting_parts_are_joined_in_order(self):
        model = StubModel()
        s = meetings.Session(model, None, 8)
        a, b = self._speechy(12), self._speechy(16)
        for c in chunks(to_webm(a), 5000):
            s.append(1, c)
        for c in chunks(to_webm(b), 5000):
            s.append(2, c)
        with self.assertRaises(meetings.SessionError):
            s.append(1, b"late")
        s.finish()
        self.assertAlmostEqual(sum(c["seconds"] for c in model.calls), (len(a) + len(b)) / SR, delta=0.1)

    def test_short_tail_is_one_turn(self):
        model = StubModel()
        s = meetings.Session(model, None, 8)
        for c in chunks(to_webm(tone_track([(2, True), (0.5, False)])), 5000):
            s.append(1, c)
        s.finish()
        self.assertEqual([c["single"] for c in model.calls], [True])

    def test_silent_windows_are_skipped(self):
        model = StubModel()
        s = meetings.Session(model, None, 8)
        s.append(1, to_webm(np.zeros(SR * 10, dtype=np.float32)))
        self.assertEqual(s.finish()["segments"], [])
        self.assertEqual(model.calls, [])

    def test_finish_twice_and_append_after_finish_fail(self):
        s = meetings.Session(StubModel(), None, 8)
        s.append(1, to_webm(self._speechy(4)))
        s.finish()
        with self.assertRaises(meetings.SessionError):
            s.finish()
        with self.assertRaises(meetings.SessionError):
            s.append(1, b"x")


class SpeakerLinking(unittest.TestCase):
    A, B, C = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0])

    def test_same_voice_keeps_its_label_across_windows(self):
        s = meetings.Session(StubModel(), None, 8)
        first = s._link({"x": (self.A, 10), "y": (self.B, 10)})
        second = s._link({"p": (self.B + 0.1 * self.C, 5), "q": (self.A * 3, 5)})
        self.assertEqual(second["q"], first["x"])
        self.assertEqual(second["p"], first["y"])
        self.assertEqual(len(s.speakers), 2)

    def test_new_voice_gets_a_new_label(self):
        s = meetings.Session(StubModel(), None, 8)
        s._link({"x": (self.A, 10)})
        names = s._link({"y": (self.C, 10)})
        self.assertEqual(names["y"], "SPEAKER_1")

    def test_one_voice_split_by_the_diarizer_maps_to_one_speaker(self):
        s = meetings.Session(StubModel(), None, 8)
        s._link({"x": (self.A, 10)})
        names = s._link({"y": (self.A, 4), "z": (self.A + 0.05 * self.B, 3)})
        self.assertEqual(names["y"], names["z"])

    def test_known_speaker_count_caps_new_speakers(self):
        s = meetings.Session(StubModel(), 2, 8)
        s._link({"x": (self.A, 10), "y": (self.B, 10)})
        names = s._link({"z": (self.C, 10)})
        self.assertIn(names["z"], ("SPEAKER_0", "SPEAKER_1"))
        self.assertEqual(len(s.speakers), 2)

    def test_end_to_end_labels_follow_the_voice(self):
        model = StubModel(voices=[self.A, self.B, self.A * 2, self.B * 2])
        s = meetings.Session(model, None, 8)
        orig = (meetings.WINDOW_SEC, meetings.CUT_SEARCH_SEC, meetings.MIN_DIARIZE_SEC)
        meetings.WINDOW_SEC, meetings.CUT_SEARCH_SEC, meetings.MIN_DIARIZE_SEC = 20.0, 5.0, 4.0
        try:
            for c in chunks(to_webm(tone_track([(3, True), (1, False)] * 18)), 8000):
                s.append(1, c)
            labels = [seg["speaker"] for seg in s.finish()["segments"]]
        finally:
            meetings.WINDOW_SEC, meetings.CUT_SEARCH_SEC, meetings.MIN_DIARIZE_SEC = orig
        self.assertEqual(labels[:4], ["SPEAKER_0", "SPEAKER_1", "SPEAKER_0", "SPEAKER_1"])


if __name__ == "__main__":
    if not os.environ.get("FFMPEG") and subprocess.run(["which", "ffmpeg"], capture_output=True).returncode:
        print("ffmpeg not found; set FFMPEG=/path/to/ffmpeg")
    unittest.main()
