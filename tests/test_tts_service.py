"""Unit tests for the TTS service plumbing (services/tts/tts_utils.py and engine.py).

The Qwen3-TTS model is replaced by a stub, so this needs only numpy, soundfile and ffmpeg:
    python tests/test_tts_service.py                       (ffmpeg on PATH, or FFMPEG=/path/to/ffmpeg)
    make test-tts-unit                                     (inside the tts container)
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

# the repo checkout, or /app inside the tts container (where the tests are mounted at /tests)
for _src in (Path(__file__).resolve().parent.parent / "services" / "tts", Path("/app")):
    if (_src / "engine.py").exists():
        sys.path.insert(0, str(_src))
        break

import engine as eng  # noqa: E402
import tts_utils as tu  # noqa: E402

SR = 24000


def tone(seconds, rate=SR):
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class StubModel:
    """Stands in for FasterQwen3TTS: 0.1 s of audio per character, streamed in 0.25 s chunks."""
    sample_rate = SR

    def __init__(self):
        self.calls = []
        self._voice_prompt_cache = {"x": 1}

    def _audio(self, text):
        return tone(0.01 * len(text))

    def generate_voice_clone(self, text, language, **kw):
        self.calls.append({"text": text, "language": language, **kw})
        return [self._audio(text)], SR

    def generate_voice_clone_streaming(self, text, language, chunk_size=12, **kw):
        self.calls.append({"text": text, "language": language, "chunk_size": chunk_size, **kw})
        audio = self._audio(text)
        step = int(0.25 * SR)
        for i in range(0, len(audio), step):
            yield audio[i:i + step], SR, {}


class Utils(unittest.TestCase):
    def test_language_names(self):
        self.assertEqual(tu.language_name("es"), "Spanish")
        self.assertEqual(tu.language_name("Spanish"), "Spanish")
        self.assertEqual(tu.language_name("english"), "English")
        self.assertEqual(tu.language_name(""), "Auto")
        self.assertEqual(tu.language_name("auto"), "Auto")
        self.assertIsNone(tu.language_name("xx"))

    def test_chunks_keep_sentences_whole(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(40))
        parts = tu.chunks(text, 100)
        self.assertTrue(all(len(p) <= 100 for p in parts))
        self.assertEqual(" ".join(parts), text)
        self.assertEqual(tu.chunks("One very long sentence " * 10, 50), [("One very long sentence " * 10).strip()])
        self.assertEqual(tu.chunks("  \n "), [])

    def test_pcm16_and_encode(self):
        audio = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        self.assertEqual(np.frombuffer(tu.pcm16(audio), "<i2").tolist(), [0, 32767, -32767, 32767])
        wav = tu.encode(tone(0.5), SR, "wav")
        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(tu.encode(tone(0.5), SR, "mp3")[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3"), True)


class Voices(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = tu.VoiceStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def clip(self, seconds, rate=44100):
        path = Path(self.tmp.name) / "upload.flac"
        sf.write(path, tone(seconds, rate), rate)
        return path.read_bytes()

    def test_add_converts_and_keeps_transcript(self):
        voice = self.store.add("ana", self.clip(8), "Hola, me llamo Ana.")
        self.assertEqual(voice.transcript, "Hola, me llamo Ana.")
        info = sf.info(str(voice.audio))
        self.assertEqual((info.samplerate, info.channels), (tu.VOICE_RATE, 1))
        self.assertFalse(voice.xvec_only("auto"))
        self.assertTrue(voice.xvec_only("xvec"))
        # re-uploading without a transcript drops the old one
        self.assertEqual(self.store.add("ana", self.clip(8)).transcript, "")
        self.assertTrue(self.store.get("ana").xvec_only("auto"))

    def test_rejects_bad_names_lengths_and_audio(self):
        with self.assertRaises(tu.VoiceError):
            self.store.add("../evil", self.clip(8))
        with self.assertRaises(tu.VoiceError):
            self.store.add("short", self.clip(1))
        with self.assertRaises(tu.VoiceError):
            self.store.add("junk", b"not audio at all")

    def test_bundled_default_voice_is_installed_once_and_never_overrides(self):
        bundled = Path(__file__).resolve().parent.parent / "services" / "tts" / "default_voice"
        if not (bundled / "default.wav").exists():
            bundled = Path("/app/default_voice")
        info = sf.info(str(bundled / "default.wav"))
        self.assertTrue(tu.MIN_REF_SEC <= info.duration <= tu.MAX_REF_SEC)
        self.assertEqual((info.samplerate, info.channels), (tu.VOICE_RATE, 1))
        self.assertTrue(self.store.seed(bundled))
        voice = self.store.get("default")
        self.assertTrue(voice.transcript.startswith("Hi there."))
        self.assertFalse(voice.xvec_only("auto"))
        # an uploaded default is never replaced
        self.store.add("default", self.clip(8), "Mine.")
        self.assertFalse(self.store.seed(bundled))
        self.assertEqual(self.store.get("default").transcript, "Mine.")

    def test_resolve_falls_back_to_default_then_errors(self):
        with self.assertRaises(tu.VoiceError):
            self.store.resolve("ana")
        self.store.add("default", self.clip(6))
        self.assertEqual(self.store.resolve("missing").name, "default")
        self.assertTrue(self.store.remove("default"))
        self.assertFalse(self.store.remove("default"))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = tu.VoiceStore(self.tmp.name)
        path = Path(self.tmp.name) / "up.wav"
        sf.write(path, tone(6), SR)
        self.store.add("default", path.read_bytes(), "Reference transcript.")
        self.model = StubModel()
        self.engine = eng.Engine(self.model, self.store, stream_chunk=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepare_validates(self):
        with self.assertRaises(eng.RequestError):
            self.engine.prepare("   ", "default", "en")
        with self.assertRaises(eng.RequestError):
            self.engine.prepare("Hello.", "default", "klingon")
        job = self.engine.prepare("Hello there.", "nobody", "es")
        self.assertEqual((job.language, job.voice.name), ("Spanish", "default"))

    def test_stream_yields_chunks_in_order_with_gaps_between_pieces(self):
        text = "First sentence here. " * 20 + "Last one."
        job = self.engine.prepare(text, "default", "en")
        self.assertGreater(len(job.pieces), 1)
        out = list(self.engine.stream(job))
        self.assertGreater(len(out), len(job.pieces))
        self.assertTrue(all(a.dtype == np.float32 for a in out))
        expected = sum(0.01 * len(p) for p in job.pieces) + eng.GAP_SEC * (len(job.pieces) - 1)
        self.assertAlmostEqual(sum(len(a) for a in out) / SR, expected, delta=0.01)
        call = self.model.calls[0]
        self.assertEqual((call["language"], call["chunk_size"], call["ref_text"], call["xvec_only"]),
                         ("English", 4, "Reference transcript.", False))
        self.assertFalse(self.engine.lock.locked())

    def test_abandoned_stream_releases_the_gpu(self):
        job = self.engine.prepare("Some words to say. " * 10, "default", "en")
        gen = self.engine.stream(job)
        next(gen)
        self.assertTrue(self.engine.lock.locked())
        gen.close()                  # a client that disconnects
        self.assertFalse(self.engine.lock.locked())

    def test_requests_are_serialized(self):
        job = self.engine.prepare("Words. " * 30, "default", "en")
        gen = self.engine.stream(job)
        next(gen)
        done = threading.Event()
        threading.Thread(target=lambda: (self.engine.synthesize(job), done.set()), daemon=True).start()
        self.assertFalse(done.wait(0.3), "a second request must wait for the GPU")
        list(gen)
        self.assertTrue(done.wait(5))

    def test_synthesize_matches_stream_length(self):
        job = self.engine.prepare("One. Two three. " * 30, "default", "en")
        whole = self.engine.synthesize(job)
        streamed = np.concatenate(list(self.engine.stream(job)))
        self.assertEqual(len(whole), len(streamed))

    def test_xvec_mode_and_voice_changes(self):
        self.engine.clone_mode = "xvec"
        list(self.engine.stream(self.engine.prepare("Hi.", "default", "en")))
        self.assertTrue(self.model.calls[-1]["xvec_only"])
        self.engine.voices_changed()
        self.assertEqual(self.model._voice_prompt_cache, {})


if __name__ == "__main__":
    unittest.main()
