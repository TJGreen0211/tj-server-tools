"""Test the speech service: WER per language, request latency, diarization accuracy, the effect of
browser audio compression, and streamed meeting sessions.

Run inside the speech container (needs soundfile + ffmpeg):
    python /tests/run.py [wer|latency|diarize|codec|meeting]"""
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.request
import uuid
from pathlib import Path

import soundfile as sf

AUDIO = Path(__file__).parent / "audio"
BASE = os.environ.get("SPEECH_URL", "http://speech:8000") + "/v1"
URL = BASE + "/audio/transcriptions"


def post(files, url=URL, **fields):
    """files: list of (filename, bytes). Returns parsed JSON, or text for response_format=text."""
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for name, data in files:
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n\r\n'.encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    raw = urllib.request.urlopen(req, timeout=600).read()
    return raw.decode() if fields.get("response_format") == "text" else json.loads(raw)


def file(path):
    return [(path.name, path.read_bytes())]


def words(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9' ]+", " ", text).split()


def edits(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def refs():
    return json.loads((AUDIO / "refs.json").read_text())


def wer():
    """Single-language clips through the plain (no diarization) path."""
    stats = {}
    for name, ref in refs().items():
        if ref["lang"] not in ("en", "es"):
            continue
        start = time.time()
        hyp = post(file(AUDIO / name))["text"]
        s = stats.setdefault(ref["lang"], {"err": 0, "words": 0, "secs": 0.0, "files": 0})
        r = words(ref["text"])
        s["err"] += edits(r, words(hyp))
        s["words"] += len(r)
        s["secs"] += time.time() - start
        s["files"] += 1
    for lang, s in stats.items():
        print(f"wer     {lang}  files={s['files']}  WER={100 * s['err'] / s['words']:.2f}%  time={s['secs']:.1f}s")


def latency():
    """Per-request time the way openlumara sends audio: 3 s previews and full commit segments, response_format=text."""
    clips = [AUDIO / n for n, r in refs().items() if r["lang"] in ("en", "es")]
    for label, seconds in (("preview 3s", 3), ("commit (full clip)", None)):
        times = []
        for path in clips:
            audio, rate = sf.read(path, dtype="float32")
            if seconds:
                audio = audio[: seconds * rate]
            buf = io.BytesIO()
            sf.write(buf, audio, rate, format="WAV")
            start = time.time()
            post([("audio.wav", buf.getvalue())], response_format="text")
            times.append(time.time() - start)
        print(f"latency {label:18}  n={len(times)}  median={1000 * statistics.median(times):.0f}ms  max={1000 * max(times):.0f}ms")


def score_diarization(out, label, elapsed):
    turns = [(float(f[3]), float(f[3]) + float(f[4]), f[7])
             for f in (line.split() for line in (AUDIO / "conversation.rttm").read_text().splitlines())]
    pairs = []
    for w in out["words"]:
        mid = (w["start"] + w["end"]) / 2
        ref = next((s for a, b, s in turns if a <= mid <= b), None)
        if ref:
            pairs.append((ref, w["speaker"]))
    # map each predicted speaker to the reference speaker it overlaps most
    best = {}
    for hyp in {h for _, h in pairs}:
        counts = {}
        for r, h in pairs:
            if h == hyp:
                counts[r] = counts.get(r, 0) + 1
        best[hyp] = max(counts, key=counts.get)
    correct = sum(best[h] == r for r, h in pairs)
    r = words(refs()["conversation.wav"]["text"])
    wer_pct = 100 * edits(r, words(out["text"])) / len(r)
    print(f"diarize {label:22}  speakers={len({h for _, h in pairs})} (true 6)  "
          f"right speaker={100 * correct / len(pairs):.1f}%  WER={wer_pct:.2f}%  time={elapsed:.1f}s")


def diarize():
    fields = {"response_format": "verbose_json", "diarize": "true"}
    for label, extra in (("auto", {}), ("num_speakers=6", {"num_speakers": 6})):
        start = time.time()
        out = post(file(AUDIO / "conversation.wav"), **fields, **extra)
        score_diarization(out, label, time.time() - start)

    # a paused meeting: the conversation as two webm/opus parts, split at a turn boundary
    audio, rate = sf.read(AUDIO / "conversation.wav", dtype="float32")
    cut = int(65.2 * rate)
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, chunk in enumerate((audio[:cut], audio[cut:])):
            wav, webm = Path(tmp) / f"p{i}.wav", Path(tmp) / f"p{i}.webm"
            sf.write(wav, chunk, rate)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", wav, "-c:a", "libopus", webm], check=True)
            parts.append((webm.name, webm.read_bytes()))
    start = time.time()
    out = post(parts, **fields)
    score_diarization(out, "2 webm parts", time.time() - start)


def to_webm(wav: Path, bitrate: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.webm"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", wav, "-c:a", "libopus", "-b:a", bitrate, out],
                       check=True)
        return out.read_bytes()


def codec():
    """The same conversation lossless vs. webm/opus at several bitrates, each sent as ONE file, so the
    only difference is compression. openlumara's meeting recorder uses 64k."""
    fields = {"response_format": "verbose_json", "diarize": "true"}
    wav = AUDIO / "conversation.wav"
    variants = [("wav (lossless)", file(wav))]
    variants += [(f"webm/opus {b}", [("conversation.webm", to_webm(wav, b))]) for b in ("32k", "64k", "128k")]
    for label, files in variants:
        start = time.time()
        out = post(files, **fields)
        score_diarization(out, label, time.time() - start)


def meeting():
    """Streams the conversation through a meeting session the way openlumara does (webm/opus 64k,
    sent in ~15 s pieces while it plays), then finishes it. Windows are MEETING_WINDOW_SEC long on
    the server (default 480 s); start the service with a smaller value to exercise several windows."""
    data = to_webm(AUDIO / "conversation.wav", "64k")
    piece = 64000 // 8 * 15
    sid = post([], url=BASE + "/meetings")["id"]
    start = time.time()
    for i in range(0, len(data), piece):
        status = post([("chunk.webm", data[i:i + piece])], url=f"{BASE}/meetings/{sid}/audio", part=1)
    print(f"meeting streamed     received={status['received_sec']}s  processed before finish={status['processed_sec']}s")
    finish_start = time.time()
    out = post([], url=f"{BASE}/meetings/{sid}/finish")
    print(f"meeting finish took  {time.time() - finish_start:.1f}s")
    score_diarization(out, "streamed session", time.time() - start)


if __name__ == "__main__":
    tests = {"wer": wer, "latency": latency, "diarize": diarize, "codec": codec, "meeting": meeting}
    for name in sys.argv[1:] or tests:
        tests[name]()
