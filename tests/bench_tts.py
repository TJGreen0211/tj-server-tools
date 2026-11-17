"""Benchmark the TTS service the way a live player uses it: streamed PCM, measured as it arrives.

Standard library only, so it runs from any machine that can reach the service:
    python tests/bench_tts.py http://tj-server-1:5004
    python tests/bench_tts.py http://tj-server-1:5004 --voice ana --runs 5
    python tests/bench_tts.py http://host:5004 --upload default --clip me.wav --text "What I say in the clip."

For each text it reports:
  first audio   time until the first audio bytes arrive (what the listener waits for)
  speed         seconds of audio per second of generation (above 1.0 = faster than real time)
  stall         the longest pause a player would hit if it started on the first chunk
                (0.00 = playback never waits for the next chunk)
"""
import argparse
import json
import statistics
import time
import urllib.request
import uuid

TEXTS = {
    "en short": "Sure, I can help with that.",
    "en medium": ("A sprint retrospective is a regular meeting held at the end of each sprint, where the team "
                  "looks back at how the work went and agrees on one or two concrete improvements."),
    "en long": ("Typically, the discussion covers three areas. What went well, so the team can keep doing it. "
                "What didn't go well, so it can be fixed. And what the team will try differently next time. "
                "The facilitator collects everyone's input, groups similar points together, and the team agrees "
                "on one or two concrete improvements to carry into the next sprint, with an owner for each one."),
    "es short": "Claro, te ayudo con eso.",
    "es medium": ("La retrospectiva es una reunión corta al final de cada sprint, en la que el equipo revisa cómo "
                  "fue el trabajo y acuerda una o dos mejoras concretas para el siguiente."),
}


def stream_once(base, text, language, voice, chunk_bytes=4096):
    body = json.dumps({"input": text, "language": language, "voice": voice, "response_format": "pcm"}).encode()
    req = urllib.request.Request(base + "/v1/audio/speech", body, {"Content-Type": "application/json; charset=utf-8"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        rate = int(resp.headers.get("X-Sample-Rate", "24000"))
        bytes_per_sec = rate * 2
        first = None
        received = 0
        stall = 0.0
        while True:
            data = resp.read1(chunk_bytes) if hasattr(resp, "read1") else resp.read(chunk_bytes)
            if not data:
                break
            now = time.perf_counter() - t0
            if first is None:
                first = now
            # a player that started at `first` has played (now - first) s; it stalls if the audio
            # received before this chunk ran out earlier than that
            played_by_now = received / bytes_per_sec
            stall = max(stall, (now - first) - played_by_now)
            received += len(data)
    total = time.perf_counter() - t0
    audio = received / bytes_per_sec
    return {"first": first or total, "total": total, "audio": audio, "speed": audio / total if total else 0,
            "stall": max(0.0, stall)}


def upload(base, name, clip, text):
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="name"\r\n\r\n{name}\r\n'.encode(),
             f'--{boundary}\r\nContent-Disposition: form-data; name="text"\r\n\r\n{text}\r\n'.encode(),
             f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip"\r\n\r\n'.encode()
             + open(clip, "rb").read() + b"\r\n",
             f"--{boundary}--\r\n".encode()]
    req = urllib.request.Request(base + "/v1/voices", b"".join(parts),
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    print("uploaded:", urllib.request.urlopen(req, timeout=120).read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="e.g. http://tj-server-1:5004")
    ap.add_argument("--voice", default="default")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--upload", metavar="NAME", help="upload --clip as this voice first")
    ap.add_argument("--clip", help="reference clip for --upload (about 10 s of clean speech)")
    ap.add_argument("--text", default="", help="transcript of --clip")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print("health:", urllib.request.urlopen(base + "/health", timeout=10).read().decode())
    if args.upload:
        upload(base, args.upload, args.clip, args.text)
        args.voice = args.upload

    stream_once(base, "Warm up.", "en", args.voice)
    print(f"\n{'text':10} {'chars':>5} {'first audio':>12} {'speed':>7} {'stall':>7} {'audio':>7}   (median of {args.runs})")
    for label, text in TEXTS.items():
        runs = [stream_once(base, text, label[:2], args.voice) for _ in range(args.runs)]
        med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        worst_stall = max(r["stall"] for r in runs)
        print(f"{label:10} {len(text):5d} {med['first']:10.2f} s {med['speed']:6.2f}x "
              f"{worst_stall:6.2f}s {med['audio']:6.1f}s")


if __name__ == "__main__":
    main()
