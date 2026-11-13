"""Send the test clips to the speech service and report WER and speed per language."""
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
import uuid
from pathlib import Path

AUDIO = Path(__file__).parent / "audio"
URL = os.environ.get("SPEECH_URL", "http://speech:8000") + "/v1/audio/transcriptions"


def post(path, **fields):
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n\r\n'.encode()
    body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(URL, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req, timeout=600))


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


def main():
    refs = json.loads((AUDIO / "refs.json").read_text())
    langs = sys.argv[1:] or ["en", "es", "mixed"]
    stats = {}
    for name, ref in refs.items():
        if ref["lang"] not in langs:
            continue
        start = time.time()
        hyp = post(AUDIO / name, response_format="json")["text"]
        elapsed = time.time() - start
        r = words(ref["text"])
        s = stats.setdefault(ref["lang"], {"err": 0, "words": 0, "secs": 0.0, "files": 0})
        s["err"] += edits(r, words(hyp))
        s["words"] += len(r)
        s["secs"] += elapsed
        s["files"] += 1
    for lang, s in stats.items():
        print(f"{lang:6} files={s['files']:3}  WER={100 * s['err'] / s['words']:5.2f}%  time={s['secs']:.1f}s")




def diarization(num_speakers=None):
    """Score speaker labels on the conversation: share of words attributed to the right speaker."""
    refs = json.loads((AUDIO / "refs.json").read_text())
    turns = [(float(f[3]), float(f[3]) + float(f[4]), f[7]) for f in (l.split() for l in (AUDIO / "conversation.rttm").read_text().splitlines())]
    fields = {"response_format": "verbose_json", "diarize": "true"}
    if num_speakers:
        fields["num_speakers"] = num_speakers
    start = time.time()
    out = post(AUDIO / "conversation.wav", **fields)
    elapsed = time.time() - start
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
    r = words(refs["conversation.wav"]["text"])
    wer = 100 * edits(r, words(out["text"])) / len(r)
    label = f"num_speakers={num_speakers}" if num_speakers else "auto"
    print(f"diarize ({label}): speakers found={len({h for _, h in pairs})} (true 6)  "
          f"words on right speaker={100 * correct / len(pairs):.1f}%  WER={wer:.2f}%  time={elapsed:.1f}s")


if __name__ == "__main__":
    if sys.argv[1:2] == ["diarize"]:
        diarization()
        diarization(num_speakers=6)
    else:
        main()
