"""Download public test clips (LibriSpeech test-clean + MLS Spanish test, both CC-BY 4.0)
and build a 6-speaker en/es conversation with known speaker turns."""
import json
import subprocess
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "audio"
ROWS = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split=test&offset={off}&length=2"
SOURCES = {
    "en": ("openslr/librispeech_asr", "clean", "text", 2620),
    "es": ("facebook/multilingual_librispeech", "spanish", "transcript", 2385),
}
CLIPS_PER_LANG = 10
GAP = 0.4  # seconds of silence between conversation turns


def rows(lang, off):
    ds, cfg, text_key, _ = SOURCES[lang]
    data = json.load(urllib.request.urlopen(ROWS.format(ds=ds, cfg=cfg, off=off)))
    return [(r["row"]["audio"][0]["src"], r["row"][text_key], str(r["row"]["speaker_id"])) for r in data["rows"]]


def fetch(url, dst):
    tmp = dst.with_suffix(".src")
    tmp.write_bytes(urllib.request.urlopen(url).read())
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", tmp, "-ac", "1", "-ar", "16000", dst], check=True)
    tmp.unlink()


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


def main():
    refs, by_speaker = {}, {}
    for lang, (_, _, _, total) in SOURCES.items():
        (OUT / lang).mkdir(parents=True, exist_ok=True)
        step = total // CLIPS_PER_LANG
        for i in range(CLIPS_PER_LANG):
            pair = rows(lang, i * step)
            for j, (url, text, spk) in enumerate(pair):
                name = f"{lang}/{i:02}_{j}.wav"
                fetch(url, OUT / name)
                refs[name] = {"lang": lang, "speaker": spk, "text": text}
                by_speaker.setdefault((lang, spk), []).append(name)
            print("fetched", lang, i)

    # Conversation: 3 English + 3 Spanish speakers, 2 turns each, alternating languages.
    en = [k for k in by_speaker if k[0] == "en" and len(by_speaker[k]) >= 2][:3]
    es = [k for k in by_speaker if k[0] == "es" and len(by_speaker[k]) >= 2][:3]
    order = [s for pair in zip(en, es) for s in pair]
    turns = [(s, by_speaker[s][0]) for s in order] + [(s, by_speaker[s][1]) for s in reversed(order)]

    silence = OUT / "silence.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=16000:cl=mono", "-t", str(GAP), silence], check=True)
    concat, rttm, t = [], [], 0.0
    for (lang, spk), clip in turns:
        d = duration(OUT / clip)
        rttm.append(f"SPEAKER conversation 1 {t:.3f} {d:.3f} <NA> <NA> {lang}_{spk} <NA> <NA>")
        concat += [f"file '{clip}'", f"file 'silence.wav'"]
        t += d + GAP
    (OUT / "concat.txt").write_text("\n".join(concat) + "\n")
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", OUT / "concat.txt", "-c", "copy", OUT / "conversation.wav"], check=True)
    (OUT / "conversation.rttm").write_text("\n".join(rttm) + "\n")
    refs["conversation.wav"] = {"lang": "mixed", "text": " ".join(refs[c]["text"] for _, c in turns)}
    (OUT / "refs.json").write_text(json.dumps(refs, indent=1, ensure_ascii=False))
    print(f"conversation: {len(turns)} turns, {len(order)} speakers, {t:.0f}s")


if __name__ == "__main__":
    main()
