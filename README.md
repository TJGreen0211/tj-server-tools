# tj-server-tools

Self-hosted speech services for an NVIDIA GPU (CUDA 12.8 images; ported from an AMD ROCm build), packaged as Docker Compose.

| Service | Model | Port | API |
|---|---|---|---|
| `speech` | Parakeet TDT 0.6B v3 + NeMo clustering diarizer | 5003 | `POST /v1/audio/transcriptions` |
| `tts` | Qwen3-TTS 0.6B via faster-qwen3-tts (voice cloning, streaming) | 5004 | `POST /v1/audio/speech`, `/v1/voices` |

Both follow the OpenAI audio API shape, with a few extra fields. English and Spanish are tested; Parakeet covers 25 European languages; Qwen3-TTS speaks 10 (English, Spanish, German, French, Italian, Portuguese, Russian, Chinese, Japanese, Korean).

## Install

Host needs Docker with the Compose plugin, an NVIDIA driver, and the NVIDIA Container Toolkit. CUDA, PyTorch and the models live in the containers.

```sh
git clone <repo> tj-server-tools && cd tj-server-tools
cp .env.example .env      # set ports if the defaults are taken
make build
make gpu-check            # should list exactly one GPU
make up
```

The first start downloads about 5 GB of models into `./models`. Nothing needs an account or token.

### Picking the GPU

The compose file pins both services to one NVIDIA GPU (`NVIDIA_DEVICE_ID` in `.env`, default `0`), so non-NVIDIA GPUs on the host are not touched. Both services share the pinned card and each handles one request at a time; on a multi-GPU host leave them on the same GPU or the per-card memory budget no longer holds.

## Speech to text

```sh
# plain transcript
curl -F file=@meeting.mp3 localhost:5003/v1/audio/transcriptions

# with speakers
curl -F file=@meeting.mp3 -F diarize=true -F response_format=text localhost:5003/v1/audio/transcriptions
# [00:00:01] SPEAKER_0: Hola, ¿cómo estás?
# [00:00:05] SPEAKER_1: Good, thanks. And you?
```

| Field | Default | Notes |
|---|---|---|
| `file` | required | Any format ffmpeg reads. Repeat the field to send several parts (e.g. a paused recording); they are joined in order |
| `response_format` | `json` | `json`, `text`, `verbose_json` (segments + words with timestamps), `srt`, `vtt` |
| `diarize` | `false` | Adds `speaker` to segments and words |
| `num_speakers` | none | Set it when you know the count; improves accuracy |
| `max_speakers` | `8` | Upper bound when the count is unknown |

Without `diarize`, the whole clip goes straight to Parakeet (about 100 ms for a 3 s clip), which suits dictation and live previews. With `diarize=true`, the audio is split into speaker turns and each turn is transcribed separately. Parakeet locks onto one language per stretch of audio, so a recording that mixes English and Spanish should use `diarize=true`, or its minority language can be dropped. A single speaker switching languages mid-sentence is a weak spot either way.

Without `diarize`, audio is padded with silence to a whole number of seconds. The GPU picks its kernels the first time it sees an input length, which adds 0.5-0.9 s to that request, and live captions send a new length every time. Startup runs each length up to `STT_WARM_SEC` (default 60) once in the background, so captions stay at about 0.2 s from the first one. Set it to `0` to skip the warm-up.

Each speaker turn is cut with 0.25 s of padding on both sides (never past the midpoint to the next turn), so the diarizer's tight boundaries don't clip first and last syllables. Clips with no speech (including whole requests) return empty text instead of a made-up word.

### Meeting sessions

For a live meeting, stream the recording while it happens instead of uploading it at the end. The service transcribes it in windows as it arrives, so finishing only has the last window left to process.

```sh
id=$(curl -s -X POST localhost:5003/v1/meetings | jq -r .id)
curl -F part=1 -F file=@chunk1.webm localhost:5003/v1/meetings/$id/audio   # repeat, in order
curl -X POST localhost:5003/v1/meetings/$id/finish                         # verbose_json with speakers
```

| Endpoint | Notes |
|---|---|
| `POST /v1/meetings` | Starts a session. Optional `num_speakers`, `max_speakers` (as above). Returns `{"id"}` |
| `POST /v1/meetings/{id}/audio` | Fields `file` (the next piece of the stream) and `part`. Pieces of one part form one continuous stream (e.g. MediaRecorder timeslices, which aren't playable on their own); a new `part` number starts a new stream, for a recording resumed after a pause. Parts are joined in order. Returns progress |
| `GET /v1/meetings/{id}` | Progress: `received_sec`, `processed_sec`, `speakers`, last background `error` |
| `POST /v1/meetings/{id}/finish` | Transcribes the rest and returns the meeting like `diarize=true` (`response_format` as above). Ends the session |
| `DELETE /v1/meetings/{id}` | Drops a session. Idle sessions are dropped after 3 hours anyway |

Each window is diarized on its own, then its speakers are matched to the meeting's speakers so far by voice (TitaNet embeddings, cosine similarity): a close match keeps the label, otherwise it's a new speaker. That is less exact than one pass over the whole recording, mostly for people who say very little in a window. A final window under 20 s is treated as one speaker turn.

| Environment variable | Default | Notes |
|---|---|---|
| `MEETING_WINDOW_SEC` | `480` | Window length. Each window ends at the quietest moment in its last minute. Longer windows give each speaker more audio to be recognised by |
| `SAME_SPEAKER_SIMILARITY` | `0.5` | Voice similarity at which two windows' speakers count as the same person. Lower merges more, higher splits more |
| `SILENCE_RMS` | `0.005` | Loudness below which a clip counts as silent (full scale = 1.0) |

## Text to speech

Every voice is a clone: Qwen3-TTS has no built-in voice. The repo ships two synthetic voices, `female` (the default) and `male` (`services/tts/bundled_voices/`, generated with Kokoro-82M, Apache-2.0). On startup each is installed into `./voices` unless a voice with that name is already there (see below).

The voices in `./voices` are the ones the service offers. `GET /v1/voices` advertises them, default first, so clients (openlumara's voice picker) show exactly these:

```json
{"voices": ["female", "male"], "default": "female",
 "details": [{"name": "female", "description": "Female, warm American English (default)", "transcript": true},
             {"name": "male", "description": "Male, calm American English", "transcript": true}],
 "with_transcript": ["female", "male"]}
```

```sh
# streamed: raw 16-bit mono PCM at the X-Sample-Rate header's rate (24 kHz), sent as it is generated
curl localhost:5004/v1/audio/speech -H 'Content-Type: application/json'   -d '{"input": "Hola, ¿qué tal?", "language": "es", "response_format": "pcm"}' -o out.pcm

# a complete file
curl localhost:5004/v1/audio/speech -H 'Content-Type: application/json'   -d '{"input": "Hello there.", "language": "en", "response_format": "mp3"}' -o out.mp3
```

| Field | Default | Notes |
|---|---|---|
| `input` | required | Long text is generated a few sentences at a time (at most 300 characters each), with a short pause between |
| `language` | `auto` | ISO code (`en`, `es`, ...), a name (`English`, `Spanish`, ...) or `auto` |
| `voice` | `female` | A voice from `GET /v1/voices`; unknown names use the default voice (`TTS_DEFAULT_VOICE`) |
| `response_format` | `wav` | `pcm` (always streamed), `wav` (streamed with `"stream": true`), `mp3`, `flac`, `opus` |
| `stream` | `false` | Stream a `wav` response as it is generated (the header declares an unknown length) |
| `speed` | `1.0` | 0.25-4.0. Changes the pace without changing the pitch (WSOLA time-stretch, applied as the audio streams, about 50 ms held back). Up to 2 the speech transcribes as accurately as the original (English and Spanish); at 2.5 words start to blur |

For live playback use `pcm` and play chunks as they arrive: the first audio comes after the first chunk is generated, not after the whole text.

One request runs on the GPU at a time, and the service doesn't rely on clients to behave: a client that disconnects (even before the first audio) stops its generation within one chunk, one that stalls without reading is cut off at `TTS_MAX_GEN_SEC`, waiting for the GPU is bounded (503 when busy, see `/health` for `busy_sec` and `waiting`), and if the model itself ever hangs, a watchdog exits the process so Docker restarts it.

### Cloning a voice

Adding and removing voices is for whoever runs the service; clients only pick from what `GET /v1/voices` lists. Upload 3-30 seconds of clean speech (about 10 s is ideal) and, ideally, `text`: what is said in the clip. With the transcript the clone sounds closest to the speaker; without it only the voice's embedding is used, which is a looser match. `description` is what clients show next to the name. Clips are stored in `./voices` as `<name>.wav` plus `<name>.txt` and `<name>.json`. Only clone voices you have permission to use.

```sh
curl -F name=alice -F file=@alice.wav -F text="What is said in the clip." -F description="Alice, British English" localhost:5004/v1/voices
curl localhost:5004/v1/voices
curl -X DELETE localhost:5004/v1/voices/alice
```

| Environment variable | Default | Notes |
|---|---|---|
| `TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` sounds better but is slower |
| `TTS_DTYPE` | `auto` | `bfloat16` on Ampere and newer, `float16` on Turing (RTX 20xx). Use `float32` if the startup log reports invalid or silent warm-up audio |
| `TTS_DEFAULT_VOICE` | `female` | The voice used when a request names none, or an unknown one; listed first by `GET /v1/voices` |
| `TTS_CLONE_MODE` | `auto` | `auto`: full cloning when the voice has a transcript. `xvec`: embedding only for every voice; try it if a cloned voice carries its accent into another language |
| `TTS_STREAM_CHUNK` | `8` | Codec steps per streamed chunk (12 = 1 s of audio). Lower gives earlier first audio but more decoding overhead |
| `TTS_MAX_INPUT_CHARS` | `4000` | Longer requests get a 400 |
| `TTS_MAX_GEN_SEC` | `300` | A generation stops at this point whatever the client does, so no request can hold the GPU longer |
| `TTS_WAIT_SEC` | `60` | Longest a request waits for the GPU; after that it gets a 503 with `Retry-After` |
| `TTS_MAX_WAITING` | `8` | When this many requests are already waiting, new ones get a 503 at once |

Startup captures CUDA graphs and, once the default voice exists, runs one warm-up sentence, so the first real request isn't the slow one.

## Testing

```sh
make test-audio   # public clips: LibriSpeech test-clean + MLS Spanish (CC-BY 4.0), plus a 6-speaker en/es conversation
make test         # needs `make up`
make test-unit    # meeting-session plumbing with the models stubbed out; no GPU needed
make test-tts-unit   # TTS service plumbing (voices, chunking, streaming, GPU lock) with the model stubbed out
python tests/bench_tts.py http://host:5004   # live TTS: time to first audio, speed vs real time, playback stalls
```

`make test` runs every mode of `tests/run.py`. To run a subset, name the modes (`wer latency diarize codec meeting`): `docker compose run --rm --no-deps -v ./tests:/tests speech python /tests/run.py codec meeting`. `codec` sends the conversation once as lossless WAV and once each as single-file webm/opus at 32k, 64k and 128k, to show what browser compression costs on its own. `meeting` streams it through a meeting session in 15 s pieces.

Results on the 7900 XTX (AMD build; numbers will differ on NVIDIA):

| Test | Result |
|---|---|
| English, 20 clips | 1.94% WER |
| Spanish, 20 clips | 3.45% WER |
| Request latency (`diarize=false`) | 3 s clip: ~100 ms median; 10–20 s clip: ~150 ms median |
| Mixed en/es conversation, 6 speakers, `diarize=true` | 2.41% WER, 6 speakers found, 100% of words on the right speaker, ~30x real time |
| Same conversation sent as 2 webm/opus parts | 4.83% WER, labels still 100% correct |

The conversation is clean read speech with no cross-talk, so real meetings will score lower.

## Notes

- Both services run Python 3.12 (a NeMo dependency doesn't support 3.14 yet; faster-qwen3-tts needs transformers 5).
- Both services share one GPU. On an 8 GB card, check `nvidia-smi` with both loaded; the 0.6B TTS model leaves the most room for meeting diarization.
- The CUDA graph decoder stays off in `speech`: audio length varies per request, so graph capture buys little.
- Each service handles one request at a time on the GPU; others queue.
- There is no authentication. The ports bind to `0.0.0.0`, so put the services behind a proxy or firewall.
