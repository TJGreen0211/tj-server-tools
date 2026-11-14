# tj-server-tools

Self-hosted speech services for an AMD GPU (built and tested on an RX 7900 XTX), packaged as Docker Compose.

| Service | Model | Port | API |
|---|---|---|---|
| `speech` | Parakeet TDT 0.6B v3 + NeMo clustering diarizer | 5003 | `POST /v1/audio/transcriptions` |
| `tts` | Chatterbox Multilingual (voice cloning) | 5004 | `POST /v1/audio/speech`, `/v1/voices` |

Both follow the OpenAI audio API shape, with a few extra fields. English and Spanish are tested; Parakeet covers 25 European languages and Chatterbox 23 languages.

## Install

Host needs only Docker with the Compose plugin and the `amdgpu` kernel driver. ROCm, PyTorch and the models live in the containers.

```sh
git clone <repo> tj-server-tools && cd tj-server-tools
cp .env.example .env      # set RENDER_NODE and group IDs for your machine
make build
make gpu-check            # should list exactly one GPU
make up
```

The first start downloads about 5 GB of models into `./models`. Nothing needs an account or token.

### Picking the GPU

`.env` maps one GPU into the containers, so other GPUs on the host are not touched.

```sh
ls -l /dev/dri/by-path/            # pci-0000:13:00.0-render -> ../renderD130
lspci | grep -iE "vga|display"     # match the PCI address to the card
getent group video render          # VIDEO_GID / RENDER_GID
```

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

```sh
curl localhost:5004/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Hola, ¿qué tal?", "language": "es", "voice": "default"}' -o out.wav
```

| Field | Default | Notes |
|---|---|---|
| `input` | required | Long text is split into sentences and joined |
| `language` | `en` | `en`, `es`, and the other Chatterbox languages |
| `voice` | `default` | A saved voice name; unknown names use the built-in voice |
| `response_format` | `wav` | `wav`, `mp3`, `flac`, `opus` |
| `exaggeration`, `cfg_weight`, `temperature` | `0.5`, `0.5`, `0.8` | Chatterbox controls for expressiveness and pacing |

### Cloning a voice

Upload about 10 seconds of clean speech, then use its name as `voice`. Clips are stored in `./voices`. Only clone voices you have permission to use; Chatterbox watermarks its output.

```sh
curl -F name=alice -F file=@alice.wav localhost:5004/v1/voices
curl localhost:5004/v1/voices
```

## Testing

```sh
make test-audio   # public clips: LibriSpeech test-clean + MLS Spanish (CC-BY 4.0), plus a 6-speaker en/es conversation
make test         # needs `make up`
make test-unit    # meeting-session plumbing with the models stubbed out; no GPU needed
```

`make test` runs every mode of `tests/run.py`. To run a subset, name the modes (`wer latency diarize codec meeting`): `docker compose run --rm --no-deps -v ./tests:/tests speech python /tests/run.py codec meeting`. `codec` sends the conversation once as lossless WAV and once each as single-file webm/opus at 32k, 64k and 128k, to show what browser compression costs on its own. `meeting` streams it through a meeting session in 15 s pieces.

Results on the 7900 XTX:

| Test | Result |
|---|---|
| English, 20 clips | 1.94% WER |
| Spanish, 20 clips | 3.45% WER |
| Request latency (`diarize=false`) | 3 s clip: ~100 ms median; 10–20 s clip: ~150 ms median |
| Mixed en/es conversation, 6 speakers, `diarize=true` | 2.41% WER, 6 speakers found, 100% of words on the right speaker, ~30x real time |
| Same conversation sent as 2 webm/opus parts | 4.83% WER, labels still 100% correct |
| TTS, en/es, built-in and cloned voice | Parakeet reads it back at ~4% WER; generation is about real time |

The conversation is clean read speech with no cross-talk, so real meetings will score lower.

## Notes

- **MIOpen is disabled** in both services. It compiles GPU kernels for every new input shape, which took minutes per request with variable-length audio, and its LSTM failed on first use. PyTorch's own ROCm kernels are as fast here.
- `speech` runs Python 3.12 (a NeMo dependency doesn't support 3.14 yet); `tts` runs 3.14 because Chatterbox pins torch 2.6 on older Pythons.
- NVIDIA's `cuda-bindings` package is removed from the speech image; with it installed NeMo crashes looking for an NVIDIA driver.
- Each service handles one request at a time on the GPU; others queue.
- There is no authentication. The ports bind to `0.0.0.0`, so put the services behind a proxy or firewall.
