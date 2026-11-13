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
| `file` | required | Any format ffmpeg reads |
| `response_format` | `json` | `json`, `text`, `verbose_json` (segments + words with timestamps), `srt`, `vtt` |
| `diarize` | `false` | Adds `speaker` to segments and words |
| `num_speakers` | none | Set it when you know the count; improves accuracy |
| `max_speakers` | `8` | Upper bound when the count is unknown |

The audio is always split into speaker turns first and each turn is transcribed separately. Parakeet locks onto one language per stretch of audio, so this keeps English and Spanish turns from dropping each other. A single speaker switching languages mid-sentence is still a weak spot.

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
```

Results on the 7900 XTX:

| Test | Result |
|---|---|
| English, 20 clips | 1.94% WER |
| Spanish, 20 clips | 3.30% WER |
| Mixed en/es conversation, 6 speakers | 2.41% WER, 6 speakers found, 100% of words on the right speaker, ~30x real time |
| TTS, en/es, built-in and cloned voice | Parakeet reads it back at ~4% WER; generation is about real time |

The conversation is clean read speech with no cross-talk, so real meetings will score lower.

## Notes

- **MIOpen is disabled** in both services. It compiles GPU kernels for every new input shape, which took minutes per request with variable-length audio, and its LSTM failed on first use. PyTorch's own ROCm kernels are as fast here.
- `speech` runs Python 3.12 (a NeMo dependency doesn't support 3.14 yet); `tts` runs 3.14 because Chatterbox pins torch 2.6 on older Pythons.
- NVIDIA's `cuda-bindings` package is removed from the speech image; with it installed NeMo crashes looking for an NVIDIA driver.
- Each service handles one request at a time on the GPU; others queue.
- There is no authentication. The ports bind to `0.0.0.0`, so put the services behind a proxy or firewall.
