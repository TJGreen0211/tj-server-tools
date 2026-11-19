# Bundled voices

Two voices ship with the service, so it can speak (and clients can offer a choice) before anything
is uploaded:

| Voice | Clip | Source voice |
|---|---|---|
| `female` (the default) | `female.wav`, 7.8 s | Kokoro `af_heart` |
| `male` | `male.wav`, 7.4 s | Kokoro `am_michael` |

Each is a 24 kHz mono reference clip, plus `<name>.txt` (exactly what is said in it) and
`<name>.json` (the description clients show when a user picks a voice).

They are synthetic voices, not recordings of real people: generated with
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) via
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), then trimmed and loudness-matched. Each
transcript was checked word for word against its clip with the speech service.

On startup the service copies every bundled voice into `/voices` (the `./voices` volume) that isn't
there yet, so an uploaded voice with the same name always wins. To go back to a bundled one, delete
the uploaded voice (`curl -X DELETE host:5004/v1/voices/male`) and restart the service.
