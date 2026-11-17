# Bundled default voice

`default.wav` (7.8 s, 24 kHz mono) and `default.txt` (exactly what is said in it) are the voice the
TTS service uses until you upload your own `default` voice.

It is a synthetic voice, not a recording of a real person: generated with
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0), voice `af_heart`, via
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), then trimmed and normalized. The
transcript was checked word for word against the clip with the speech service.

On startup the service copies these into `/voices` (the `./voices` volume) only if no `default`
voice exists there, so an uploaded `default` always wins. To go back to this one, delete the
uploaded voice (`curl -X DELETE host:5004/v1/voices/default`) and restart the service.
