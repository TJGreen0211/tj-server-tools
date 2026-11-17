.PHONY: build up down logs gpu-check test-audio test test-unit

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

# Confirm the containers see exactly one GPU.
gpu-check:
	docker compose run --rm --no-deps speech python -c "import torch; \
	print('cuda', torch.version.cuda, '| devices', torch.cuda.device_count()); \
	[print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

# Download public test clips (LibriSpeech + MLS Spanish, CC-BY 4.0) into tests/audio.
test-audio:
	docker compose run --rm --no-deps --user $$(id -u):$$(id -g) -v ./tests:/tests -w /tests speech python fetch_audio.py

# WER per language, request latency and diarization accuracy against the running speech service.
test:
	docker compose run --rm --no-deps -v ./tests:/tests speech python /tests/run.py

# Meeting-session plumbing (decoder, windows, speaker linking); no GPU or running service needed.
test-unit:
	docker compose run --rm --no-deps -v ./tests:/tests speech python /tests/test_meetings.py
