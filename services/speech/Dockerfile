FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

ENV HF_HOME=/models/hf TORCH_HOME=/models/torch NEMO_CACHE_DIR=/models/nemo
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py diar_config.yaml .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
