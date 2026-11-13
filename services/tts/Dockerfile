FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/rocm7.2

ENV HF_HOME=/models/hf TORCH_HOME=/models/torch HOME=/models/tts-home
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
