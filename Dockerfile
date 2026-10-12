# Base image ships torch + CUDA, so torch is not reinstalled via pip
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Point the HF cache at Runpod's cached-model / network-volume path
# (falls back to container disk when no volume is attached)
ENV HF_HOME=/runpod-volume/huggingface-cache \
    PYTHONUNBUFFERED=1

# git is needed to pip-install transformers from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
