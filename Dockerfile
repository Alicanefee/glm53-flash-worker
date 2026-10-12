# torch + CUDA hazır gelen imaj: torch'u pip ile tekrar kurmuyoruz (build hızlı, sürücü uyumlu)
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Model cache'i Runpod network volume / cached model yoluna yönlendir
# (volume yoksa container diskine iner)
ENV HF_HOME=/runpod-volume/huggingface-cache \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U -r /app/requirements.txt

COPY rp_handler.py /app/rp_handler.py

CMD ["python", "-u", "rp_handler.py"]
