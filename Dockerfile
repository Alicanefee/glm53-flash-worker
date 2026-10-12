FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (bazı HF/torch senaryolarında lazım olabiliyor)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
