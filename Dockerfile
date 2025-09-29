FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg tesseract-ocr curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "markitdown[all]" fastapi uvicorn python-multipart openai

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 5704
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5704"]
