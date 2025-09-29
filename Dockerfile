FROM python:3.12-slim

# Binaries utiles (OCR + tests HTTP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Dépendances Python (ajout: pymupdf, pytesseract, pillow)
RUN pip install --no-cache-dir \
    "markitdown[all]" \
    fastapi \
    uvicorn \
    python-multipart \
    openai \
    pymupdf \
    pytesseract \
    pillow

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 5704
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5704"]
