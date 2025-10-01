FROM python:3.12-slim

# Installer dépendances système (ffmpeg éventuellement nécessaire pour certaines conversions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Installer dépendances Python (MarkItDown + FastAPI + PaddleOCR et ses libs)
RUN pip install --no-cache-dir \
    "markitdown[all]" \
    fastapi \
    uvicorn \
    python-multipart \
    pymupdf \
    Pillow \
    opencv-python-headless \
    pandas \
    paddlepaddle \
    paddleocr

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 5704
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5704"]
