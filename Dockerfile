FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
RUN pip install --no-cache-dir \
    markitdown \
    mammoth \
    fastapi \
    uvicorn \
    python-multipart \
    pymupdf \
    pillow

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 5704
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5704", "--workers", "4"]
