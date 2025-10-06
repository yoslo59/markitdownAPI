FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "markitdown[all]" \
    fastapi \
    uvicorn \
    python-multipart \
    pymupdf \
    pillow \
    opencv-python-headless \
    pandas \
    numpy \
    paddlepaddle \
    paddleocr

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 5704
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5704"]
