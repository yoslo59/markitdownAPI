FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Versions stables et alignées (CPU)
# - paddlepaddle==2.6.x est OK avec paddleocr>=2.7
RUN pip install \
    "paddlepaddle==2.6.1" \
    "paddleocr==2.7.3" \
    "opencv-python-headless>=4.8" \
    "pymupdf>=1.24" \
    "Pillow>=10.2" \
    "numpy>=1.26" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "python-multipart>=0.0.9" \
    "pandas>=2.2" \
    "markitdown[all]" 

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 5704
# Tu peux ajuster les workers dans le compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5704", "--workers", "4"]
