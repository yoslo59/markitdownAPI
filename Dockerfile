FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1

# dépendances système min pour OpenCV et PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN pip install "numpy==1.26.4"
RUN pip install "opencv-python-headless==4.9.0.80"
RUN pip install \
    "pymupdf>=1.24" \
    "Pillow>=10.2" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "python-multipart>=0.0.9" \
    "pandas>=2.2" \
    "markitdown[all]" \
    "paddlepaddle==2.6.1" \
    "paddleocr==2.7.3"

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 5704
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5704", "--workers", "1"]
