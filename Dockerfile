FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN pip install "numpy==1.26.4"
RUN pip install "opencv-python-headless==4.9.0.80"

RUN pip install \
    "paddlepaddle==2.6.1" \
    "paddleocr==2.7.3" \
    "pymupdf==1.24.10" \
    "pillow>=10.3.0" \
    "pandas>=2.2.2" \
    "openpyxl>=3.1.2" \
    "python-docx>=0.8.11" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    "markitdown>=0.1.4"

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 5704

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5704", "--workers", "1"]
