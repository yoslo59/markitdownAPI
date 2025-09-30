FROM python:3.12-slim

# OS deps (OCR + OpenCV + fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    libgl1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps
# - markitdown[all] : plugins DOCX/XLSX/HTML
# - pandas+tabulate : tableaux Markdown (xlsx/csv + tables ASCII)
# - pdfplumber (optionnel) : extraction tables PDF si besoin
# - opencv-python-headless : prétraitement OCR (binarisation/deskew)
RUN pip install --no-cache-dir \
    "markitdown[all]" \
    fastapi \
    uvicorn \
    python-multipart \
    openai \
    pymupdf \
    pytesseract \
    pillow \
    opencv-python-headless \
    pandas \
    tabulate \
    pdfplumber

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 5704
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5704"]
