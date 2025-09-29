import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from markitdown import MarkItDown
from typing import Optional

SAVE_UPLOADS  = os.getenv("SAVE_UPLOADS", "false").lower() == "true"
SAVE_OUTPUTS  = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"
UPLOAD_DIR    = os.getenv("UPLOAD_DIR", "/data/uploads")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "/data/outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="MarkItDown API", version="1.0")

@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    use_plugins: bool = Form(False),
    docintel_endpoint: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None)
):
    md = MarkItDown(
        enable_plugins=use_plugins,
        docintel_endpoint=docintel_endpoint
    )

    content = await file.read()
    # Sauvegarde d'entrée si demandé
    if SAVE_UPLOADS:
        in_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(in_path, "wb") as f:
            f.write(content)

    result = md.convert_stream(content, file_name=file.filename)
    markdown = result.text_content or ""

    # Sauvegarde de sortie si demandé
    out_name = f"{os.path.splitext(file.filename)[0]}.md"
    if SAVE_OUTPUTS:
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown)

    return JSONResponse({
        "filename": file.filename,
        "output_filename": out_name if SAVE_OUTPUTS else None,
        "markdown": markdown,
        "metadata": result.metadata,
    })
