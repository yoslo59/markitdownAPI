from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from markitdown import MarkItDown
from typing import Optional

app = FastAPI(title="MarkItDown API", version="1.0")

@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    use_plugins: bool = Form(False),
    docintel_endpoint: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None)  # ex: "gpt-4o-mini" si tu veux décrire des images
):
    # Instanciation MarkItDown (LLM image captions & Azure DI sont optionnels)
    md = MarkItDown(
        enable_plugins=use_plugins,
        docintel_endpoint=docintel_endpoint
        # Pour LLM images: fournir un client OpenAI dans ton code si tu veux pousser l’option.
    )
    content = await file.read()
    result = md.convert_stream(content, file_name=file.filename)

    return JSONResponse({
        "filename": file.filename,
        "markdown": result.text_content,
        "metadata": result.metadata,   # exif/structure, selon format
        "warnings": getattr(result, "warnings", None)
    })
