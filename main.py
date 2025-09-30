import os
import io
import re
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from markitdown import MarkItDown
from openai import AzureOpenAI

# OCR libs
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import fitz  # PyMuPDF

# ---------------------------
# Config via variables d'env
# ---------------------------
SAVE_UPLOADS  = os.getenv("SAVE_UPLOADS", "false").lower() == "true"
SAVE_OUTPUTS  = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"
UPLOAD_DIR    = os.getenv("UPLOAD_DIR", "/data/uploads")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "/data/outputs")

# Azure OpenAI
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_KEY        = os.getenv("AZURE_OPENAI_KEY", "").strip()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini").strip()
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

# OCR (tunable sans rebuild)
OCR_ENABLED        = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANGS          = os.getenv("OCR_LANGS", "fra+eng").strip()
OCR_DPI            = int(os.getenv("OCR_DPI", "350"))              # 350 par défaut (screenshots)
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "25"))
OCR_MIN_CHARS      = int(os.getenv("OCR_MIN_CHARS", "500"))
OCR_MODE           = os.getenv("OCR_MODE", "append").strip()       # replace_when_empty | append
OCR_KEEP_SPACES    = os.getenv("OCR_KEEP_SPACES", "true").lower() == "true"
OCR_TWO_PASS       = os.getenv("OCR_TWO_PASS", "true").lower() == "true"
OCR_TABLE_MODE     = os.getenv("OCR_TABLE_MODE", "true").lower() == "true"
OCR_PSMS           = [p.strip() for p in os.getenv("OCR_PSMS", "6,4,11").split(",")]           # 6=block, 4=columns, 11=sparse
OCR_DPI_CANDIDATES = [int(x) for x in os.getenv("OCR_DPI_CANDIDATES", "300,350,400").split(",")]

# (Optionnel) Endpoint Azure Document Intelligence par défaut
DEFAULT_DOCINTEL_ENDPOINT = os.getenv("DEFAULT_DOCINTEL_ENDPOINT", "").strip()

# Dossiers persistants
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="1.8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

def get_azure_client() -> Optional[AzureOpenAI]:
    if AZURE_ENDPOINT and AZURE_KEY:
        return AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version=AZURE_API_VER
        )
    return None

# ---------------------------
# Helpers OCR
# ---------------------------
_table_chars = re.compile(r"[|+\-=_]{3,}")

def _tess_config(psm: str, keep_spaces: bool, table_mode: bool) -> str:
    cfg = f"--psm {psm} --oem 1"
    if keep_spaces:
        cfg += " -c preserve_interword_spaces=1"
    if table_mode:
        cfg += " -c tessedit_write_images=false"
    return cfg

def _preprocess_for_ocr(im: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    g = g.point(lambda p: 255 if p > 190 else (0 if p < 110 else p))
    return g

def _score_text_for_table(txt: str) -> float:
    """Score simple favorisant les rendus type tableau/monospace et pénalisant le bruit évident."""
    if not txt:
        return 0.0
    lines = txt.splitlines()
    n = max(1, len(lines))
    pipes = sum(l.count("|") for l in lines)
    plus  = sum(l.count("+") for l in lines)
    dashes= sum(l.count("-") for l in lines)
    ascii_blocks = sum(1 for l in lines if _table_chars.search(l))
    noise = sum(1 for l in lines if "nnn" in l or "Se ne" in l or "—" in l)
    return (pipes*1.0 + plus*0.6 + dashes*0.3 + ascii_blocks*2.0)/n - noise*0.2 + len(txt)/5000.0

def _wrap_tables_as_code(txt: str) -> str:
    """Détecte des blocs ASCII et les emballe dans ```text``` pour préserver l’alignement en Markdown."""
    if not txt:
        return txt
    out = []
    buf = []
    in_blk = False
    for line in txt.splitlines():
        is_tbl = _table_chars.search(line) is not None or line.strip().startswith("|")
        if is_tbl and not in_blk:
            in_blk = True
            out.append("```text")
            buf = []
        if in_blk and not is_tbl and buf:
            # fin de bloc
            out.extend(buf)
            out.append("```")
            in_blk = False
            out.append(line)
            buf = []
            continue
        if in_blk:
            buf.append(line)
        else:
            out.append(line)
    if in_blk:
        out.extend(buf)
        out.append("```")
    return "\n".join(out)

def _ocr_image_best(im: Image.Image, langs: str) -> str:
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    candidates = []
    for psm in OCR_PSMS:
        cfg = _tess_config(psm, OCR_KEEP_SPACES, OCR_TABLE_MODE)
        # Pass brute
        t1 = pytesseract.image_to_string(im, lang=langs, config=cfg) or ""
        best = t1
        # Pass pré-traitée
        if OCR_TWO_PASS:
            im2 = _preprocess_for_ocr(im)
            t2 = pytesseract.image_to_string(im2, lang=langs, config=cfg) or ""
            if len(t2) > len(best):
                best = t2
        candidates.append((best, _score_text_for_table(best)))
    candidates.sort(key=lambda x: x[1], reverse=True)  # meilleur score d'abord
    return candidates[0][0].strip() if candidates else ""

def ocr_image_bytes(img_bytes: bytes, langs: str) -> str:
    with Image.open(io.BytesIO(img_bytes)) as im:
        txt = _ocr_image_best(im, langs)
        return _wrap_tables_as_code(txt)

def _raster_pdf_page(page, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_pdf_bytes(pdf_bytes: bytes, langs: str, dpi: int, max_pages: int) -> tuple[str, int]:
    out = []
    pages_done = 0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = doc.page_count
        for i in range(min(total_pages, max_pages)):
            page = doc.load_page(i)
            best_txt = ""
            best_score = -1e9
            # multi-DPI par page
            for d in OCR_DPI_CANDIDATES:
                im = _raster_pdf_page(page, d)
                txt = _ocr_image_best(im, langs)
                sc  = _score_text_for_table(txt)
                if sc > best_score:
                    best_txt, best_score = txt, sc
            if best_txt.strip():
                out.append(f"\n\n## Page {i+1}\n\n{_wrap_tables_as_code(best_txt)}")
            pages_done += 1
    finally:
        doc.close()
    return ("\n".join(out).strip(), pages_done)

def guess_is_pdf(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower() in ("application/pdf", "pdf"):
        return True
    return filename.lower().endswith(".pdf")

def guess_is_image(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    return any(filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))

# ---------------------------
# Mini interface web
# ---------------------------
HTML_PAGE = r'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>MarkItDown UI</title>
  <style>
    :root{color-scheme: light dark;}
    body{font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; padding:24px; max-width: 980px; margin:auto; line-height:1.45}
    h1{margin-bottom:0.5rem}
    .card{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin-top:16px;box-shadow:0 1px 2px rgba(0,0,0,.05)}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    label{font-weight:600}
    button{padding:10px 16px;border:1px solid #111;border-radius:10px;background:#111;color:#fff;cursor:pointer}
    button:disabled{opacity:.5;cursor:not-allowed}
    textarea{width:100%;min-height:260px;padding:12px;border-radius:10px;border:1px solid #e5e7eb;font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace}
    .muted{color:#6b7280;font-size:12px}
    input[type="checkbox"]{transform: scale(1.2);}
    a#download{padding:10px 16px;border:1px solid #111;border-radius:10px;text-decoration:none}
    input[type="text"]{padding:8px 10px;border:1px solid #e5e7eb;border-radius:10px;min-width:420px}
  </style>
</head>
<body>
  <h1>MarkItDown — Conversion</h1>
  <p class="muted">Upload un document (PDF, DOCX, PPTX, XLSX, HTML, etc.) → Markdown. Optionnel : résumé Azure OpenAI et OCR fallback pour PDF/images.</p>

  <div class="card">
    <div class="row">
      <label for="file">Fichier :</label>
      <input id="file" type="file" />
    </div>
    <div class="row" style="margin-top:8px">
      <label for="plugins">Activer plugins MarkItDown</label>
      <input id="plugins" type="checkbox" />
      <label for="llm">Résumé Azure LLM</label>
      <input id="llm" type="checkbox" />
      <label for="forceocr">Forcer OCR</label>
      <input id="forceocr" type="checkbox" />
    </div>
    <div class="row" style="margin-top:8px; gap:8px; align-items:baseline;">
      <label for="di">Endpoint Azure Document Intelligence</label>
      <input id="di" type="text" placeholder="https://<resource>.cognitiveservices.azure.com/"/>
      <span class="muted">Optionnel (meilleur OCR/structure pour PDF scannés)</span>
    </div>
    <div class="row" style="margin-top:8px">
      <button id="convert">Convertir</button>
      <a id="download" download="sortie.md" style="display:none;margin-left:8px">Télécharger Markdown</a>
    </div>
    <p id="status" class="muted" style="margin-top:8px"></p>
  </div>

  <div class="card">
    <label>Markdown</label>
    <textarea id="md"></textarea>
  </div>

  <div class="card">
    <label>Métadonnées (JSON)</label>
    <textarea id="meta" style="min-height:140px"></textarea>
  </div>

<script>
const $ = (id)=>document.getElementById(id);
const endpoint = "/convert";

fetch("/config").then(r=>r.ok?r.json():null).then(j=>{
  if(j && j.docintel_default){ $("di").value = j.docintel_default; }
}).catch(()=>{});

$("convert").onclick = async () => {
  const f = $("file").files[0];
  if(!f){ alert("Choisis un fichier."); return; }
  $("convert").disabled = true;
  $("status").textContent = "Conversion en cours...";
  $("md").value = "";
  $("meta").value = "";
  $("download").style.display = "none";

  const fd = new FormData();
  fd.append("file", f);
  fd.append("use_plugins", $("plugins").checked ? "true" : "false");
  fd.append("use_llm", $("llm").checked ? "true" : "false");
  fd.append("docintel_endpoint", $("di").value || "");
  fd.append("force_ocr", $("forceocr").checked ? "true" : "false");

  try{
    const res = await fetch(endpoint, { method:"POST", body: fd });
    if(!res.ok){ throw new Error("HTTP "+res.status); }
    const json = await res.json();
    $("md").value = json.markdown || "";
    $("meta").value = JSON.stringify(json.metadata || {}, null, 2);

    const blob = new Blob([$("md").value], {type:"text/markdown;charset=utf-8"});
    const url  = URL.createObjectURL(blob);
    const a = $("download");
    a.href = url;
    a.download = (json.output_filename || "sortie.md");
    a.style.display = "inline-block";
    $("status").textContent = "OK";
  }catch(e){
    $("status").textContent = "Erreur : " + (e && e.message ? e.message : e);
  }finally{
    $("convert").disabled = false;
  }
};
</script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_PAGE)

@app.get("/config", response_class=JSONResponse)
def get_config():
    return JSONResponse({"docintel_default": DEFAULT_DOCINTEL_ENDPOINT})

# ---------------------------
# Endpoint API de conversion
# ---------------------------
@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    use_plugins: bool = Form(False),
    docintel_endpoint: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),   # ignoré pour Azure; gardé pour compat
    use_llm: bool = Form(False),
    force_ocr: bool = Form(False),
):
    """
    Convertit le fichier avec MarkItDown.
    Fallback OCR (Tesseract) si le texte est pauvre, ou si force_ocr=true.
    Optionnel: résumé Azure OpenAI (use_llm=true).
    """
    try:
        # Fallback DI si non fourni
        if not docintel_endpoint:
            docintel_endpoint = DEFAULT_DOCINTEL_ENDPOINT

        md = MarkItDown(
            enable_plugins=use_plugins,
            docintel_endpoint=docintel_endpoint
        )

        content = await file.read()

        # Save input
        in_path = None
        if SAVE_UPLOADS:
            in_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(in_path, "wb") as f:
                f.write(content)

        # MarkItDown
        stream = io.BytesIO(content)
        result = md.convert_stream(stream, file_name=file.filename)

        markdown = getattr(result, "text_content", "") or ""
        metadata = getattr(result, "metadata", None) or {}
        warnings = getattr(result, "warnings", None)
        if warnings:
            metadata["warnings"] = warnings

        # OCR fallback si nécessaire
        is_pdf = guess_is_pdf(file.filename, file.content_type)
        is_img = guess_is_image(file.filename, file.content_type)
        needs_ocr = OCR_ENABLED and (force_ocr or (len(markdown.strip()) < OCR_MIN_CHARS and (is_pdf or is_img)))

        if needs_ocr:
            ocr_text = ""
            pages_done = 0
            if is_pdf:
                ocr_text, pages_done = ocr_pdf_bytes(content, OCR_LANGS, OCR_DPI, OCR_MAX_PAGES)
                metadata["ocr_pages"] = pages_done
                metadata["ocr_langs"] = OCR_LANGS
                metadata["ocr_dpi"] = OCR_DPI
            elif is_img:
                ocr_text = ocr_image_bytes(content, OCR_LANGS)
                metadata["ocr_pages"] = 1
                metadata["ocr_langs"] = OCR_LANGS

            if ocr_text.strip():
                if OCR_MODE == "append" and markdown.strip():
                    markdown += "\n\n# OCR (extrait)\n" + ocr_text
                else:
                    if len(markdown.strip()) < OCR_MIN_CHARS:
                        markdown = ocr_text if not is_pdf else f"# OCR\n{ocr_text}"
                    else:
                        markdown += "\n\n# OCR (extrait)\n" + ocr_text
            else:
                metadata["ocr_note"] = "OCR tenté mais aucun texte détecté."

        # Save output
        out_name = f"{os.path.splitext(file.filename)[0]}.md"
        out_path = None
        if SAVE_OUTPUTS:
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(markdown)

        # Azure résumé (optionnel)
        if use_llm:
            client = get_azure_client()
            if client:
                try:
                    snippet = markdown[:12000] if markdown else "[document vide]"
                    print(f"[AZURE] endpoint={AZURE_ENDPOINT} deployment={AZURE_DEPLOYMENT} ver={AZURE_API_VER}")
                    resp = client.chat.completions.create(
                        model=AZURE_DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": "Tu es un assistant qui résume des documents techniques en français, de manière concise et structurée."},
                            {"role": "user", "content": f"Résume le document suivant en 10 points maximum, avec un titre en H1 et des sous-titres:\n\n{snippet}"}
                        ],
                        max_completion_tokens=800
                    )
                    content_msg = None
                    if resp and getattr(resp, "choices", None):
                        msg = resp.choices[0].message
                        if hasattr(msg, "content") and isinstance(msg.content, str):
                            content_msg = msg.content
                        elif hasattr(msg, "content") and isinstance(msg.content, list):
                            parts = []
                            for part in msg.content:
                                if isinstance(part, dict) and "text" in part:
                                    parts.append(part["text"])
                            content_msg = "".join(parts) if parts else None
                    metadata["azure_summary"] = content_msg or "[Résumé vide (vérifie le déploiement et le contenu)]"
                except Exception as e:
                    metadata["azure_summary"] = f"[Erreur Azure OpenAI: {type(e).__name__}: {e}]"
            else:
                metadata["azure_summary"] = "[Azure OpenAI non configuré]"

        if SAVE_UPLOADS and in_path:
            metadata["saved_input_path"] = in_path
        if SAVE_OUTPUTS and out_path:
            metadata["saved_output_path"] = out_path

        return JSONResponse({
            "filename": file.filename,
            "output_filename": out_name if SAVE_OUTPUTS else None,
            "markdown": markdown,
            "metadata": metadata,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {type(e).__name__}: {e}")

# ---------------------------
# Healthcheck
# ---------------------------
@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"
