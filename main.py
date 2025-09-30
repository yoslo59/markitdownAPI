import os
import io
import re
import base64
import statistics
from typing import Optional, Tuple, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from markitdown import MarkItDown
from openai import AzureOpenAI

# OCR
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
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

# OCR (tunable sans rebuild)
OCR_ENABLED        = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANGS          = os.getenv("OCR_LANGS", "fra+eng").strip()
OCR_DPI            = int(os.getenv("OCR_DPI", "350"))
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "25"))
OCR_MIN_CHARS      = int(os.getenv("OCR_MIN_CHARS", "500"))
OCR_KEEP_SPACES    = os.getenv("OCR_KEEP_SPACES", "true").lower() == "true"
OCR_TWO_PASS       = os.getenv("OCR_TWO_PASS", "true").lower() == "true"
OCR_PSMS           = [p.strip() for p in os.getenv("OCR_PSMS", "6,4,11").split(",")]  # 6=block, 4=columns, 11=sparse
OCR_DPI_CANDIDATES = [int(x) for x in os.getenv("OCR_DPI_CANDIDATES", "300,350,400").split(",")]
OCR_SCORE_GOOD_ENOUGH = float(os.getenv("OCR_SCORE_GOOD_ENOUGH", "0.6"))

# Embedding images base64
EMBED_IMAGES       = os.getenv("EMBED_IMAGES", "ocr_only").strip()   # none | ocr_only | all
IMG_FORMAT         = os.getenv("IMG_FORMAT", "png").strip().lower()  # png | jpeg
IMG_JPEG_QUALITY   = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH      = int(os.getenv("IMG_MAX_WIDTH", "1400"))
IMG_ALT_PREFIX     = os.getenv("IMG_ALT_PREFIX", "Capture").strip()

# (Optionnel) Azure Document Intelligence (MarkItDown plugin)
DEFAULT_DOCINTEL_ENDPOINT = os.getenv("DEFAULT_DOCINTEL_ENDPOINT", "").strip()

# Dossiers persistants
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="2.3")

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
# Helpers types / utils
# ---------------------------
def guess_is_pdf(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower() in ("application/pdf", "pdf"):
        return True
    return filename.lower().endswith(".pdf")

def guess_is_image(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    return any(filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))

# ---------------------------
# OCR helpers (Tesseract)
# ---------------------------
_table_chars = re.compile(r"[|+\-=_]{3,}")

def _tess_config(psm: str, keep_spaces: bool) -> str:
    cfg = f"--psm {psm} --oem 1"
    if keep_spaces:
        cfg += " -c preserve_interword_spaces=1"
    return cfg

def _preprocess_for_ocr(im: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    # légère binarisation
    g = g.point(lambda p: 255 if p > 190 else (0 if p < 110 else p))
    return g

def _score_text_for_table(txt: str) -> float:
    if not txt:
        return 0.0
    lines = txt.splitlines()
    n = max(1, len(lines))
    pipes = sum(l.count("|") for l in lines)
    ascii_blocks = sum(1 for l in lines if _table_chars.search(l))
    noise = sum(1 for l in lines if "nnn" in l or "—" in l or "$---" in l)
    return (pipes*1.0 + ascii_blocks*2.0)/n - noise*0.25 + len(txt)/6000.0

def _wrap_ascii_tables(txt: str) -> str:
    if not txt:
        return txt
    out, buf, in_blk = [], [], False
    for line in txt.splitlines():
        looks_table = _table_chars.search(line) is not None or line.strip().startswith("|")
        if looks_table and not in_blk:
            in_blk = True
            out.append("```text")
            buf = []
        if in_blk and not looks_table and buf:
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

def ocr_pil_best(im: Image.Image) -> Tuple[str, float]:
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    best_txt, best_score = "", -1e9
    for psm in OCR_PSMS:
        cfg = _tess_config(psm, OCR_KEEP_SPACES)
        t1 = pytesseract.image_to_string(im, lang=OCR_LANGS, config=cfg) or ""
        s1 = _score_text_for_table(t1)
        cand_txt, cand_score = t1, s1
        if OCR_TWO_PASS:
            im2 = _preprocess_for_ocr(im)
            t2 = pytesseract.image_to_string(im2, lang=OCR_LANGS, config=cfg) or ""
            s2 = _score_text_for_table(t2)
            if s2 > cand_score:
                cand_txt, cand_score = t2, s2
        if cand_score > best_score:
            best_txt, best_score = cand_txt, cand_score
        if best_score >= OCR_SCORE_GOOD_ENOUGH:
            break
    return _wrap_ascii_tables(best_txt.strip()), best_score

def pil_to_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    if IMG_FORMAT in ("jpeg", "jpg"):
        im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=IMG_JPEG_QUALITY, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def resize_max(im: Image.Image, max_w: int) -> Image.Image:
    if max_w and im.width > max_w:
        r = max_w / im.width
        return im.resize((max_w, int(im.height*r)), Image.LANCZOS)
    return im

# ---------------------------
# PDF structured rebuild (rawdict)
# ---------------------------
BulletRe = re.compile(r"^\s*([•\-–‣●◦▪]|(\d+[\.\)]))\s+")
CodeFenceRe = re.compile(r"```")

def spans_to_text(spans: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Merge spans in a line-ish text, return text + average font size."""
    pieces = []
    sizes = []
    for sp in spans:
        t = sp.get("text", "")
        if not t:
            continue
        pieces.append(t)
        sz = sp.get("size")
        if isinstance(sz, (int, float)):
            sizes.append(float(sz))
    avg = statistics.fmean(sizes) if sizes else 0.0
    return "".join(pieces), avg

def classify_heading(avg_size: float, page_sizes: List[float]) -> int:
    """Return 0 for normal, 1..3 for #..### based on font size quantiles."""
    if not page_sizes:
        return 0
    q75 = statistics.quantiles(page_sizes, n=4)[2] if len(page_sizes) >= 4 else max(page_sizes)
    q90 = statistics.quantiles(page_sizes, n=10)[8] if len(page_sizes) >= 10 else q75
    if avg_size >= q90: return 1
    if avg_size >= q75: return 2
    return 0

def looks_tableish(txt: str) -> bool:
    if _table_chars.search(txt):
        return True
    # crude: many pipes in few lines
    lines = txt.splitlines()
    if len(lines) >= 3 and sum(l.count("|") for l in lines) >= 4:
        return True
    return False

def looks_code_block(txt: str) -> bool:
    # if it already contains fences, don’t double wrap
    if CodeFenceRe.search(txt):
        return True
    # contains lots of mono characters / alignment? heuristic off for now
    return False

def crop_to_image(page: fitz.Page, bbox: List[float], dpi: int) -> Image.Image:
    x0, y0, x1, y1 = bbox
    rect = fitz.Rect(x0, y0, x1, y1)
    # Render just this rect at the dpi scale
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def md_for_text_block(txt: str, avg_size: float, page_sizes: List[float]) -> str:
    txt = txt.strip()
    if not txt:
        return ""
    # lists
    if BulletRe.match(txt):
        # keep as-is; convert “- ” / bullets to markdown dash
        line = BulletRe.sub(lambda m: f"- ", txt)
        return line
    # headings by font size
    h = classify_heading(avg_size, page_sizes)
    if h == 1:
        return f"# {txt}"
    if h == 2:
        return f"## {txt}"
    # tables / code-ish
    if looks_tableish(txt):
        return f"```text\n{txt}\n```"
    # normal paragraph
    return txt

def pdf_to_markdown_structured(pdf_bytes: bytes, force_ocr: bool) -> Tuple[str, Dict[str, Any]]:
    """
    Walks the PDF by visual blocks; text->MD, images->OCR or base64; preserves order.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = []
        meta_stats = {"pages": doc.page_count, "ocr_pages": 0, "images_embedded": 0, "images_ocr_text": 0}
        for pno in range(min(doc.page_count, OCR_MAX_PAGES)):
            page = doc.load_page(pno)
            raw = page.get_text("rawdict")
            page_sizes: List[float] = []
            # collect sizes for heading thresholds
            for b in raw.get("blocks", []):
                if b.get("type") == 0:  # text
                    for l in b.get("lines", []):
                        for sp in l.get("spans", []):
                            sz = sp.get("size")
                            if isinstance(sz, (int, float)):
                                page_sizes.append(float(sz))
            out_lines: List[str] = []
            for b in raw.get("blocks", []):
                btype = b.get("type")
                if btype == 0:  # text block
                    # concatenate spans preserving line breaks
                    block_txt_parts = []
                    block_sizes = []
                    for l in b.get("lines", []):
                        t, avg = spans_to_text(l.get("spans", []))
                        if t.strip():
                            block_txt_parts.append(t)
                            block_sizes.append(avg)
                    block_txt = "\n".join(block_txt_parts)
                    avg_block_size = statistics.fmean(block_sizes) if block_sizes else 0.0
                    md = md_for_text_block(block_txt, avg_block_size, page_sizes)
                    if md:
                        out_lines.append(md)
                elif btype == 1:  # image block with bbox
                    bbox = b.get("bbox")
                    if not bbox:
                        continue
                    # If not forcing OCR: embed image only when EMBED_IMAGES says so.
                    # If forcing OCR: try OCR first; if weak → embed image.
                    try:
                        im = crop_to_image(page, bbox, OCR_DPI)
                        im = resize_max(im, IMG_MAX_WIDTH)
                    except Exception:
                        continue
                    do_ocr = force_ocr and OCR_ENABLED
                    inserted = False
                    if do_ocr:
                        txt, score = ocr_pil_best(im)
                        if txt and (score >= OCR_SCORE_GOOD_ENOUGH or len(txt) >= OCR_MIN_CHARS//4):
                            # format OCR text like a code/table block if needed
                            if looks_tableish(txt) and not looks_code_block(txt):
                                txt = f"```text\n{txt}\n```"
                            out_lines.append(txt)
                            meta_stats["images_ocr_text"] += 1
                            meta_stats["ocr_pages"] += 1
                            inserted = True
                    if not inserted:
                        if EMBED_IMAGES != "none" and (EMBED_IMAGES == "all" or not do_ocr):
                            data_uri = pil_to_b64(im)
                            out_lines.append(f'![{IMG_ALT_PREFIX} p{pno+1}]({data_uri})')
                            meta_stats["images_embedded"] += 1
                        elif EMBED_IMAGES == "ocr_only" and do_ocr:
                            # OCR failed/weak → embed
                            data_uri = pil_to_b64(im)
                            out_lines.append(f'![{IMG_ALT_PREFIX} p{pno+1}]({data_uri})')
                            meta_stats["images_embedded"] += 1
            page_md = "\n\n".join(out_lines).strip()
            if page_md:
                pages.append(page_md)
        final_md = "\n\n".join(pages).strip()
        return final_md, meta_stats
    finally:
        doc.close()

# ---------------------------
# Light post-formatter for MarkItDown (titles/lists/tables)
# ---------------------------
H1_RE = re.compile(r"^\s*(?P<t>[A-ZÉÈÀÂÎÔÛÄËÏÖÜ0-9][^\n]{4,})\s*$")

def post_format_markdown(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out = []
    for i, line in enumerate(lines):
        s = line.strip()

        # promote headings if line is stand-alone and next line blank
        if s and (i+1 < len(lines) and not lines[i+1].strip()):
            if len(s) <= 120 and not s.endswith(":") and H1_RE.match(s):
                out.append(f"## {s}" if i > 3 else f"# {s}")  # first big title as H1, others H2
                continue

        # bullets normalization
        if re.match(r"^\s*[•\-–‣●◦▪]\s+", s):
            out.append(re.sub(r"^\s*[•\-–‣●◦▪]\s+", "- ", s))
            continue

        # wrap ASCII-ish tables
        if _table_chars.search(s) and len(s) > 6:
            out.append("```text")
            out.append(s)
            out.append("```")
            continue

        out.append(line)
    return "\n".join(out)

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
    body{font-family: system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Arial; padding:24px; max-width: 980px; margin:auto; line-height:1.45}
    h1{margin-bottom:0.5rem}
    .card{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin-top:16px;box-shadow:0 1px 2px rgba(0,0,0,.05)}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    label{font-weight:600}
    button{padding:10px 16px;border:1px solid #111;border-radius:10px;background:#111;color:#fff;cursor:pointer}
    button:disabled{opacity:.5;cursor:not-allowed}
    textarea{width:100%;min-height:260px;padding:12px;border-radius:10px;border:1px solid #e5e7eb;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
    .muted{color:#6b7280;font-size:12px}
    input[type="checkbox"]{transform: scale(1.2);}
    a#download{padding:10px 16px;border:1px solid #111;border-radius:10px;text-decoration:none}
    input[type="text"]{padding:8px 10px;border:1px solid #e5e7eb;border-radius:10px;min-width:420px}
  </style>
</head>
<body>
  <h1>MarkItDown — Conversion</h1>
  <p class="muted">PDF & co → Markdown. Plugins MarkItDown (texte structuré). Forcer OCR = insérer texte des images à l’emplacement d’origine, ou image en base64 si non OCR-isable.</p>

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
      <span class="muted">Optionnel (MarkItDown plugins)</span>
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
    llm_model: Optional[str] = Form(None),   # compat
    use_llm: bool = Form(False),
    force_ocr: bool = Form(False),
):
    """
    - Plugins seuls: MarkItDown (+ post-format titres/listes/tables).
    - Plugins + Forcer OCR: reconstruction par blocs PyMuPDF; OCR sur images;
      insertion texte OCR à la position d'origine; sinon image base64.
    - Non PDF: MarkItDown (+ OCR si image et forcer OCR).
    """
    try:
        if not docintel_endpoint:
            docintel_endpoint = DEFAULT_DOCINTEL_ENDPOINT

        content = await file.read()

        # Save input
        in_path = None
        if SAVE_UPLOADS:
            in_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(in_path, "wb") as f:
                f.write(content)

        is_pdf = guess_is_pdf(file.filename, file.content_type)
        is_img = guess_is_image(file.filename, file.content_type)

        metadata: Dict[str, Any] = {"pipeline": None}

        markdown = ""

        if is_pdf and force_ocr:
            # Full structured rebuild with OCR/images
            markdown, stats = pdf_to_markdown_structured(content, force_ocr=True)
            metadata["pipeline"] = "pdf_structured_ocr"
            metadata.update(stats)
            # If for some reason it's empty, fallback to MarkItDown
            if not markdown.strip():
                md = MarkItDown(enable_plugins=use_plugins, docintel_endpoint=docintel_endpoint)
                res = md.convert_stream(io.BytesIO(content), file_name=file.filename)
                markdown = post_format_markdown(getattr(res, "text_content", "") or "")
                metadata["pipeline_fallback"] = "markitdown"
                metadata["warnings"] = ["Structured OCR returned empty; fell back to MarkItDown."]
        else:
            # Non-forced path → MarkItDown first
            md = MarkItDown(enable_plugins=use_plugins, docintel_endpoint=docintel_endpoint)
            res = md.convert_stream(io.BytesIO(content), file_name=file.filename)
            markdown = getattr(res, "text_content", "") or ""
            metadata.update(getattr(res, "metadata", {}) or {})
            metadata["pipeline"] = "markitdown"
            # Post-format for titles/lists/tables
            markdown = post_format_markdown(markdown)

            # If forcing OCR on an image file, add OCR text / embed
            if is_img and force_ocr and OCR_ENABLED:
                try:
                    with Image.open(io.BytesIO(content)) as im:
                        im = resize_max(im, IMG_MAX_WIDTH)
                        txt, score = ocr_pil_best(im)
                        if txt and (score >= OCR_SCORE_GOOD_ENOUGH or len(txt) >= OCR_MIN_CHARS//4):
                            markdown += f"\n\n# OCR (image)\n{txt}\n"
                            metadata["images_ocr_text"] = 1
                        else:
                            if EMBED_IMAGES != "none":
                                data_uri = pil_to_b64(im)
                                markdown += f'\n\n![{IMG_ALT_PREFIX}]({data_uri})\n'
                                metadata["images_embedded"] = 1
                except Exception:
                    pass

        # Persist output
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
                    resp = client.chat.completions.create(
                        model=AZURE_DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": "Tu es un assistant qui résume des documents techniques en français, de manière concise et structurée."},
                            {"role": "user", "content": f"Résume le document suivant en 10 points maximum, avec un titre en H1 et des sous-titres:\n\n{snippet}"}
                        ],
                        max_completion_tokens=800
                    )
                    msg = resp.choices[0].message if resp and getattr(resp, "choices", None) else None
                    metadata["azure_summary"] = getattr(msg, "content", None) or "[Résumé vide (vérifie le déploiement et le contenu)]"
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
