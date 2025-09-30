import os
import io
import re
import base64
from typing import Optional, Tuple, List, Dict, Any

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
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

# OCR (tunable)
OCR_ENABLED        = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANGS          = os.getenv("OCR_LANGS", "fra+eng").strip()
OCR_DPI            = int(os.getenv("OCR_DPI", "350"))
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "25"))
OCR_MIN_CHARS      = int(os.getenv("OCR_MIN_CHARS", "500"))
OCR_MODE           = os.getenv("OCR_MODE", "append").strip()
OCR_KEEP_SPACES    = os.getenv("OCR_KEEP_SPACES", "true").lower() == "true"
OCR_TWO_PASS       = os.getenv("OCR_TWO_PASS", "true").lower() == "true"
OCR_TABLE_MODE     = os.getenv("OCR_TABLE_MODE", "true").lower() == "true"
OCR_PSMS           = [p.strip() for p in os.getenv("OCR_PSMS", "6,4,11").split(",")]     # 6=block, 4=columns, 11=sparse
OCR_DPI_CANDIDATES = [int(x) for x in os.getenv("OCR_DPI_CANDIDATES", "300,350,400").split(",")]
OCR_SCORE_GOOD_ENOUGH = float(os.getenv("OCR_SCORE_GOOD_ENOUGH", "0.6"))

# Embedding images base64
EMBED_IMAGES       = os.getenv("EMBED_IMAGES", "ocr_only").strip()  # none | ocr_only | all
IMG_FORMAT         = os.getenv("IMG_FORMAT", "png").strip().lower()  # png | jpeg
IMG_JPEG_QUALITY   = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH      = int(os.getenv("IMG_MAX_WIDTH", "1400"))         # resize max (px), 0 = no limit
IMG_ALT_PREFIX     = os.getenv("IMG_ALT_PREFIX", "Capture").strip()

# (Optionnel) Azure Document Intelligence
DEFAULT_DOCINTEL_ENDPOINT = os.getenv("DEFAULT_DOCINTEL_ENDPOINT", "").strip()

# Nouveau : mode de rendu
# - preserve : respecte l’ordre des blocs PDF (texte/images) et insère OCR/IMG au bon endroit
# - auto     : garde l’ancien pipeline (MarkItDown d’abord, puis OCR global en append)
LAYOUT_MODE       = os.getenv("LAYOUT_MODE", "preserve").strip()  # preserve | auto

# Heuristiques titres/listes/tableaux
HEADINGS_DETECT   = os.getenv("HEADINGS_DETECT", "true").lower() == "true"
LISTS_DETECT      = os.getenv("LISTS_DETECT", "true").lower() == "true"
TABLES_DETECT     = os.getenv("TABLES_DETECT", "true").lower() == "true"

# Dossiers persistants
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="2.1")

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
# Helpers génériques
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
# OCR utils
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
    if not txt:
        return 0.0
    lines = txt.splitlines()
    n = max(1, len(lines))
    pipes = sum(l.count("|") for l in lines)
    plus  = sum(l.count("+") for l in lines)
    dashes= sum(l.count("-") for l in lines)
    ascii_blocks = sum(1 for l in lines if _table_chars.search(l))
    noise = sum(1 for l in lines if "nnn" in l or "—" in l or "$-----" in l)
    return (pipes*1.0 + plus*0.6 + dashes*0.3 + ascii_blocks*2.0)/n - noise*0.25 + len(txt)/5000.0

def _wrap_tables_as_code(txt: str) -> str:
    if not txt:
        return txt
    out, buf, in_blk = [], [], False
    for line in txt.splitlines():
        is_tbl = _table_chars.search(line) is not None or line.strip().startswith("|")
        if is_tbl and not in_blk:
            in_blk = True
            out.append("```text")
            buf = []
        if in_blk and not is_tbl and buf:
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

def _ocr_image_best(im: Image.Image, langs: str) -> Tuple[str, float]:
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    best_txt, best_score = "", -1e9
    for psm in OCR_PSMS:
        cfg = _tess_config(psm, OCR_KEEP_SPACES, OCR_TABLE_MODE)
        t1 = pytesseract.image_to_string(im, lang=langs, config=cfg) or ""
        s1 = _score_text_for_table(t1)
        cand_txt, cand_score = t1, s1
        if OCR_TWO_PASS:
            im2 = _preprocess_for_ocr(im)
            t2 = pytesseract.image_to_string(im2, lang=langs, config=cfg) or ""
            s2 = _score_text_for_table(t2)
            if s2 > cand_score:
                cand_txt, cand_score = t2, s2
        if cand_score > best_score:
            best_txt, best_score = cand_txt, cand_score
        if best_score >= OCR_SCORE_GOOD_ENOUGH:
            break
    return best_txt.strip(), best_score

def _pil_resize_max(im: Image.Image, max_w: int) -> Image.Image:
    if max_w and im.width > max_w:
        ratio = max_w / im.width
        new_h = int(im.height * ratio)
        return im.resize((max_w, new_h), Image.LANCZOS)
    return im

def _pil_to_base64(im: Image.Image, fmt: str = "png", quality: int = 85) -> str:
    buf = io.BytesIO()
    if fmt.lower() in ("jpeg","jpg"):
        im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def should_embed_images_for_text(ocr_txt: str, ocr_score: float) -> bool:
    if EMBED_IMAGES == "all":
        return True
    if EMBED_IMAGES == "none":
        return False
    return (not ocr_txt or len(ocr_txt.strip()) < OCR_MIN_CHARS or ocr_score < OCR_SCORE_GOOD_ENOUGH)

# ---------------------------
# Layout-aware PDF → Markdown
# ---------------------------
_bullet_rx  = re.compile(r"^\s*([•·\-–—\*]|[0-9]+[.)])\s+")
_heading_rx = re.compile(r"^\s*(?:[A-Z][A-Z ]{3,}|[#]{1,6}\s+)")  # renfort éventuel

def _classify_heading_size(font_size: float, median: float, p90: float) -> int:
    """Retourne niveau H1..H3 (ou 0 si paragraphe) selon taille relative."""
    if font_size >= max(p90, median*1.35):  # très grand
        return 1
    if font_size >= median*1.20:
        return 2
    if font_size >= median*1.10:
        return 3
    return 0

def _normalize_line_list(line: str) -> str:
    # Convertit bullets visuelles en markdown
    m = _bullet_rx.match(line)
    if m:
        tok = m.group(1)
        if tok.isdigit() or tok[:-1].isdigit():
            # numérotée
            return re.sub(r"^\s*[0-9]+[.)]\s+", lambda x: f"{x.group(0).strip()} ", line)
        return re.sub(r"^\s*([•·\-–—\*])\s+", "- ", line)
    return line

def _block_has_table_traits(text: str) -> bool:
    if not text:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    # Heuristiques simples : pipes, +---, colonnes alignées par espaces multiples
    pipes = sum(l.count("|") for l in lines)
    ascii_borders = sum(1 for l in lines if _table_chars.search(l))
    multi_spaces = sum(1 for l in lines if "  " in l)
    return pipes >= 3 or ascii_borders >= 2 or multi_spaces >= int(0.3*len(lines))

def _page_draw_has_grid(page: fitz.Page, rect: fitz.Rect) -> bool:
    # Heuristique : présence de nombreuses lignes horizontales/verticales dans le rect
    try:
        drawings = page.get_drawings()
    except Exception:
        return False
    lines = 0
    for d in drawings:
        for p in d.get("items", []):
            if p[0] == "l":  # line
                (x1, y1), (x2, y2) = p[1], p[2]
                r = fitz.Rect(min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
                if r.intersects(rect) and (abs(y1-y2) < 0.5 or abs(x1-x2) < 0.5):
                    lines += 1
    return lines >= 6  # seuil empirique

def pdf_to_markdown_preserve_layout(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Reconstruit le Markdown en respectant l’ordre des blocs (texte/images) par page.
    Détecte titres/listes/tableaux. Fait de l’OCR sur les blocs image.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_out: List[str] = []
    meta: Dict[str, Any] = {"layout_mode": "preserve", "pages": doc.page_count}
    ocr_pages = 0

    try:
        for pno in range(min(doc.page_count, OCR_MAX_PAGES)):
            page = doc.load_page(pno)
            raw = page.get_text("rawdict")  # contient blocks -> lines -> spans et blocks type=1 (image)
            blocks = raw.get("blocks", [])
            # Collecte tailles de police pour seuils relatifs
            sizes = []
            for b in blocks:
                if b.get("type", 0) == 0:
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            if "size" in s:
                                sizes.append(s["size"])
            median = sorted(sizes)[len(sizes)//2] if sizes else 10.0
            p90 = sorted(sizes)[int(0.9*len(sizes))] if sizes else median

            page_md: List[str] = []
            page_md.append(f"<!-- Page {pno+1} -->")

            for b in blocks:
                btype = b.get("type", 0)
                bbox = b.get("bbox", None)
                rect = fitz.Rect(bbox) if bbox else None

                if btype == 0:
                    # ------- TEXTE -------
                    lines_out: List[str] = []
                    for l in b.get("lines", []):
                        line_txt = ""
                        max_size = 0.0
                        boldish = False
                        for s in l.get("spans", []):
                            txt = s.get("text", "")
                            if not txt:
                                continue
                            size = float(s.get("size", median))
                            max_size = max(max_size, size)
                            # heuristique "gras" : flags & 2 (bold) dans PyMuPDF ? pas toujours porté; on applique simple.
                            boldish = boldish or ("bold" in s.get("font", "").lower())
                            line_txt += txt
                        line_txt = line_txt.rstrip()
                        if not line_txt.strip():
                            continue

                        # Lists
                        if LISTS_DETECT:
                            line_txt = _normalize_line_list(line_txt)

                        # Headings
                        if HEADINGS_DETECT:
                            h = _classify_heading_size(max_size, median, p90)
                            if h > 0:
                                lines_out.append(f"{'#'*h} {line_txt.strip()}")
                                continue

                        lines_out.append(line_txt)

                    text_block = "\n".join(lines_out).strip()
                    if not text_block:
                        continue

                    # Tables ?
                    if TABLES_DETECT and (_block_has_table_traits(text_block) or (rect and _page_draw_has_grid(page, rect))):
                        page_md.append("```text")
                        page_md.append(text_block)
                        page_md.append("```")
                    else:
                        page_md.append(text_block)

                else:
                    # ------- IMAGE -------
                    if not rect:
                        continue
                    # Rendu du crop en image PIL
                    # On rend la page à un DPI suffisant puis on crop
                    scale = OCR_DPI / 72.0
                    mat = fitz.Matrix(scale, scale)
                    pix = page.get_pixmap(matrix=mat, alpha=False, clip=rect)
                    im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    ocr_txt, score = ("", 0.0)
                    if OCR_ENABLED:
                        ocr_txt, score = _ocr_image_best(im, OCR_LANGS)

                    if should_embed_images_for_text(ocr_txt, score):
                        # embed base64
                        im = _pil_resize_max(im, IMG_MAX_WIDTH)
                        data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                        page_md.append(f'![{IMG_ALT_PREFIX} p{pno+1}]({data_uri})')
                    else:
                        # insérer le texte OCR à la position du bloc
                        if ocr_txt:
                            page_md.append(_wrap_tables_as_code(ocr_txt))
                            ocr_pages += 1

            # Nettoyage petit bruit
            page_text = "\n\n".join([t for t in page_md if t.strip()])
            md_out.append(page_text)

        meta["ocr_pages"] = ocr_pages
        return ("\n\n".join(md_out).strip(), meta)
    finally:
        doc.close()

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
  <h1>MarkItDown — Conversion (layout-aware)</h1>
  <p class="muted">Respect de la structure PDF (texte/images), détection de titres/listes/tableaux, OCR contextuel et images base64 si nécessaire.</p>

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
      <label for="forceocr">Forcer OCR global (mode auto)</label>
      <input id="forceocr" type="checkbox" />
    </div>
    <div class="row" style="margin-top:8px; gap:8px; align-items:baseline;">
      <label for="di">Endpoint Azure Document Intelligence</label>
      <input id="di" type="text" placeholder="https://<resource>.cognitiveservices.azure.com/"/>
      <span class="muted">Optionnel</span>
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
    return JSONResponse({"docintel_default": DEFAULT_DOCINTEL_ENDPOINT, "layout_mode": LAYOUT_MODE})

# ---------------------------
# Endpoint API de conversion
# ---------------------------
@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    use_plugins: bool = Form(False),
    docintel_endpoint: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    use_llm: bool = Form(False),
    force_ocr: bool = Form(False),
):
    """
    Deux modes:
    - preserve (par défaut): reconstruit la page PDF bloc par bloc (texte/images), avec OCR ciblé et embed image si OCR faible.
    - auto: pipeline précédent (MarkItDown -> fallback OCR global en append).
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

        metadata: Dict[str, Any] = {}

        if LAYOUT_MODE == "preserve" and is_pdf:
            markdown, meta = pdf_to_markdown_preserve_layout(content)
            metadata.update(meta)
        else:
            # --------- Mode auto (ancien pipeline) ----------
            md = MarkItDown(
                enable_plugins=use_plugins,
                docintel_endpoint=docintel_endpoint
            )
            stream = io.BytesIO(content)
            result = md.convert_stream(stream, file_name=file.filename)
            markdown = getattr(result, "text_content", "") or ""
            meta2 = getattr(result, "metadata", None) or {}
            warnings = getattr(result, "warnings", None)
            if warnings:
                meta2["warnings"] = warnings
            metadata.update(meta2)

            if OCR_ENABLED and (force_ocr or (len(markdown.strip()) < OCR_MIN_CHARS and (is_pdf or is_img))):
                if is_pdf:
                    # OCR global "append"
                    from_page = fitz.open(stream=content, filetype="pdf")
                    try:
                        ocr_md_all: List[str] = []
                        scores_all: List[float] = []
                        for i in range(min(from_page.page_count, OCR_MAX_PAGES)):
                            page = from_page.load_page(i)
                            # raster page @ DPI + OCR
                            scale = OCR_DPI / 72.0
                            mat = fitz.Matrix(scale, scale)
                            pix = page.get_pixmap(matrix=mat, alpha=False)
                            im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            txt, score = _ocr_image_best(im, OCR_LANGS)
                            scores_all.append(score)
                            if txt.strip():
                                ocr_md_all.append(f"\n\n## Page {i+1}\n\n{_wrap_tables_as_code(txt)}")
                        if ocr_md_all:
                            if len(markdown.strip()) < OCR_MIN_CHARS:
                                markdown = "# OCR\n" + "".join(ocr_md_all)
                            else:
                                markdown += "\n\n# OCR (extrait)\n" + "".join(ocr_md_all)
                        metadata["ocr_pages"] = len(scores_all)
                    finally:
                        from_page.close()
                elif is_img:
                    txt, score = _ocr_image_best(Image.open(io.BytesIO(content)), OCR_LANGS)
                    if txt.strip():
                        if len(markdown.strip()) < OCR_MIN_CHARS:
                            markdown = _wrap_tables_as_code(txt)
                        else:
                            markdown += "\n\n# OCR (extrait)\n" + _wrap_tables_as_code(txt)

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
