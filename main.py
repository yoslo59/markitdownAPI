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
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini").strip()
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

# OCR (tunable sans rebuild)
OCR_ENABLED        = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANGS          = os.getenv("OCR_LANGS", "fra+eng").strip()
OCR_DPI            = int(os.getenv("OCR_DPI", "350"))
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "50"))  # ↑ traite plus de pages par défaut
OCR_MIN_CHARS      = int(os.getenv("OCR_MIN_CHARS", "500"))
OCR_MODE           = os.getenv("OCR_MODE", "append").strip()
OCR_KEEP_SPACES    = os.getenv("OCR_KEEP_SPACES", "true").lower() == "true"
OCR_TWO_PASS       = os.getenv("OCR_TWO_PASS", "true").lower() == "true"
OCR_TABLE_MODE     = os.getenv("OCR_TABLE_MODE", "true").lower() == "true"
OCR_PSMS           = [p.strip() for p in os.getenv("OCR_PSMS", "6,4,11").split(",")]     # 6=block, 4=columns, 11=sparse
OCR_DPI_CANDIDATES = [int(x) for x in os.getenv("OCR_DPI_CANDIDATES", "300,350,400").split(",")]

# Short-circuit : si on atteint ce score, on arrête la recherche multi-DPI/PSM pour gagner du temps
OCR_SCORE_GOOD_ENOUGH = float(os.getenv("OCR_SCORE_GOOD_ENOUGH", "0.6"))

# Embedding images base64
# none      : n’embarque aucune image
# ocr_only  : embarque les images uniquement si l’OCR n’est pas exploitable (court/bruyant)
# all       : embarque toutes les images du PDF
EMBED_IMAGES       = os.getenv("EMBED_IMAGES", "ocr_only").strip()  # none | ocr_only | all
IMG_FORMAT         = os.getenv("IMG_FORMAT", "png").strip().lower()  # png | jpeg
IMG_JPEG_QUALITY   = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH      = int(os.getenv("IMG_MAX_WIDTH", "1400"))         # resize max (px), 0 = no limit
IMG_ALT_PREFIX     = os.getenv("IMG_ALT_PREFIX", "Capture").strip()

# (Optionnel) Azure Document Intelligence
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

def _md_cleanup(md: str) -> str:
    """Post-format léger pour la sortie MarkItDown: normalisation listes/espaces/tableaux simples."""
    if not md:
        return md
    lines = []
    for L in md.replace("\r","").split("\n"):
        l = re.sub(r"[ \t]+$", "", L)
        l = re.sub(r"^\s*[•·●◦▪]\s+", "- ", l)      # bullets unicode -> '- '
        l = re.sub(r"^\s*(\d+)[\)\]]\s+", r"\1. ", l)  # "1)" -> "1. "
        lines.append(l)
    txt = "\n".join(lines)

    # bordures ASCII -> fence
    def wrap_ascii(block: str) -> str:
        return "```text\n" + block + "\n```"

    txt = re.sub(
        r"(?:^|\n)([|+\-=_].*?\n)(?:(?:.*[|+\-=_].*\n)+)",
        lambda m: wrap_ascii(m.group(0).strip()),
        txt,
        flags=re.S
    )
    return txt.strip()

# ---------------------------
# Helpers OCR
# ---------------------------
_table_chars = re.compile(r"[|+\-=_]{3,}")  # heuristique ASCII

def _tess_config(psm: str, keep_spaces: bool, table_mode: bool) -> str:
    cfg = f"--psm {psm} --oem 1"  # LSTM
    if keep_spaces:
        cfg += " -c preserve_interword_spaces=1"
    if table_mode:
        cfg += " -c tessedit_write_images=false"
    return cfg

def _preprocess_for_ocr(im: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    # binarisation douce
    g = g.point(lambda p: 255 if p > 190 else (0 if p < 110 else p))
    return g

def _score_text_for_table(txt: str) -> float:
    """Score simple favorisant tableaux/monospace, pénalise bruit évident."""
    if not txt:
        return 0.0
    lines = txt.splitlines()
    n = max(1, len(lines))
    pipes = sum(l.count("|") for l in lines)
    plus  = sum(l.count("+") for l in lines)
    dashes= sum(l.count("-") for l in lines)
    ascii_blocks = sum(1 for l in lines if _table_chars.search(l))
    noise = sum(1 for l in lines if "nnn" in l or "Se ne" in l or "—" in l or "$-----" in l)
    return (pipes*1.0 + plus*0.6 + dashes*0.3 + ascii_blocks*2.0)/n - noise*0.25 + len(txt)/5000.0

def _wrap_tables_as_code(txt: str) -> str:
    """Emballe blocs ASCII en ```text``` pour garder l’alignement Markdown."""
    if not txt:
        return txt
    out, buf, in_blk = [], [], False
    for line in txt.splitlines():
        is_tbl = _table_chars.search(line) is not None or line.strip().startswith("|")
        if is_tbl and not in_blk:
            in_blk = True
            out.append("```text"); buf = []
        if in_blk and not is_tbl and buf:
            out.extend(buf); out.append("```")
            in_blk = False; out.append(line); buf = []; continue
        if in_blk: buf.append(line)
        else: out.append(line)
    if in_blk:
        out.extend(buf); out.append("```")
    return "\n".join(out)

def _ocr_image_best(im: Image.Image, langs: str) -> Tuple[str, float]:
    """Essaie plusieurs PSM, 2 passes (brute + prétraitée). Short-circuit si score OK."""
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    best_txt, best_score = "", -1e9
    for psm in OCR_PSMS:
        cfg = _tess_config(psm, OCR_KEEP_SPACES, OCR_TABLE_MODE)
        # Pass brute
        t1 = pytesseract.image_to_string(im, lang=langs, config=cfg) or ""
        s1 = _score_text_for_table(t1)
        cand_txt, cand_score = t1, s1
        # Pass prétraitée
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

def ocr_image_bytes(img_bytes: bytes, langs: str) -> Tuple[str, float]:
    with Image.open(io.BytesIO(img_bytes)) as im:
        txt, score = _ocr_image_best(im, langs)
        return _wrap_tables_as_code(txt), score

def _raster_pdf_page(page, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# ---------------------------
# Extraction/embedding images utilitaires
# ---------------------------
def _pil_resize_max(im: Image.Image, max_w: int) -> Image.Image:
    if max_w and im.width > max_w:
        ratio = max_w / im.width
        new_h = int(im.height * ratio)
        return im.resize((max_w, new_h), Image.LANCZOS)
    return im

def _pil_to_base64(im: Image.Image, fmt: str = "png", quality: int = 85) -> str:
    buf = io.BytesIO()
    if fmt.lower() in ("jpeg", "jpg"):
        im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def crop_bbox_image(page: fitz.Page, bbox: Tuple[float,float,float,float], dpi: int) -> Optional[Image.Image]:
    try:
        x0,y0,x1,y1 = bbox
        rect = fitz.Rect(x0,y0,x1,y1)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None

# ---------------------------
# Rendu PDF inline (texte + images à la bonne place)
# ---------------------------
# flags MuPDF : BOLD=1, ITALIC=2, MONO=4, SERIF=8, SYMBOL=16, FAKEBOLD=32, FAKEITALIC=64, SMALLCAPS=128
def _is_bold(flags: int) -> bool:
    return bool(flags & 1 or flags & 32)

def _classify_heading(size: float, median_size: float) -> Optional[str]:
    # heuristiques simples basées sur la taille relative
    if median_size <= 0:
        return None
    if size >= median_size * 1.8:
        return "#"
    if size >= median_size * 1.5:
        return "##"
    if size >= median_size * 1.25:
        return "###"
    return None

_bullet_re = re.compile(r"^\s*(?:[-–—•·●◦▪]|\d+[.)])\s+")

def _line_to_md(spans: List[Dict[str,Any]], median_size: float) -> str:
    # concatène les spans en respectant le gras
    parts = []
    max_size = 0.0
    for sp in spans:
        t = sp.get("text","")
        if not t: 
            continue
        size = float(sp.get("size", 0))
        max_size = max(max_size, size)
        if _is_bold(int(sp.get("flags",0))):
            parts.append(f"**{t}**")
        else:
            parts.append(t)
    raw = "".join(parts).strip()
    if not raw:
        return ""
    # heading ?
    h = _classify_heading(max_size, median_size)
    if h and len(raw) < 180:  # pas de Hx pour les lignes très longues
        return f"{h} {raw}"
    # liste ?
    if _bullet_re.match(raw):
        return f"{raw}"
    return raw

def _median_font_size(page_raw: Dict[str,Any]) -> float:
    sizes = []
    for b in page_raw.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if s.get("text","").strip():
                    sizes.append(float(s.get("size",0)))
    if not sizes:
        return 0.0
    sizes.sort()
    mid = len(sizes)//2
    return (sizes[mid] if len(sizes)%2==1 else (sizes[mid-1]+sizes[mid])/2.0)

def render_pdf_markdown_inline(pdf_bytes: bytes) -> Tuple[str, Dict[str,Any]]:
    """
    Reconstruit le Markdown d'un PDF en parcourant les blocks (texte/images)
    dans l'ordre visuel. Images : OCR sinon base64 inline.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_lines: List[str] = []
    meta: Dict[str,Any] = {"engine": "pymupdf_inline", "pages": doc.page_count}
    try:
        total_pages = min(doc.page_count, OCR_MAX_PAGES)
        for p in range(total_pages):
            page = doc.load_page(p)
            raw = page.get_text("rawdict") or {}
            median_size = _median_font_size(raw)
            blocks = raw.get("blocks", [])
            # trier par y puis x (robuste)
            blocks.sort(key=lambda b: (b.get("bbox", [0,0,0,0])[1], b.get("bbox", [0,0,0,0])[0]))

            # Option : ajouter un titre de page si besoin
            # md_lines.append(f"\n<!-- Page {p+1} -->")

            for b in blocks:
                btype = b.get("type", 0)
                bbox = tuple(b.get("bbox", (0,0,0,0)))
                if btype == 0:
                    # bloc texte
                    para_buf = []
                    for line in b.get("lines", []):
                        md_line = _line_to_md(line.get("spans", []), median_size)
                        if md_line:
                            para_buf.append(md_line)
                    if para_buf:
                        # heuristique tableaux ASCII
                        block_txt = "\n".join(para_buf)
                        if _table_chars.search(block_txt):
                            md_lines.append("```text")
                            md_lines.append(block_txt)
                            md_lines.append("```")
                        else:
                            md_lines.append("\n".join(para_buf))
                elif btype == 1:
                    # bloc image -> OCR sinon base64
                    im = crop_bbox_image(page, bbox, OCR_DPI)
                    if im is None:
                        # fallback via xref si possible
                        info = b.get("image")
                        if isinstance(info, dict) and "xref" in info:
                            try:
                                pix = fitz.Pixmap(doc, info["xref"])
                                if pix.n >= 4:
                                    pix = fitz.Pixmap(fitz.csRGB, pix)
                                im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            except Exception:
                                im = None
                    if im is None:
                        continue
                    im = _pil_resize_max(im, IMG_MAX_WIDTH)

                    txt, score = _ocr_image_best(im, OCR_LANGS)
                    if txt and score >= OCR_SCORE_GOOD_ENOUGH:
                        md_lines.append(_wrap_tables_as_code(txt))
                    else:
                        if EMBED_IMAGES in ("all","ocr_only"):
                            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            md_lines.append(f'![{IMG_ALT_PREFIX} p{p+1}]({data_uri})')

        return ("\n\n".join([l for l in md_lines if l.strip()]).strip() or "", meta)
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
  <h1>MarkItDown — Conversion</h1>
  <p class="muted">Plugins seul : texte via MarkItDown. Plugins + Forcer OCR : rendu PDF inline (texte + OCR/images) au bon endroit.</p>

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
      <span class="muted">Optionnel (utilisé par MarkItDown seulement)</span>
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
    - Plugins seul : MarkItDown → texte, post-format léger. (Aucune image)
    - Plugins + Forcer OCR (PDF) : rendu inline PyMuPDF => texte + OCR / base64 aux bons emplacements.
    - Images seules (PNG/JPG) : OCR + éventuel embed image selon EMBED_IMAGES.
    """
    try:
        if not docintel_endpoint:
            docintel_endpoint = DEFAULT_DOCINTEL_ENDPOINT

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Fichier vide")

        # Save input
        in_path = None
        if SAVE_UPLOADS:
            in_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(in_path, "wb") as f:
                f.write(content)

        is_pdf = guess_is_pdf(file.filename, file.content_type)
        is_img = guess_is_image(file.filename, file.content_type)

        metadata: Dict[str,Any] = {}

        # === Cas PDF + Plugins + Forcer OCR  => pipeline PyMuPDF inline (texte + images à leur place)
        if is_pdf and use_plugins and force_ocr and OCR_ENABLED:
            markdown, meta_pdf = render_pdf_markdown_inline(content)
            metadata.update(meta_pdf)

        else:
            # === Pipeline MarkItDown (texte)
            if not use_plugins:
                raise HTTPException(status_code=400, detail="Coche 'Activer plugins MarkItDown' pour extraire le texte.")
            md_engine = MarkItDown(enable_plugins=True, docintel_endpoint=docintel_endpoint)
            result = md_engine.convert_stream(io.BytesIO(content), file_name=file.filename)

            markdown = getattr(result, "text_content", "") or ""
            metadata.update(getattr(result, "metadata", {}) or {})
            warnings = getattr(result, "warnings", None)
            if warnings:
                metadata["warnings"] = warnings

            # post-traitement léger Markdown
            markdown = _md_cleanup(markdown)

            # === OCR fallback / images uniquement si on a coché Forcer OCR (image file) ===
            if force_ocr and OCR_ENABLED and is_img:
                ocr_text, score = ocr_image_bytes(content, OCR_LANGS)
                if ocr_text.strip():
                    markdown += "\n\n# OCR (extrait)\n" + ocr_text
                if EMBED_IMAGES in ("all", "ocr_only") and (not ocr_text or score < OCR_SCORE_GOOD_ENOUGH):
                    try:
                        with Image.open(io.BytesIO(content)) as im:
                            im = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            markdown += f'\n\n![{IMG_ALT_PREFIX}]({data_uri})\n'
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
