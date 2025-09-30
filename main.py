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
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini").strip()
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

# OCR (tunable)
OCR_ENABLED        = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANGS          = os.getenv("OCR_LANGS", "fra+eng").strip()
OCR_DPI            = int(os.getenv("OCR_DPI", "350"))
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "200"))  # laisse grand, on arrêtera au doc.page_count
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

# (Optionnel) Azure Document Intelligence pour MarkItDown (non utilisé pour PDF ici)
DEFAULT_DOCINTEL_ENDPOINT = os.getenv("DEFAULT_DOCINTEL_ENDPOINT", "").strip()

# Dossiers persistants
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="3.1")

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
# Utils
# ---------------------------
def guess_is_pdf(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower() in ("application/pdf", "pdf"):
        return True
    return filename.lower().endswith(".pdf")

def guess_is_image(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    return any(filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))

def _md_escape(s: str) -> str:
    return s.replace("\r", "")

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
    g = g.point(lambda p: 255 if p > 190 else (0 if p < 110 else p))
    return g

def _score_text_for_table(txt: str) -> float:
    if not txt:
        return 0.0
    lines = txt.splitlines()
    n = max(1, len(lines))
    pipes = sum(l.count("|") for l in lines)
    ascii_blocks = sum(1 for l in lines if _table_chars.search(l))
    noise = sum(1 for l in lines if "nnn" in l or "—" in l or "$---" in l or "Se ne" in l)
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

def resize_max(im: Image.Image, max_w: int) -> Image.Image:
    if max_w and im.width > max_w:
        r = max_w / im.width
        return im.resize((max_w, int(im.height*r)), Image.LANCZOS)
    return im

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

# ---------------------------
# PyMuPDF helpers (blocs PDF)
# ---------------------------
BulletStart = re.compile(r"^\s*([•\-–‣●◦▪]|(\d+[\.\)]))\s+")
HasLetter = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")

def _get_page_items(page: fitz.Page) -> List[Dict[str, Any]]:
    """Retourne items ordonnés: text / image / table avec bbox."""
    items: List[Dict[str, Any]] = []

    # 1) Texte & images depuis rawdict (avec bbox)
    rd = page.get_text("rawdict")
    for b in rd.get("blocks", []):
        tp = b.get("type", 0)
        bbox = tuple(b.get("bbox", (0,0,0,0)))
        if tp == 0:
            # reconstruire texte brut du bloc + récupérer tailles de spans
            lines, sizes, fonts = [], [], []
            for l in b.get("lines", []):
                seg = []
                for s in l.get("spans", []):
                    t = s.get("text", "")
                    if t:
                        seg.append(t)
                        sizes.append(float(s.get("size", 0)))
                        f = s.get("font", "")
                        if f: fonts.append(f)
                if seg:
                    lines.append("".join(seg))
            txt = "\n".join(lines).strip()
            if txt:
                items.append({"kind":"text", "bbox":bbox, "text":txt, "sizes":sizes, "fonts":fonts})
        elif tp == 1:
            xref = None
            imginfo = b.get("image")
            if isinstance(imginfo, dict):
                xref = imginfo.get("xref")
            items.append({"kind":"image", "bbox":bbox, "xref":xref})

    # 2) Tables détectées
    try:
        tbls = page.find_tables()
        for t in tbls.tables:
            items.append({"kind":"table", "bbox":tuple(t.bbox), "table": t.extract()})
    except Exception:
        pass

    # 3) Tri de lecture (haut-gauche → bas-droite)
    def key_it(it):
        x0,y0,x1,y1 = it["bbox"]
        return (y0, x0, y1, x1)
    items.sort(key=key_it)

    # 4) Supprimer le texte qui tombe "dans" des tables (pour éviter le doublon)
    table_rects = [fitz.Rect(*it["bbox"]) for it in items if it["kind"]=="table"]
    if table_rects:
        kept = []
        for it in items:
            if it["kind"]=="text":
                rect = fitz.Rect(*it["bbox"])
                if any(r.contains(rect) or r.intersects(rect) for r in table_rects):
                    continue
            kept.append(it)
        items = kept

    return items

def _sizes_stats(items: List[List[Dict[str,Any]]]) -> float:
    """Renvoie la taille de police 'corps' (médiane des spans contenant des lettres) sur tout le document."""
    all_sizes = []
    for page_items in items:
        for it in page_items:
            if it["kind"]=="text" and it.get("sizes"):
                if HasLetter.search(it.get("text","")):
                    all_sizes.extend(it["sizes"])
    if not all_sizes:
        return 12.0
    all_sizes.sort()
    mid = len(all_sizes)//2
    return float(all_sizes[mid]) if len(all_sizes)%2==1 else float((all_sizes[mid-1]+all_sizes[mid])/2)

def _classify_heading(line: str, block_sizes: List[float], body_size: float, fonts: List[str]) -> Optional[str]:
    """Retourne 'H1'/'H2'/'H3' ou None selon taille / bold / présence de lettres."""
    if not HasLetter.search(line):
        return None
    if not block_sizes:
        return None
    max_size = max(block_sizes)
    # bold heuristique
    is_bold = any("Bold" in f or "Semibold" in f or "Black" in f for f in fonts)
    # seuils
    if max_size >= body_size*1.7 or (is_bold and max_size >= body_size*1.5):
        return "H1"
    if max_size >= body_size*1.45 or (is_bold and max_size >= body_size*1.3):
        return "H2"
    if max_size >= body_size*1.25:
        return "H3"
    return None

def _format_text_block(text: str, sizes: List[float], fonts: List[str], body_size: float) -> str:
    """Transforme un bloc texte en Markdown (titres/listes/paragraphes)."""
    text = _md_escape(text)
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        # listes (puces / numérotées)
        m = BulletStart.match(line)
        if m:
            # normaliser la puce
            core = line[m.end():].strip()
            out.append(f"- {core}")
            continue
        # potentiellement un titre
        head = _classify_heading(line, sizes, body_size, fonts)
        if head == "H1":
            out.append(f"# {line}")
            continue
        if head == "H2":
            out.append(f"## {line}")
            continue
        if head == "H3":
            out.append(f"### {line}")
            continue
        out.append(line)
    return "\n".join(out)

def _markdown_table_from_cells(cells: List[List[str]]) -> str:
    if not cells or not any(row for row in cells):
        return ""
    rows = [[(c or "").strip() for c in row] for row in cells]
    header = rows[0]
    body = rows[1:] if len(rows)>1 else []
    # sécuriser pipes
    header = [h.replace("|","\\|") for h in header]
    body = [[c.replace("|","\\|") for c in r] for r in body]
    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"]*len(header)) + " |")
    for r in body:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)

def _crop_region_image(page: fitz.Page, bbox: Tuple[float,float,float,float], dpi: int) -> Optional[Image.Image]:
    try:
        x0,y0,x1,y1 = bbox
        rect = fitz.Rect(x0,y0,x1,y1)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None

# ---------------------------
# Construction Markdown (PDF)
# ---------------------------
def build_pdf_markdown(pdf_bytes: bytes, want_text: bool, want_inline_images: bool, want_ocr: bool) -> Tuple[str, Dict[str,Any]]:
    """
    Construit un Markdown final en respectant l'ordre des blocs:
      - texte → titres / listes / paragraphes
      - table → table Markdown
      - image → OCR in-place ou image base64 (selon score / EMBED_IMAGES)
    want_text:       inclure le texte
    want_inline_images: traiter et inclure les blocs image (embed / OCR)
    want_ocr:        si True tente l’OCR sur image, sinon image seule (selon EMBED_IMAGES)
    """
    meta: Dict[str,Any] = {"engine":"pymupdf_layout"}
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        # Pré-collecte pour déterminer la taille "corps"
        all_items = []
        total_pages = doc.page_count
        maxp = min(total_pages, OCR_MAX_PAGES)
        for p in range(maxp):
            all_items.append(_get_page_items(doc.load_page(p)))
        body_size = _sizes_stats(all_items)
        meta["body_font_size"] = body_size

        pages_md = []
        for pno in range(maxp):
            page = doc.load_page(pno)
            items = all_items[pno]
            lines: List[str] = [f"## Page {pno+1}"]
            for it in items:
                kind = it["kind"]
                bbox = it["bbox"]

                if kind == "text" and want_text:
                    lines.append(_format_text_block(it["text"], it.get("sizes",[]), it.get("fonts",[]), body_size))

                elif kind == "table" and want_text:
                    table = it.get("table") or {}
                    cells = table.get("cells") if isinstance(table, dict) else None
                    if not cells and hasattr(table, "extract"):
                        try:
                            cells = table.extract().get("cells")
                        except Exception:
                            cells = None
                    if cells:
                        # convert list of lists-of-strings from extract()
                        md_tbl = _markdown_table_from_cells(cells)
                        if md_tbl:
                            lines.append(md_tbl)

                elif kind == "image" and want_inline_images:
                    # image native / crop bbox fallback
                    im = None
                    if it.get("xref"):
                        try:
                            pix = fitz.Pixmap(doc, it["xref"])
                            if pix.n >= 4:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        except Exception:
                            im = None
                    if im is None:
                        im = _crop_region_image(page, bbox, OCR_DPI)
                    if im is None:
                        continue
                    im = resize_max(im, IMG_MAX_WIDTH)

                    if want_ocr and OCR_ENABLED:
                        txt, score = ocr_pil_best(im)
                        if txt.strip() and (score >= OCR_SCORE_GOOD_ENOUGH or len(txt) >= OCR_MIN_CHARS//4):
                            lines.append(txt)
                        else:
                            if EMBED_IMAGES in ("all","ocr_only"):
                                lines.append(f'![{IMG_ALT_PREFIX} p{pno+1}]({pil_to_b64(im)})')
                    else:
                        if EMBED_IMAGES != "none":
                            lines.append(f'![{IMG_ALT_PREFIX} p{pno+1}]({pil_to_b64(im)})')

            pages_md.append("\n\n".join([l for l in lines if l is not None]).strip())

        return ("\n\n".join(pages_md).strip(), meta)

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
  <h1>Conversion → Markdown</h1>
  <p class="muted">PDF : reconstruction par blocs (titres/listes/tables). « Forcer OCR » : texte des images (ou image base64) inséré <em>in-place</em>. Autres formats : MarkItDown.</p>

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
      <span class="muted">Optionnel (MarkItDown, non utilisé pour PDF)</span>
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
fetch("/config").then(r=>r.ok?r.json():null).then(j=>{ if(j && j.docintel_default){ $("di").value = j.docintel_default; } }).catch(()=>{});

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
    use_plugins: bool = Form(False),          # pour DOCX/PPTX/XLSX/HTML → MarkItDown
    docintel_endpoint: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),    # compat
    use_llm: bool = Form(False),
    force_ocr: bool = Form(False),
):
    """
    PDF:
      - plugins cochés, OCR décoché   → texte seul (titres/listes/tables), pas d'images.
      - plugins cochés + Forcer OCR   → texte + OCR des images in-place (ou image base64 si OCR faible).
    Autres formats → MarkItDown.
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
        markdown = ""

        if is_pdf:
            # Case matrix:
            # - use_plugins True & !force_ocr  => texte seul
            # - use_plugins True & force_ocr   => texte + images (OCR/BASE64) in-place
            want_text = True
            want_inline_images = bool(force_ocr)   # images seulement si OCR coché
            want_ocr = bool(force_ocr)
            markdown, meta = build_pdf_markdown(content, want_text, want_inline_images, want_ocr)
            metadata.update(meta)

        elif is_img:
            # Image standalone
            body = []
            try:
                with Image.open(io.BytesIO(content)) as im:
                    im = resize_max(im, IMG_MAX_WIDTH)
                    if OCR_ENABLED and force_ocr:
                        txt, score = ocr_pil_best(im)
                        if txt.strip() and (score >= OCR_SCORE_GOOD_ENOUGH or len(txt) >= OCR_MIN_CHARS//4):
                            body.append("# OCR (image)\n" + txt)
                        else:
                            if EMBED_IMAGES != "none":
                                body.append(f'![{IMG_ALT_PREFIX}]({pil_to_b64(im)})')
                    else:
                        if EMBED_IMAGES != "none":
                            body.append(f'![{IMG_ALT_PREFIX}]({pil_to_b64(im)})')
            except Exception:
                pass
            markdown = "\n\n".join(body).strip() or "(image)"

        else:
            # Tous les autres formats → MarkItDown
            md = MarkItDown(enable_plugins=use_plugins, docintel_endpoint=docintel_endpoint)
            res = md.convert_stream(io.BytesIO(content), file_name=file.filename)
            markdown = (getattr(res, "text_content", "") or "").strip()
            metadata.update(getattr(res, "metadata", {}) or {})

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
                    metadata["azure_summary"] = getattr(msg, "content", None) or "[Résumé vide]"
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
