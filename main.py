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

# PDF engine: hybrid | markitdown
PDF_MODE = os.getenv("PDF_MODE", "hybrid").strip().lower()

# Azure OpenAI (optionnel pour résumé)
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_KEY        = os.getenv("AZURE_OPENAI_KEY", "").strip()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini").strip()
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

# OCR (tunable sans rebuild)
OCR_ENABLED        = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANGS          = os.getenv("OCR_LANGS", "fra+eng").strip()
OCR_DPI            = int(os.getenv("OCR_DPI", "350"))
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "50"))
OCR_MIN_CHARS      = int(os.getenv("OCR_MIN_CHARS", "500"))
OCR_KEEP_SPACES    = os.getenv("OCR_KEEP_SPACES", "true").lower() == "true"
OCR_TWO_PASS       = os.getenv("OCR_TWO_PASS", "true").lower() == "true"
OCR_TABLE_MODE     = os.getenv("OCR_TABLE_MODE", "true").lower() == "true"
OCR_PSMS           = [p.strip() for p in os.getenv("OCR_PSMS", "6,4,11").split(",")]  # 6=block, 4=columns, 11=sparse
OCR_SCORE_GOOD_ENOUGH = float(os.getenv("OCR_SCORE_GOOD_ENOUGH", "0.6"))

# Embedding images base64 (utilisé en mode OCR sur PDF)
# none | ocr_only | all
EMBED_IMAGES       = os.getenv("EMBED_IMAGES", "ocr_only").strip().lower()
IMG_FORMAT         = os.getenv("IMG_FORMAT", "png").strip().lower()  # png | jpeg
IMG_JPEG_QUALITY   = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH      = int(os.getenv("IMG_MAX_WIDTH", "1400"))
IMG_ALT_PREFIX     = os.getenv("IMG_ALT_PREFIX", "Capture").strip()

# (Optionnel) Azure Document Intelligence (non utilisé ici)
DEFAULT_DOCINTEL_ENDPOINT = os.getenv("DEFAULT_DOCINTEL_ENDPOINT", "").strip()

# Dossiers persistants
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------
# App
# ---------------------------
app = FastAPI(title="MarkItDown API", version="4.0")

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
BULLET_RE = re.compile(r"^(\u2022|•|·|–|-|\*|\d+[\.\)])\s+")

CODE_FONT_HINTS = ("mono", "courier", "consolas", "dejavusansmono", "menlo")

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

def _is_mono(span: Dict[str, Any]) -> bool:
    font = (span.get("font") or "").lower()
    return any(h in font for h in CODE_FONT_HINTS)

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

def ocr_image_best(im: Image.Image, langs: str) -> Tuple[str, float]:
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
    return _wrap_tables_as_code(best_txt.strip()), best_score


# ---------------------------
# PDF parsing (HYBRID)
# ---------------------------
def _median_font_size(doc: fitz.Document, sample_pages: int = 10) -> float:
    sizes: List[float] = []
    for i in range(min(doc.page_count, sample_pages)):
        raw = doc.load_page(i).get_text("rawdict")
        for b in raw.get("blocks", []):
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    size = float(span.get("size", 0) or 0)
                    if size > 0:
                        sizes.append(size)
    return statistics.median(sizes) if sizes else 12.0

def _heading_level(max_size: float, median: float) -> Optional[int]:
    if max_size >= median * 1.45:
        return 1
    if max_size >= median * 1.25:
        return 2
    if max_size >= median * 1.15:
        return 3
    return None

def _bbox_key(bbox):
    x0, y0, x1, y1 = bbox
    return (round(y0, 2), round(x0, 2))

def _emit_list_or_para(lines: List[str]) -> str:
    if not lines:
        return ""
    bullet_like = sum(1 for l in lines if BULLET_RE.match(l.strip()))
    if bullet_like >= max(2, len(lines)//2):
        items = []
        for l in lines:
            m = BULLET_RE.match(l.strip())
            if m:
                items.append("- " + l.strip()[len(m.group(0)):].strip())
            else:
                items.append(l.strip())
        return "\n".join(items)
    return " ".join(s.strip() for s in lines if s.strip())

def _render_clip(page: fitz.Page, bbox, dpi: int) -> Image.Image:
    rect = fitz.Rect(bbox)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _page_tables_map(page: fitz.Page):
    """Retourne la liste des tables (markdown + bbox) si disponible."""
    results = []
    try:
        finder = getattr(page, "find_tables", None)
        if callable(finder):
            found = page.find_tables()
            for t in getattr(found, "tables", []) or []:
                md = t.to_markdown()
                bbox = getattr(t, "bbox", None)
                if md and md.strip():
                    results.append({"md": md.strip(), "bbox": bbox})
    except Exception:
        pass
    return results

def _overlap_y(a, b) -> float:
    """Rapport de recouvrement vertical entre 2 bbox (0..1)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    top = max(ay0, by0); bot = min(ay1, by1)
    inter = max(0.0, bot - top)
    h = max(1e-6, min(ay1 - ay0, by1 - by0))
    return inter / h

def pdf_hybrid_text_only(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """Texte uniquement (titres/listes/tableaux). Pas d'images, pas d'OCR."""
    meta: Dict[str, Any] = {"mode": "pdf_hybrid_text_only"}
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        median_size = _median_font_size(doc)
        meta["median_font_size"] = median_size
        pages_out: List[str] = []

        for pi in range(min(doc.page_count, OCR_MAX_PAGES)):
            page = doc.load_page(pi)
            raw = page.get_text("rawdict")
            blocks = sorted(raw.get("blocks", []), key=lambda b: _bbox_key(b.get("bbox", (0,0,0,0))))
            tables = _page_tables_map(page)

            page_segments: List[str] = []
            used_table = set()

            for bi, b in enumerate(blocks):
                btype = b.get("type", 0)
                bbox = b.get("bbox", (0,0,0,0))

                if btype != 0:
                    # on ignore les images en "texte seulement"
                    continue

                # Si le bloc chevauche fortement une table connue → insérer la table (une seule fois)
                inserted_tbl = False
                for ti, tb in enumerate(tables):
                    tbbox = tb.get("bbox")
                    if tbbox and _overlap_y(bbox, tbbox) > 0.6 and ti not in used_table:
                        page_segments.append(tb["md"])
                        used_table.add(ti)
                        inserted_tbl = True
                        break
                if inserted_tbl:
                    continue

                # Sinon on reconstruit le bloc texte
                lines_collect: List[str] = []
                max_span_size = 0.0
                any_mono = False

                for line in b.get("lines", []):
                    segs = []
                    for span in line.get("spans", []):
                        txt = span.get("text") or ""
                        if not txt:
                            continue
                        segs.append(txt)
                        size = float(span.get("size", 0) or 0)
                        if size > max_span_size:
                            max_span_size = size
                        if _is_mono(span):
                            any_mono = True
                    if segs:
                        line_txt = "".join(segs).rstrip()
                        if line_txt:
                            lines_collect.append(line_txt)

                if not lines_collect:
                    continue

                lvl = _heading_level(max_span_size, median_size)
                if lvl:
                    text = " ".join(s.strip() for s in lines_collect).strip()
                    if text:
                        page_segments.append("#"*lvl + " " + text)
                    continue

                if any_mono:
                    code_text = "\n".join(lines_collect).rstrip()
                    if code_text:
                        page_segments.append("```text\n" + code_text + "\n```")
                    continue

                block_text = "\n".join(lines_collect).strip()
                # heuristique ASCII table si pas de table structurée détectée
                is_tableish = False
                if block_text:
                    lines = [l for l in block_text.splitlines() if l.strip()]
                    pipes = sum(l.count("|") for l in lines)
                    ascii_borders = sum(1 for l in lines if _table_chars.search(l))
                    if pipes >= 3 or ascii_borders >= 2:
                        is_tableish = True

                if is_tableish:
                    page_segments.append("```text\n" + block_text + "\n```")
                else:
                    page_segments.append(_emit_list_or_para(lines_collect))

            page_md = "\n\n".join(seg for seg in page_segments if seg and seg.strip())
            if page_md:
                pages_out.append(page_md)

        return ("\n\n".join(pages_out).strip(), meta)

    finally:
        doc.close()

def pdf_hybrid_text_plus_ocr(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """Texte + OCR des blocs image. Si OCR pauvre → image base64. Insertion à l'emplacement des images."""
    meta: Dict[str, Any] = {"mode": "pdf_hybrid_text_plus_ocr", "ocr_langs": OCR_LANGS, "embed_images": EMBED_IMAGES}
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        median_size = _median_font_size(doc)
        meta["median_font_size"] = median_size
        pages_out: List[str] = []

        for pi in range(min(doc.page_count, OCR_MAX_PAGES)):
            page = doc.load_page(pi)
            raw = page.get_text("rawdict")
            blocks = sorted(raw.get("blocks", []), key=lambda b: _bbox_key(b.get("bbox", (0,0,0,0))))
            tables = _page_tables_map(page)
            used_table = set()

            page_segments: List[str] = []

            # On respecte strictement l'ordre des blocs (texte et images)
            for bi, b in enumerate(blocks):
                btype = b.get("type", 0)
                bbox = b.get("bbox", (0,0,0,0))

                if btype == 1:  # image → OCR ou image base64
                    try:
                        im = _render_clip(page, bbox, OCR_DPI)
                        used_text = False
                        ocr_txt, ocr_score = ("", 0.0)
                        if OCR_ENABLED:
                            ocr_txt, ocr_score = ocr_image_best(im, OCR_LANGS)
                            if ocr_txt and (len(ocr_txt) >= 25 or ocr_score >= OCR_SCORE_GOOD_ENOUGH):
                                page_segments.append(ocr_txt)
                                used_text = True

                        if EMBED_IMAGES == "all" or (EMBED_IMAGES == "ocr_only" and not used_text):
                            im2 = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_base64(im2, IMG_FORMAT, IMG_JPEG_QUALITY)
                            page_segments.append(f"![{IMG_ALT_PREFIX} p{pi+1}]({data_uri})")
                    except Exception:
                        # on n’interrompt pas le flux en cas d’erreur sur un bloc image
                        continue
                    continue

                if btype != 0:
                    continue

                # Bloc texte : d'abord, si recouvrement fort avec une table détectée → insérer la table
                inserted_tbl = False
                for ti, tb in enumerate(tables):
                    tbbox = tb.get("bbox")
                    if tbbox and _overlap_y(bbox, tbbox) > 0.6 and ti not in used_table:
                        page_segments.append(tb["md"])
                        used_table.add(ti)
                        inserted_tbl = True
                        break
                if inserted_tbl:
                    continue

                # Reconstruction du texte
                lines_collect: List[str] = []
                max_span_size = 0.0
                any_mono = False

                for line in b.get("lines", []):
                    segs = []
                    for span in line.get("spans", []):
                        txt = span.get("text") or ""
                        if not txt:
                            continue
                        segs.append(txt)
                        size = float(span.get("size", 0) or 0)
                        if size > max_span_size:
                            max_span_size = size
                        if _is_mono(span):
                            any_mono = True
                    if segs:
                        line_txt = "".join(segs).rstrip()
                        if line_txt:
                            lines_collect.append(line_txt)

                if not lines_collect:
                    continue

                lvl = _heading_level(max_span_size, median_size)
                if lvl:
                    text = " ".join(s.strip() for s in lines_collect).strip()
                    if text:
                        page_segments.append("#"*lvl + " " + text)
                    continue

                if any_mono:
                    code_text = "\n".join(lines_collect).rstrip()
                    if code_text:
                        page_segments.append("```text\n" + code_text + "\n```")
                    continue

                block_text = "\n".join(lines_collect).strip()
                # Heuristique ASCII table si rien de structuré
                is_tableish = False
                if block_text:
                    lines = [l for l in block_text.splitlines() if l.strip()]
                    pipes = sum(l.count("|") for l in lines)
                    ascii_borders = sum(1 for l in lines if _table_chars.search(l))
                    if pipes >= 3 or ascii_borders >= 2:
                        is_tableish = True

                if is_tableish:
                    page_segments.append("```text\n" + block_text + "\n```")
                else:
                    page_segments.append(_emit_list_or_para(lines_collect))

            page_md = "\n\n".join(seg for seg in page_segments if seg and seg.strip())
            if page_md:
                pages_out.append(page_md)

        return ("\n\n".join(pages_out).strip(), meta)

    finally:
        doc.close()


# ---------------------------
# Mini interface web (inchangée)
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
  <p class="muted">PDF (mode hybrid): “Activer plugins MarkItDown” = texte seul (titres/listes/tableaux). + “Forcer OCR” = OCR des images et insertion du texte à l’emplacement; sinon image base64.</p>

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
      <label for="forceocr">Forcer OCR (PDF)</label>
      <input id="forceocr" type="checkbox" />
    </div>
    <div class="row" style="margin-top:8px; gap:8px; align-items:baseline;">
      <label for="di">Endpoint Azure Document Intelligence</label>
      <input id="di" type="text" placeholder="https://<resource>.cognitiveservices.azure.com/"/>
      <span class="muted">Optionnel (non utilisé par le mode hybrid)</span>
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


# ---------------------------
# Endpoint API de conversion
# ---------------------------
@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    use_plugins: bool = Form(False),           # "Activer plugins MarkItDown"
    docintel_endpoint: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),     # ignoré
    use_llm: bool = Form(False),
    force_ocr: bool = Form(False),             # "Forcer OCR"
):
    """
    Comportement souhaité :
    - PDF + use_plugins + !force_ocr -> texte seul (titres/listes/tableaux) via HYBRID
    - PDF + use_plugins + force_ocr  -> texte + OCR images (ou base64 si OCR pauvre) via HYBRID
    - Autres formats -> MarkItDown natif
    """
    try:
        content = await file.read()

        # Save input
        in_path = None
        if SAVE_UPLOADS:
            in_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(in_path, "wb") as f:
                f.write(content)

        is_pdf = guess_is_pdf(file.filename, file.content_type)
        is_img = guess_is_image(file.filename, file.content_type)

        metadata: Dict[str, Any] = {"source_filename": file.filename, "pdf_mode": PDF_MODE}
        markdown = ""

        if is_pdf and use_plugins:
            if PDF_MODE == "hybrid":
                if not force_ocr:
                    md, meta = pdf_hybrid_text_only(content)
                else:
                    md, meta = pdf_hybrid_text_plus_ocr(content)
                markdown = md
                metadata.update(meta)
            else:
                # PDF_MODE = markitdown : laisser MarkItDown gérer le PDF (moins fidèle sur structure/ocr)
                md_conv = MarkItDown(enable_plugins=True, docintel_endpoint=None)
                result = md_conv.convert_stream(io.BytesIO(content), file_name=file.filename)
                markdown = getattr(result, "text_content", "") or ""
                metadata.update(getattr(result, "metadata", {}) or {})

        else:
            # Non-PDF (ou plugins décochés) : MarkItDown standard
            md_conv = MarkItDown(enable_plugins=use_plugins, docintel_endpoint=None)
            result = md_conv.convert_stream(io.BytesIO(content), file_name=file.filename)
            markdown = getattr(result, "text_content", "") or ""
            metadata.update(getattr(result, "metadata", {}) or {})

            # Fichiers image : OCR si force_ocr
            if is_img and OCR_ENABLED and force_ocr:
                try:
                    with Image.open(io.BytesIO(content)) as im:
                        txt, score = ocr_image_best(im, OCR_LANGS)
                        if txt.strip():
                            markdown = (markdown + "\n\n# OCR (extrait)\n" + txt).strip() if markdown.strip() else txt
                        if EMBED_IMAGES in ("all", "ocr_only") and (EMBED_IMAGES == "all" or not txt.strip()):
                            im2 = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_base64(im2, IMG_FORMAT, IMG_JPEG_QUALITY)
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
                    metadata["azure_summary"] = content_msg or "[Résumé vide]"
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
