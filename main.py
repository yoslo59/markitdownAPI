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

# Azure OpenAI
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_KEY        = os.getenv("AZURE_OPENAI_KEY", "").strip()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
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

# Embedding images base64
# none      : n’embarque aucune image
# ocr_only  : embarque les images uniquement si l’OCR n’est pas exploitable
# all       : embarque toutes les images du PDF
EMBED_IMAGES       = os.getenv("EMBED_IMAGES", "ocr_only").strip()  # none | ocr_only | all
IMG_FORMAT         = os.getenv("IMG_FORMAT", "png").strip().lower()  # png | jpeg
IMG_JPEG_QUALITY   = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH      = int(os.getenv("IMG_MAX_WIDTH", "1400"))
IMG_ALT_PREFIX     = os.getenv("IMG_ALT_PREFIX", "Capture").strip()

# (Optionnel) Azure Document Intelligence
DEFAULT_DOCINTEL_ENDPOINT = os.getenv("DEFAULT_DOCINTEL_ENDPOINT", "").strip()

# Dossiers persistants
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App
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
    noise = sum(1 for l in lines if "nnn" in l or "Se ne" in l or "—" in l or "$-----" in l)
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
        # Brute
        t1 = pytesseract.image_to_string(im, lang=langs, config=cfg) or ""
        s1 = _score_text_for_table(t1)
        cand_txt, cand_score = t1, s1
        # Prétraitée
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
# PDF layout extraction (texte + images à la bonne place)
# ---------------------------
BULLET_RE = re.compile(r"^(\u2022|•|-|\*|\d+[\.\)])\s+")
CODE_FONT_HINTS = ("Mono", "Courier", "Consolas", "DejaVuSansMono", "Menlo")

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

def _is_mono(span: Dict[str, Any]) -> bool:
    font = (span.get("font") or "").lower()
    return any(h.lower() in font for h in CODE_FONT_HINTS)

def _is_bold(span: Dict[str, Any]) -> bool:
    font = (span.get("font") or "").lower()
    return "bold" in font or "semibold" in font

def _heading_level_for_block(max_size: float, median_size: float) -> Optional[int]:
    # heuristiques simples
    if max_size >= median_size * 1.45:
        return 1
    if max_size >= median_size * 1.25:
        return 2
    if max_size >= median_size * 1.15:
        return 3
    return None

def _bbox_key(bbox):
    # tri lecture haut -> bas, puis gauche -> droite
    x0, y0, x1, y1 = bbox
    return (round(y0, 2), round(x0, 2))

def _render_clip(page: fitz.Page, bbox, dpi: int) -> Image.Image:
    rect = fitz.Rect(bbox)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _emit_list_or_para(lines: List[str]) -> str:
    # si beaucoup de puces num/bullets => liste
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
        return "\n".join(items) + "\n"
    # sinon paragraphe
    text = " ".join(s.strip() for s in lines if s.strip())
    return text + "\n" if text else ""

def extract_pdf_markdown_with_layout(pdf_bytes: bytes, meta_out: Dict[str, Any]) -> str:
    """
    Reconstitue le Markdown d’un PDF en respectant l’ordre des blocs (texte/images).
    - Titres via taille police
    - Listes via bullets
    - Code via police mono
    - OCR sur blocs image (region-based); embed base64 si OCR médiocre (selon EMBED_IMAGES)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        median_size = _median_font_size(doc)
        meta_out["median_font_size"] = median_size

        md_pages: List[str] = []
        max_pages = min(doc.page_count, OCR_MAX_PAGES)
        for pi in range(max_pages):
            page = doc.load_page(pi)
            raw = page.get_text("rawdict")
            blocks = raw.get("blocks", [])
            # ordonner
            blocks = sorted(blocks, key=lambda b: _bbox_key(b.get("bbox", (0,0,0,0))))
            page_lines: List[str] = [f"## Page {pi+1}"]

            for bi, b in enumerate(blocks, start=1):
                btype = b.get("type", 0)
                bbox = b.get("bbox", (0,0,0,0))

                if btype == 0:
                    # Bloc texte
                    lines_collect: List[str] = []
                    max_span_size = 0.0
                    any_mono = False
                    any_bold = False

                    for line in b.get("lines", []):
                        # concatène les spans d’une ligne
                        segs = []
                        for span in line.get("spans", []):
                            txt = span.get("text") or ""
                            if not txt:
                                continue
                            segs.append(txt)
                            size = float(span.get("size", 0) or 0)
                            if size > max_span_size:
                                max_span_size = size
                            if _is_mono(span): any_mono = True
                            if _is_bold(span): any_bold = True
                        if segs:
                            line_txt = "".join(segs).rstrip()
                            if line_txt:
                                lines_collect.append(line_txt)

                    # heading ?
                    lvl = _heading_level_for_block(max_span_size, median_size)
                    if lvl:
                        text = " ".join(s.strip() for s in lines_collect).strip()
                        if text:
                            page_lines.append("#"*lvl + " " + text)
                        continue

                    # code ?
                    if any_mono:
                        code_text = "\n".join(lines_collect).rstrip()
                        if code_text:
                            page_lines.append("```text\n" + code_text + "\n```")
                        continue

                    # liste / paragraphe
                    para = _emit_list_or_para(lines_collect)
                    if para.strip():
                        page_lines.append(para.rstrip())

                elif btype == 1:
                    # Bloc image -> OCR region + option embed
                    try:
                        im = _render_clip(page, bbox, OCR_DPI)
                        ocr_txt, ocr_score = ("", 0.0)
                        if OCR_ENABLED:
                            ocr_txt, ocr_score = ocr_image_best(im, OCR_LANGS)

                        used_text = False
                        if ocr_txt and (len(ocr_txt) >= 25 or ocr_score >= OCR_SCORE_GOOD_ENOUGH):
                            page_lines.append(ocr_txt)
                            used_text = True

                        if EMBED_IMAGES == "all" or (EMBED_IMAGES == "ocr_only" and not used_text):
                            im2 = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_base64(im2, IMG_FORMAT, IMG_JPEG_QUALITY)
                            page_lines.append(f"![{IMG_ALT_PREFIX} p{pi+1}-{bi}]({data_uri})")

                    except Exception:
                        # ne pas bloquer la page si un bloc image plante
                        continue

                # type 2 (dessin) ignoré

            md_pages.append("\n\n".join(s for s in page_lines if s and s.strip()))

        return "\n\n".join(md_pages).strip()

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
  <p class="muted">Upload un document (PDF, DOCX, PPTX, XLSX, HTML, etc.) → Markdown. Optionnel : résumé Azure OpenAI, OCR avec détection de tableaux, et intégration d’images en base64.</p>

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
      <span class="muted">Optionnel (meilleur OCR/layout pour PDF scannés)</span>
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
    llm_model: Optional[str] = Form(None),   # ignoré pour Azure; compat
    use_llm: bool = Form(False),
    force_ocr: bool = Form(False),
):
    """
    PDF : passe "layout" (PyMuPDF) qui recompose le doc (titres/listes/code) et place texte OCR / images base64
          exactement à l’endroit d’origine.
    Autres formats : MarkItDown + OCR d’appoint si vraiment peu de contenu.
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

        metadata: Dict[str, Any] = {
            "source_filename": file.filename,
            "ocr_langs": OCR_LANGS,
            "embed_images": EMBED_IMAGES,
        }

        is_pdf = guess_is_pdf(file.filename, file.content_type)
        is_img = guess_is_image(file.filename, file.content_type)

        markdown = ""
        # PDF : priorité à la passe "layout"
        if is_pdf:
            md_layout = extract_pdf_markdown_with_layout(content, metadata)
            if md_layout and (len(md_layout) >= max(300, OCR_MIN_CHARS//2)) and not force_ocr:
                markdown = md_layout
                metadata["mode"] = "pdf_layout"
            else:
                # fallback MarkItDown + OCR global (append)
                md = MarkItDown(enable_plugins=use_plugins, docintel_endpoint=docintel_endpoint)
                result = md.convert_stream(io.BytesIO(content), file_name=file.filename)
                base_md = getattr(result, "text_content", "") or ""
                metadata["mode"] = "fallback_markitdown"
                if OCR_ENABLED:
                    # OCR page-level (global), en append
                    try:
                        ocr_doc = fitz.open(stream=content, filetype="pdf")
                        pages = min(ocr_doc.page_count, OCR_MAX_PAGES)
                        ocr_pages_out = []
                        for i in range(pages):
                            im = _render_clip(ocr_doc.load_page(i), ocr_doc.load_page(i).rect, OCR_DPI)
                            txt, score = ocr_image_best(im, OCR_LANGS)
                            if txt.strip():
                                ocr_pages_out.append(f"## Page {i+1}\n\n{txt}")
                        ocr_doc.close()
                        if ocr_pages_out:
                            markdown = (base_md.strip() + "\n\n# OCR (extrait)\n" + "\n\n".join(ocr_pages_out)).strip()
                        else:
                            markdown = base_md.strip()
                    except Exception:
                        markdown = base_md.strip()
                else:
                    markdown = base_md.strip()

        else:
            # Non-PDF : MarkItDown d'abord
            md = MarkItDown(enable_plugins=use_plugins, docintel_endpoint=docintel_endpoint)
            result = md.convert_stream(io.BytesIO(content), file_name=file.filename)
            base_md = getattr(result, "text_content", "") or ""
            metadata["mode"] = "markitdown"

            if OCR_ENABLED and (force_ocr or (len(base_md.strip()) < OCR_MIN_CHARS and (is_img or file.filename.lower().endswith((".tif", ".tiff"))))):
                # Image unique : OCR + embed optionnel
                if is_img:
                    try:
                        with Image.open(io.BytesIO(content)) as im:
                            txt, score = ocr_image_best(im, OCR_LANGS)
                            if txt.strip():
                                base_md = (base_md + "\n\n# OCR (extrait)\n" + txt).strip() if base_md.strip() else txt
                            if EMBED_IMAGES in ("all", "ocr_only") and (EMBED_IMAGES == "all" or not txt.strip()):
                                im2 = _pil_resize_max(im, IMG_MAX_WIDTH)
                                data_uri = _pil_to_base64(im2, IMG_FORMAT, IMG_JPEG_QUALITY)
                                base_md += f'\n\n![{IMG_ALT_PREFIX}]({data_uri})\n'
                    except Exception:
                        pass

            markdown = base_md.strip()

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
