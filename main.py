import os
import io
import re
import time
import base64
import json
import traceback
from typing import Optional, Tuple, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Librairie Microsoft MarkItDown
from markitdown import MarkItDown

# Traitement d'images et PDF
from PIL import Image
import fitz  # PyMuPDF

# Traitement DOCX
import mammoth

# Traitement PPTX
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

# Traitement HTML
from html.parser import HTMLParser
import html

# ---------------------------
# Config via variables d'env
# ---------------------------
SAVE_UPLOADS  = os.getenv("SAVE_UPLOADS", "false").lower() == "true"
SAVE_OUTPUTS  = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"
UPLOAD_DIR    = os.getenv("UPLOAD_DIR", "/data/uploads")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "/data/outputs")

# Intégration images base64
IMG_FORMAT         = os.getenv("IMG_FORMAT", "png").strip().lower()
IMG_JPEG_QUALITY   = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH      = int(os.getenv("IMG_MAX_WIDTH", "1400"))
IMG_ALT_PREFIX     = os.getenv("IMG_ALT_PREFIX", "Capture").strip()

# Dossiers persistents
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="5.2-json-pptx")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers génériques
# ---------------------------
def guess_file_type(filename: str, content_type: Optional[str]) -> str:
    fname = filename.lower()
    ctype = (content_type or "").lower()
    
    if ctype == "application/pdf" or fname.endswith(".pdf"):
        return "pdf"
    if "html" in ctype or fname.endswith((".html", ".htm")):
        return "html"
    if fname.endswith(".docx"):
        return "docx"
    if fname.endswith((".pptx", ".ppt")):
        return "pptx"
    if "json" in ctype or fname.endswith(".json"):
        return "json"
    if fname.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
        return "image"
    return "unknown"

def _md_cleanup(md: str) -> str:
    """Nettoyage cosmétique du Markdown final."""
    if not md: return md
    lines = []
    for L in md.replace("\r", "").split("\n"):
        l = re.sub(r"[ \t]+$", "", L)
        l = re.sub(r"^\s*[•·●◦▪]\s+", "- ", l) # Normalisation puces
        l = re.sub(r"^\s*(\d+)[\)\]]\s+", r"\1. ", l) # Normalisation listes numérotées
        lines.append(l)
    return "\n".join(lines).strip()

def _remove_headers_footers(md_lines: List[str]) -> List[str]:
    if len(md_lines) < 2: return md_lines
    first_lines, last_lines = [], []
    for c in md_lines:
        lines = [l for l in c.splitlines() if l.strip()]
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])
    
    # Détection répétitions (>50%)
    h_counts = {l: first_lines.count(l) for l in first_lines}
    f_counts = {l: last_lines.count(l) for l in last_lines}
    h_rem = {l for l, c in h_counts.items() if c >= 2 and c >= 0.5 * len(md_lines)}
    f_rem = {l for l, c in f_counts.items() if c >= 2 and c >= 0.5 * len(md_lines)}

    cleaned = []
    for c in md_lines:
        lines = c.splitlines()
        if lines and lines[0].strip() in h_rem: lines = lines[1:]
        if lines and lines[-1].strip() in f_rem: lines = lines[:-1]
        cleaned.append("\n".join(lines).strip())
    return cleaned

# --------------------------------------------------------------------
# Gestion Images Base64
# --------------------------------------------------------------------
def _pil_resize_max(im: Image.Image, max_w: int) -> Image.Image:
    if max_w and im.width > max_w:
        ratio = max_w / im.width
        new_h = int(im.height * ratio)
        return im.resize((max_w, new_h), Image.LANCZOS)
    return im

def _pil_to_base64(im: Image.Image, fmt: str = "png", quality: int = 85) -> str:
    buf = io.BytesIO()
    fmt = fmt.lower()
    if fmt in ("jpg", "jpeg"):
        im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _crop_bbox_image(page: fitz.Page, bbox: Tuple[float, float, float, float], dpi: int = 200) -> Optional[Image.Image]:
    try:
        rect = fitz.Rect(*bbox)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), clip=rect, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except: return None

# --------------------------------------------------------------------
# HTML / DOCX Helpers
# --------------------------------------------------------------------
_HTML_DATA_URI_IN_IMG = re.compile(r'<img\b[^>]*\bsrc=["\'](data:image/[^"\']+)["\']', flags=re.IGNORECASE)
_MD_DATA_IMG = re.compile(r'!\[[^\]]*\]\(\s*(?P<src>data:image/[^)]+)\s*\)')
_HTML_IMG_DATA_TAG = re.compile(r'<img\b[^>]*\bsrc=["\'](?P<src>data:image/[^"\']+)["\'][^>]*>', flags=re.IGNORECASE)

def _extract_html_data_imgs(html_text: str) -> List[str]:
    return _HTML_DATA_URI_IN_IMG.findall(html_text) if html_text else []

def _html_img_datauri_to_markdown(md: str, alt: str = "Capture") -> str:
    if not md or "<img" not in md.lower(): return md
    return _HTML_IMG_DATA_TAG.sub(lambda m: f'![{alt}]({m.group("src")})', md)

def _inject_full_data_uris_into_markdown(md: str, uris: List[str], prefix: str = "Capture") -> str:
    if not md or not uris: return md
    idx = 0
    def repl(m): nonlocal idx; src = uris[idx] if idx < len(uris) else m.group("src"); idx+=1; return f'![{prefix} – {idx}]({src})'
    return _MD_DATA_IMG.sub(repl, md)

# ---------------------------
# Renderers Spécifiques
# ---------------------------

def render_json_markdown(content: bytes) -> str:
    """Formatte le JSON proprement."""
    try:
        data = json.loads(content)
        # Pretty print avec indentation
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        return f"```json\n{formatted}\n```"
    except json.JSONDecodeError:
        return "```text\n(Fichier JSON invalide)\n```"

def render_pptx_markdown(content: bytes) -> str:
    """Convertit PPTX -> Markdown (Texte + Images Base64)."""
    try:
        prs = Presentation(io.BytesIO(content))
        md_slides = []
        
        for i, slide in enumerate(prs.slides):
            slide_content = []
            
            # Titre de la slide
            if slide.shapes.title:
                slide_content.append(f"## {slide.shapes.title.text}")
            else:
                slide_content.append(f"## Slide {i+1}")

            # Tri des formes par position (haut -> bas, gauche -> droite)
            # PPTX ne stocke pas forcement dans l'ordre visuel
            shapes = list(slide.shapes)
            shapes.sort(key=lambda s: (s.top, s.left))

            for shape in shapes:
                # 1. Texte
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        txt = paragraph.text.strip()
                        if txt:
                            # Tentative de detection de liste à puce basique
                            prefix = "- " if paragraph.level > 0 else ""
                            slide_content.append(f"{prefix}{txt}")

                # 2. Images
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        image_blob = shape.image.blob
                        with Image.open(io.BytesIO(image_blob)) as im:
                            im = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            slide_content.append(f"![{IMG_ALT_PREFIX} – Slide {i+1}]({data_uri})")
                    except Exception:
                        pass # Ignore images corrompues

            md_slides.append("\n\n".join(slide_content))
            
        return "\n\n---\n\n".join(md_slides)
    except Exception as e:
        return f"Erreur PPTX: {str(e)}"

def render_docx_markdown(content: bytes) -> Tuple[str, Dict]:
    try:
        res = mammoth.convert_to_html(io.BytesIO(content))
        uris = _extract_html_data_imgs(res.value)
        
        md_eng = MarkItDown()
        out = md_eng.convert_stream(io.BytesIO(res.value.encode('utf-8')), file_extension=".html")
        md = getattr(out, "text_content", "") or ""
        
        md = _html_img_datauri_to_markdown(md)
        md = _inject_full_data_uris_into_markdown(md, uris, IMG_ALT_PREFIX)
        return _md_cleanup(md), {"messages": [m.message for m in res.messages]}
    except Exception as e: return f"Erreur DOCX: {e}", {"error": str(e)}

def render_pdf_markdown_inline(content: bytes, meta: Dict) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    meta["pages"] = doc.page_count
    md_pages = []
    
    try:
        for p in range(doc.page_count):
            page = doc.load_page(p)
            raw = page.get_text("dict") or {}
            median = _median_font_size(raw)
            items = []
            
            for b in raw.get("blocks", []):
                bbox = b.get("bbox")
                if b["type"] == 0: # Texte
                    block_txt = []
                    max_s, is_bold = 0.0, False
                    for l in b["lines"]:
                        line_parts = []
                        for s in l["spans"]:
                            t = s["text"]
                            if not t.strip(): continue
                            max_s = max(max_s, s["size"]); is_bold = is_bold or _is_bold(s["flags"])
                            line_parts.append(f"**{t}**" if _is_bold(s["flags"]) else t)
                        if line_parts: block_txt.append("".join(line_parts))
                    
                    if block_txt:
                        full = " ".join(block_txt)
                        h = _classify_heading(max_s, median, is_bold)
                        items.append((bbox[1], f"{h} {full}" if h else full))
                        
                elif b["type"] == 1: # Image
                    im = _crop_bbox_image(page, bbox, 300)
                    if im:
                        im = _pil_resize_max(im, IMG_MAX_WIDTH)
                        uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                        items.append((bbox[1], f"\n![{IMG_ALT_PREFIX} – page {p+1}]({uri})\n"))

            items.sort(key=lambda x: x[0])
            md_pages.append("\n\n".join([x[1] for x in items]))

        cleaned = _remove_headers_footers(md_pages)
        return _md_cleanup("\n\n---\n\n".join(cleaned))
    finally: doc.close()

# PDF Utils
def _is_bold(flags): return bool(flags & 1 or flags & 32)
def _median_font_size(raw):
    sizes = [s["size"] for b in raw.get("blocks",[]) if b["type"]==0 for l in b["lines"] for s in l["spans"] if s["text"].strip()]
    if not sizes: return 0.0
    sizes.sort(); mid = len(sizes)//2
    return sizes[mid] if len(sizes)%2 else (sizes[mid-1]+sizes[mid])/2.0
def _classify_heading(s, med, b):
    if med <= 0: return None
    if s >= med*1.8: return "#"
    if s >= med*1.5: return "##"
    if b and s >= med*1.1: return "###"
    return None

# ---------------------------
# UI
# ---------------------------
HTML_PAGE = r'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>MarkItDown</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root{ color-scheme: dark; --bg: #111827; --card: #1f2937; --text: #f3f4f6; --accent: #3b82f6; --accent-hover: #2563eb; --border: #374151; --muted: #9ca3af; }
    body{ font-family: -apple-system, system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.5; }
    .container { max-width: 1400px; margin: 0 auto; }
    h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .desc { color: var(--muted); margin-bottom: 2rem; text-align: center; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    #dropzone { border: 2px dashed var(--border); border-radius: 12px; padding: 2rem; text-align: center; color: var(--muted); cursor: pointer; transition: border 0.2s; }
    #dropzone.dragover { border-color: var(--accent); background: rgba(59, 130, 246, 0.1); }
    .controls { display: flex; gap: 1rem; justify-content: center; margin-top: 1.5rem; flex-wrap: wrap; }
    button { background: var(--accent); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
    button:hover { background: var(--accent-hover); transform: translateY(-1px); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.ghost { background: transparent; border: 1px solid var(--border); color: var(--text); }
    button.ghost:hover { background: var(--border); }
    .stats { display: flex; gap: 1rem; font-size: 0.85rem; color: var(--muted); justify-content: center; margin-top: 1rem; }
    .pill { background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 99px; }
    .editor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; height: 70vh; }
    .column h3 { margin: 0 0 0.5rem 0; font-size: 1rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
    textarea, #preview { flex: 1; width: 100%; background: #0f1319; border: 1px solid var(--border); border-radius: 8px; padding: 1rem; font-family: 'Menlo', monospace; font-size: 0.9rem; resize: none; overflow-y: auto; height: 100%; box-sizing: border-box; }
    textarea { color: #e5e7eb; outline: none; }
    textarea:focus { border-color: var(--accent); }
    #preview { background: #1f2937; color: #d1d5db; font-family: -apple-system, system-ui, sans-serif; line-height: 1.6; }
    #preview img { max-width: 100%; height: auto; border-radius: 4px; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    #preview h1, #preview h2 { color: #f3f4f6; margin-top: 1.5em; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
    #preview pre { background: #111827; padding: 1em; border-radius: 6px; overflow-x: auto; }
    #preview code { font-family: monospace; }
    #preview table { width: 100%; border-collapse: collapse; margin: 1em 0; }
    #preview th, #preview td { border: 1px solid var(--border); padding: 8px 12px; }
    @media (max-width: 768px) { .editor-grid { grid-template-columns: 1fr; height: auto; } textarea, #preview { height: 500px; } }
  </style>
</head>
<body>
  <div class="container">
    <h1>MarkItDown</h1>
    <div class="desc">PDF · DOCX · PPTX · JSON · HTML · Image</div>

    <div class="card">
      <div id="dropzone">Glissez votre fichier ici ou cliquez pour parcourir</div>
      <input type="file" id="file" style="display:none" />
      <div class="controls">
        <button id="convert">Convertir</button>
        <button id="download" class="ghost" style="display:none">Télécharger .md</button>
        <button id="copy" class="ghost">Copier</button>
      </div>
      <div class="stats">
        <span class="pill" id="status">En attente...</span>
        <span class="pill">Temps: <span id="timer">0s</span></span>
      </div>
    </div>

    <div class="editor-grid">
      <div class="column"><h3>Markdown</h3><textarea id="output" spellcheck="false"></textarea></div>
      <div class="column"><h3>Aperçu</h3><div id="preview"></div></div>
    </div>
  </div>

<script>
const $ = id => document.getElementById(id);
const dz = $("dropzone"), inp = $("file"), out = $("output"), prev = $("preview");
marked.use({ breaks: true, gfm: true });

function updatePreview() { prev.innerHTML = out.value.trim() ? marked.parse(out.value) : '<div style="color:#666;text-align:center;margin-top:2rem">Aperçu ici</div>'; }
out.addEventListener("input", updatePreview);
dz.onclick = () => inp.click();
dz.ondragover = e => { e.preventDefault(); dz.classList.add("dragover"); };
dz.ondragleave = () => dz.classList.remove("dragover");
dz.ondrop = e => { e.preventDefault(); dz.classList.remove("dragover"); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); };
inp.onchange = () => { if(inp.files[0]) handleFile(inp.files[0]); };

function handleFile(f) { inp.files = (new DataTransfer().items.add(f) && new DataTransfer().files) || inp.files; $("status").innerText = f.name; $("convert").disabled = false; }

$("convert").onclick = async () => {
    const f = inp.files[0];
    if(!f) return alert("Fichier manquant");
    const btn = $("convert"); btn.disabled = true; btn.innerText = "..."; $("status").innerText = "Conversion...";
    const t0 = performance.now();
    try {
        const fd = new FormData(); fd.append("file", f);
        const res = await fetch("/convert", { method: "POST", body: fd });
        const json = await res.json();
        $("timer").innerText = ((performance.now()-t0)/1000).toFixed(2) + "s";
        if(res.ok) {
            out.value = json.markdown; updatePreview(); $("status").innerText = "OK";
            const url = URL.createObjectURL(new Blob([json.markdown], {type: "text/markdown"}));
            const dl = $("download"); dl.style.display = "inline-block";
            dl.onclick = () => { const a = document.createElement("a"); a.href = url; a.download = json.output_filename; a.click(); };
        } else { out.value = "Erreur: " + JSON.stringify(json); $("status").innerText = "Erreur"; }
    } catch(e) { out.value = e.toString(); $("status").innerText = "Erreur réseau"; }
    finally { btn.disabled = false; btn.innerText = "Convertir"; }
};
$("copy").onclick = () => { navigator.clipboard.writeText(out.value); const old = $("copy").innerText; $("copy").innerText = "Copié !"; setTimeout(() => $("copy").innerText = old, 1000); };
updatePreview();
</script></body></html>
'''

@app.get("/", response_class=HTMLResponse)
def index(): return HTMLResponse(HTML_PAGE)

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    try:
        t0 = time.perf_counter()
        content = await file.read()
        if SAVE_UPLOADS:
            with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as f: f.write(content)

        ftype = guess_file_type(file.filename, file.content_type)
        meta = {"type": ftype}
        
        if ftype == "pdf":
            md = render_pdf_markdown_inline(content, meta)
        elif ftype == "docx":
            md, m_docx = render_docx_markdown(content)
            meta.update(m_docx)
        elif ftype == "pptx":
            md = render_pptx_markdown(content)
        elif ftype == "json":
            md = render_json_markdown(content)
        elif ftype == "html":
            txt = content.decode("utf-8", errors="ignore")
            uris = _extract_html_data_imgs(txt)
            eng = MarkItDown()
            res = eng.convert_stream(io.BytesIO(content), file_extension=".html")
            md = getattr(res, "text_content", "") or ""
            md = _html_img_datauri_to_markdown(md)
            md = _inject_full_data_uris_into_markdown(md, uris, IMG_ALT_PREFIX)
            md = _md_cleanup(md)
        elif ftype == "image":
            with Image.open(io.BytesIO(content)) as im:
                im = _pil_resize_max(im, IMG_MAX_WIDTH)
                uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                md = f"![{file.filename}]({uri})"
        else:
            eng = MarkItDown()
            try:
                res = eng.convert_stream(io.BytesIO(content), file_extension=os.path.splitext(file.filename)[1])
                md = getattr(res, "text_content", "") or ""
            except:
                md = content.decode("utf-8", errors="replace")

        out_name = f"{os.path.splitext(file.filename)[0]}.md"
        if SAVE_OUTPUTS:
            with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f: f.write(md)

        meta["duration"] = round(time.perf_counter() - t0, 3)
        return JSONResponse({"filename": file.filename, "output_filename": out_name, "markdown": md, "metadata": meta})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
