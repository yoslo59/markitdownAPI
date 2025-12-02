import os
import io
import re
import time
import base64
import hashlib
import traceback
from typing import Optional, Tuple, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Librairie Microsoft MarkItDown (ou compatible)
from markitdown import MarkItDown

# Traitement d'images et PDF
from PIL import Image
import fitz  # PyMuPDF

# Traitement DOCX
import mammoth

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
app = FastAPI(title="MarkItDown API", version="5.1-ui-preview")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers génériques & markdown
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
    if fname.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
        return "image"
    return "unknown"

def _md_cleanup(md: str) -> str:
    """Nettoyage cosmétique du Markdown final."""
    if not md:
        return md
    lines = []
    for L in md.replace("\r", "").split("\n"):
        l = re.sub(r"[ \t]+$", "", L)
        l = re.sub(r"^\s*[•·●◦▪]\s+", "- ", l)
        l = re.sub(r"^\s*(\d+)[\)\]]\s+", r"\1. ", l)
        lines.append(l)
    txt = "\n".join(lines)
    return txt.strip()

def _remove_headers_footers(md_lines: List[str]) -> List[str]:
    """
    Supprime les en-têtes et pieds de page répétés sur plusieurs pages.
    """
    if len(md_lines) < 2:
        return md_lines

    first_lines: List[str] = []
    last_lines: List[str] = []
    
    for content in md_lines:
        lines = [l for l in content.splitlines() if l.strip()]
        if not lines:
            continue
        first_lines.append(lines[0])
        last_lines.append(lines[-1])

    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}

    for line in first_lines:
        header_counts[line] = header_counts.get(line, 0) + 1
    for line in last_lines:
        footer_counts[line] = footer_counts.get(line, 0) + 1

    headers_to_remove = {line for line, count in header_counts.items() if count >= 2 and count >= 0.5 * len(md_lines)}
    footers_to_remove = {line for line, count in footer_counts.items() if count >= 2 and count >= 0.5 * len(md_lines)}

    cleaned_pages = []
    for content in md_lines:
        lines = content.splitlines()
        if lines and lines[0].strip() and lines[0] in headers_to_remove:
            lines = lines[1:]
        if lines and lines[-1].strip() and lines[-1] in footers_to_remove:
            lines = lines[:-1]
        cleaned_pages.append("\n".join(lines).strip())
    
    return cleaned_pages

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
        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0, y0, x1, y1)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None

# --------------------------------------------------------------------
# HTML Helpers (Réinjection Base64)
# --------------------------------------------------------------------
_HTML_DATA_URI_IN_IMG = re.compile(r'<img\b[^>]*\bsrc=["\'](data:image/[^"\']+)["\']', flags=re.IGNORECASE)
_MD_DATA_IMG = re.compile(r'!\[[^\]]*\]\(\s*(?P<src>data:image/[^)]+)\s*\)')
_HTML_IMG_DATA_TAG = re.compile(r'<img\b[^>]*\bsrc=["\'](?P<src>data:image/[^"\']+)["\'][^>]*>', flags=re.IGNORECASE)

def _extract_html_data_imgs(html_text: str) -> List[str]:
    if not html_text: return []
    return _HTML_DATA_URI_IN_IMG.findall(html_text)

def _html_img_datauri_to_markdown(md_or_html: str, alt_text: str = "Capture") -> str:
    if not md_or_html or "<img" not in md_or_html.lower():
        return md_or_html
    return _HTML_IMG_DATA_TAG.sub(lambda m: f'![{alt_text}]({m.group("src")})', md_or_html)

def _inject_full_data_uris_into_markdown(md: str, data_uris: List[str], alt_prefix: str = "Capture") -> str:
    if not md or not data_uris: return md
    idx = 0
    def repl(m: re.Match) -> str:
        nonlocal idx
        src_full = data_uris[idx] if idx < len(data_uris) else m.group("src")
        idx += 1
        return f'![{alt_prefix} – {idx}]({src_full})'
    return _MD_DATA_IMG.sub(repl, md)

# ---------------------------
# Traitement DOCX (via Mammoth)
# ---------------------------
def render_docx_markdown(docx_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    try:
        result = mammoth.convert_to_html(io.BytesIO(docx_bytes))
        html_content = result.value
        messages = result.messages
        
        data_uris = _extract_html_data_imgs(html_content)
        
        md_engine = MarkItDown()
        res_md = md_engine.convert_stream(io.BytesIO(html_content.encode('utf-8')), file_extension=".html")
        markdown = getattr(res_md, "text_content", "") or ""

        markdown = _html_img_datauri_to_markdown(markdown)
        markdown = _inject_full_data_uris_into_markdown(markdown, data_uris, IMG_ALT_PREFIX)
        markdown = _md_cleanup(markdown)

        meta = {"engine": "mammoth+markitdown", "messages": [m.message for m in messages]}
        return markdown, meta
    except Exception as e:
        return f"Erreur DOCX: {e}", {"error": str(e)}

# ---------------------------
# Traitement PDF (Structure + Images, sans OCR)
# ---------------------------
def _is_bold(flags: int) -> bool:
    return bool(flags & 1 or flags & 32)

def _median_font_size(page_raw: Dict[str, Any]) -> float:
    sizes = []
    for b in page_raw.get("blocks", []):
        if b.get("type", 0) != 0: continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if s.get("text", "").strip(): sizes.append(float(s.get("size", 0)))
    if not sizes: return 0.0
    sizes.sort()
    mid = len(sizes) // 2
    return sizes[mid] if len(sizes) % 2 == 1 else (sizes[mid-1] + sizes[mid]) / 2.0

def _classify_heading(size: float, median_size: float, has_bold: bool) -> Optional[str]:
    if median_size <= 0: return None
    if size >= median_size * 1.8: return "#"
    if size >= median_size * 1.5: return "##"
    if has_bold and size >= median_size * 1.1: return "###"
    return None

def render_pdf_markdown_inline(pdf_bytes: bytes, meta_out: Dict[str, Any]) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_pages: List[str] = []
    
    meta_out["pages"] = doc.page_count
    meta_out["engine"] = "pymupdf_structure"

    try:
        for p in range(doc.page_count):
            page = doc.load_page(p)
            raw = page.get_text("dict") or {}
            median_size = _median_font_size(raw)
            
            page_content: List[Tuple[float, str]] = [] # (y_pos, md_text)
            
            blocks = raw.get("blocks", [])
            for b in blocks:
                btype = b.get("type", 0)
                bbox = b.get("bbox")
                
                if btype == 0: # TEXTE
                    block_text_parts = []
                    block_max_size = 0.0
                    block_bold = False
                    
                    for line in b.get("lines", []):
                        line_parts = []
                        for sp in line.get("spans", []):
                            txt = sp.get("text", "")
                            if not txt.strip(): continue
                            size = sp.get("size", 0)
                            bold = _is_bold(sp.get("flags", 0))
                            block_max_size = max(block_max_size, size)
                            block_bold = block_bold or bold
                            line_parts.append(f"**{txt}**" if bold else txt)
                        if line_parts:
                            block_text_parts.append("".join(line_parts))
                            
                    if block_text_parts:
                        full_block = " ".join(block_text_parts)
                        h = _classify_heading(block_max_size, median_size, block_bold)
                        if h:
                            full_block = f"{h} {full_block}"
                        page_content.append((bbox[1], full_block))

                elif btype == 1: # IMAGE
                    im = _crop_bbox_image(page, bbox, dpi=300)
                    if im:
                        im = _pil_resize_max(im, IMG_MAX_WIDTH)
                        data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                        md_img = f'\n![{IMG_ALT_PREFIX} – page {p+1}]({data_uri})\n'
                        page_content.append((bbox[1], md_img))

            page_content.sort(key=lambda x: x[0])
            page_md = "\n\n".join([item[1] for item in page_content])
            md_pages.append(page_md)

        cleaned_pages = _remove_headers_footers(md_pages)
        final_md = "\n\n---\n\n".join(cleaned_pages)
        return _md_cleanup(final_md)

    except Exception as e:
        traceback.print_exc()
        return f"Error processing PDF: {e}"
    finally:
        doc.close()

# ---------------------------
# UI (Mise à jour avec Split View)
# ---------------------------
HTML_PAGE = r'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>MarkItDown</title>
  <!-- Librairie légère pour le rendu Markdown en direct -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root{
      color-scheme: dark;
      --bg: #111827; --card: #1f2937; --text: #f3f4f6; --accent: #3b82f6; --accent-hover: #2563eb;
      --border: #374151; --muted: #9ca3af;
    }
    body{ font-family: -apple-system, system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.5; }
    .container { max-width: 1400px; margin: 0 auto; }
    
    h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .desc { color: var(--muted); margin-bottom: 2rem; text-align: center; }
    
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    
    /* Zone de glisser-déposer */
    #dropzone { border: 2px dashed var(--border); border-radius: 12px; padding: 2rem; text-align: center; color: var(--muted); cursor: pointer; transition: border 0.2s; }
    #dropzone.dragover { border-color: var(--accent); background: rgba(59, 130, 246, 0.1); }
    
    /* Contrôles */
    .controls { display: flex; gap: 1rem; justify-content: center; margin-top: 1.5rem; flex-wrap: wrap; }
    button { background: var(--accent); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
    button:hover { background: var(--accent-hover); transform: translateY(-1px); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.ghost { background: transparent; border: 1px solid var(--border); color: var(--text); }
    button.ghost:hover { background: var(--border); }
    
    .stats { display: flex; gap: 1rem; font-size: 0.85rem; color: var(--muted); justify-content: center; margin-top: 1rem; }
    .pill { background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 99px; }

    /* Grille d'édition */
    .editor-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        height: 70vh;
    }
    
    .column { display: flex; flex-direction: column; gap: 0.5rem; }
    .column h3 { margin: 0; font-size: 1rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }

    /* Zones de texte et prévisualisation */
    textarea, #preview {
        flex: 1;
        width: 100%;
        background: #0f1319;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Menlo', monospace;
        font-size: 0.9rem;
        resize: none;
        overflow-y: auto;
    }
    
    textarea { color: #e5e7eb; outline: none; transition: border 0.2s; }
    textarea:focus { border-color: var(--accent); }

    /* Styles pour le rendu Markdown */
    #preview { background: #1f2937; color: #d1d5db; font-family: -apple-system, system-ui, sans-serif; line-height: 1.6; }
    #preview img { max-width: 100%; height: auto; border-radius: 4px; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    #preview h1, #preview h2, #preview h3 { color: #f3f4f6; margin-top: 1.5em; margin-bottom: 0.5em; line-height: 1.2; }
    #preview h1 { border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
    #preview code { background: rgba(255,255,255,0.1); padding: 0.2em 0.4em; border-radius: 4px; font-family: monospace; font-size: 0.85em; }
    #preview pre { background: #111827; padding: 1em; border-radius: 6px; overflow-x: auto; margin: 1em 0; }
    #preview pre code { background: transparent; padding: 0; }
    #preview blockquote { border-left: 4px solid var(--accent); margin: 1em 0; padding-left: 1em; color: var(--muted); }
    #preview table { width: 100%; border-collapse: collapse; margin: 1em 0; }
    #preview th, #preview td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
    #preview th { background: rgba(255,255,255,0.05); }
    #preview hr { border: 0; border-top: 1px solid var(--border); margin: 2rem 0; }

    @media (max-width: 768px) { .editor-grid { grid-template-columns: 1fr; height: auto; } textarea, #preview { height: 500px; } }
  </style>
</head>
<body>
  <div class="container">
    <h1>MarkItDown</h1>
    <div class="desc">Convertisseur PDF / DOCX / HTML / Image vers Markdown</div>

    <div class="card">
      <div id="dropzone">Glissez votre fichier ici ou cliquez pour parcourir</div>
      <input type="file" id="file" style="display:none" />
      
      <div class="controls">
        <button id="convert">Convertir le document</button>
        <button id="download" class="ghost" style="display:none">Télécharger .md</button>
        <button id="copy" class="ghost">Copier le code</button>
      </div>
      
      <div class="stats">
        <span class="pill" id="status">En attente de fichier...</span>
        <span class="pill">Temps: <span id="timer">0s</span></span>
      </div>
    </div>

    <div class="editor-grid">
      <div class="column">
        <h3>Code Markdown</h3>
        <textarea id="output" spellcheck="false" placeholder="Le code Markdown apparaîtra ici..."></textarea>
      </div>
      <div class="column">
        <h3>Rendu Visuel</h3>
        <div id="preview">
            <div style="color: var(--muted); text-align: center; margin-top: 2rem;">
                La prévisualisation apparaîtra ici.
            </div>
        </div>
      </div>
    </div>
  </div>

<script>
const $ = id => document.getElementById(id);
const dz = $("dropzone"), inp = $("file"), out = $("output"), prev = $("preview");

// Setup Marked (Optionnel: configuration)
marked.use({ breaks: true, gfm: true });

// Mise à jour de la prévisualisation
function updatePreview() {
    const md = out.value;
    if(!md.trim()) {
        prev.innerHTML = '<div style="color: var(--muted); text-align: center; margin-top: 2rem;">La prévisualisation apparaîtra ici.</div>';
    } else {
        prev.innerHTML = marked.parse(md);
    }
}

// Écouteur pour la frappe en direct
out.addEventListener("input", updatePreview);

// Drag & Drop
dz.onclick = () => inp.click();
dz.ondragover = e => { e.preventDefault(); dz.classList.add("dragover"); };
dz.ondragleave = () => dz.classList.remove("dragover");
dz.ondrop = e => { e.preventDefault(); dz.classList.remove("dragover"); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); };
inp.onchange = () => { if(inp.files[0]) handleFile(inp.files[0]); };

function handleFile(f) {
    inp.files = createFileList(f);
    $("status").innerText = "Fichier sélectionné : " + f.name;
    $("convert").disabled = false;
}

function createFileList(file) {
    const dt = new DataTransfer(); dt.items.add(file); return dt.files;
}

$("convert").onclick = async () => {
    const f = inp.files[0];
    if(!f) return alert("Veuillez sélectionner un fichier");
    
    const btn = $("convert");
    btn.disabled = true;
    btn.innerText = "Traitement...";
    $("status").innerText = "Conversion en cours...";
    
    const t0 = performance.now();
    const fd = new FormData();
    fd.append("file", f);

    try {
        const res = await fetch("/convert", { method: "POST", body: fd });
        const json = await res.json();
        const t1 = performance.now();
        $("timer").innerText = ((t1-t0)/1000).toFixed(2) + "s";
        
        if(res.ok) {
            out.value = json.markdown;
            updatePreview(); // Déclenche le rendu visuel
            
            $("status").innerText = "Terminé !";
            const blob = new Blob([json.markdown], {type: "text/markdown"});
            const url = URL.createObjectURL(blob);
            const dl = $("download");
            dl.style.display = "inline-block";
            dl.onclick = () => {
                const a = document.createElement("a");
                a.href = url;
                a.download = json.output_filename || "document.md";
                a.click();
            };
        } else {
            out.value = "Erreur: " + JSON.stringify(json, null, 2);
            $("status").innerText = "Erreur serveur";
        }
    } catch(e) {
        $("status").innerText = "Erreur réseau";
        out.value = e.toString();
    } finally {
        btn.disabled = false;
        btn.innerText = "Convertir le document";
    }
};

$("copy").onclick = () => {
    navigator.clipboard.writeText(out.value);
    const old = $("copy").innerText;
    $("copy").innerText = "Copié !";
    setTimeout(() => $("copy").innerText = old, 2000);
};
</script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_PAGE)

# ---------------------------
# Endpoint API
# ---------------------------
@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    try:
        t_start = time.perf_counter()
        content = await file.read()
        
        if SAVE_UPLOADS:
            with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as f: f.write(content)

        ftype = guess_file_type(file.filename, file.content_type)
        metadata = {"type": ftype}
        markdown = ""

        if ftype == "pdf":
            markdown = render_pdf_markdown_inline(content, metadata)

        elif ftype == "docx":
            md_docx, meta_docx = render_docx_markdown(content)
            markdown = md_docx
            metadata.update(meta_docx)

        elif ftype == "html":
            html_text = content.decode("utf-8", errors="ignore")
            data_uris = _extract_html_data_imgs(html_text)
            md_engine = MarkItDown()
            res = md_engine.convert_stream(io.BytesIO(content), file_extension=".html")
            markdown = getattr(res, "text_content", "") or ""
            markdown = _html_img_datauri_to_markdown(markdown)
            markdown = _inject_full_data_uris_into_markdown(markdown, data_uris, IMG_ALT_PREFIX)
            markdown = _md_cleanup(markdown)

        elif ftype == "image":
            try:
                with Image.open(io.BytesIO(content)) as im:
                    im = _pil_resize_max(im, IMG_MAX_WIDTH)
                    data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                    markdown = f"![{file.filename}]({data_uri})"
            except Exception as e:
                markdown = f"Erreur image: {e}"

        else:
            md_engine = MarkItDown()
            try:
                res = md_engine.convert_stream(io.BytesIO(content), file_extension=os.path.splitext(file.filename)[1])
                markdown = getattr(res, "text_content", "") or ""
            except:
                try:
                    markdown = content.decode("utf-8")
                except:
                    markdown = "Format non supporté et non lisible."

        out_name = f"{os.path.splitext(file.filename)[0]}.md"
        if SAVE_OUTPUTS:
            with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
                f.write(markdown)

        metadata["duration"] = round(time.perf_counter() - t_start, 3)

        return JSONResponse({
            "filename": file.filename,
            "output_filename": out_name,
            "markdown": markdown,
            "metadata": metadata,
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
