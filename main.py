import os
import io
import re
import time
import base64
import tempfile
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ---- Libs pour les formats ----
import fitz  # PyMuPDF (PDF)
import pandas as pd  # CSV/XLSX
from zipfile import ZipFile  # images DOCX
try:
    import docx  # python-docx
except ImportError:
    docx = None

# MarkItDown est optionnel: si présent on s'en sert pour certains formats
try:
    from markitdown import MarkItDown
except Exception:
    MarkItDown = None

# ---- PaddleOCR ----
from paddleocr import PaddleOCR, PPStructureV3

# =========================
# Config via variables d'env
# =========================
SAVE_UPLOADS = os.getenv("SAVE_UPLOADS", "false").lower() == "true"
SAVE_OUTPUTS = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/data/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/outputs")

# OCR
DEFAULT_OCR_LANGS = os.getenv("OCR_LANGS", "fra+eng").strip()
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "100"))
OCR_DPI = int(os.getenv("OCR_DPI", "300"))  # rendu raster pour OCR PDF

# Images intégrées
EMBED_IMAGES = os.getenv("EMBED_IMAGES", "ocr_only").lower()  # "ocr_only" | "all"
IMG_FORMAT = os.getenv("IMG_FORMAT", "png").lower()           # "png" | "jpeg"
IMG_JPEG_QUALITY = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH = int(os.getenv("IMG_MAX_WIDTH", "1600"))
IMG_ALT_PREFIX = os.getenv("IMG_ALT_PREFIX", "Capture")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# App FastAPI
# =========================
app = FastAPI(title="Doc2Markdown (Paddle + MarkItDown)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========
# Utilitaires
# ==========
def _determine_lang_code(s: Optional[str]) -> str:
    if not s:
        s = DEFAULT_OCR_LANGS
    s = s.strip().lower()
    # PaddleOCR codes usuels
    if "fra" in s or s.startswith("fr"):
        return "fr"
    if "eng" in s or s.startswith("en"):
        return "en"
    if "deu" in s or s.startswith("de"):
        return "de"
    if "es" in s or "spa" in s:
        return "es"
    if "jap" in s or s.startswith("ja"):
        return "japan"
    if "kor" in s or s.startswith("ko"):
        return "korean"
    if "zh" in s or "chi" in s:
        return "ch"
    # fallback (paddle sait "fr", "en", …)
    return s

_bullet_rx = re.compile(r"^\s*[•·●◦▪]\s+")
def _md_cleanup(md: str) -> str:
    if not md:
        return md
    lines = []
    for L in md.replace("\r", "").split("\n"):
        if _bullet_rx.match(L):
            L = _bullet_rx.sub("- ", L)
        # listes numérotées "1) " -> "1. "
        L = re.sub(r"^\s*(\d+)[\)\]]\s+", r"\1. ", L)
        lines.append(L.rstrip())
    return "\n".join(lines).strip()

def _pil_resize_max(im: Image.Image, max_w: int) -> Image.Image:
    if max_w and im.width > max_w:
        ratio = max_w / im.width
        new_h = int(im.height * ratio)
        return im.resize((max_w, new_h), Image.LANCZOS)
    return im

def _pil_to_data_uri(im: Image.Image, fmt: str = "png", quality: int = 85) -> str:
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

def _save_if_needed(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def _guess_ext(name: str) -> str:
    return os.path.splitext(name)[1].lower().lstrip(".")

# =====================================
# OCR engines (lazy init pour démarrage)
# =====================================
_ocr_basic: Optional[PaddleOCR] = None
_ocr_struct: Optional[PPStructureV3] = None

def get_ocr_basic(lang: str) -> PaddleOCR:
    global _ocr_basic
    if _ocr_basic is None or getattr(_ocr_basic, "_lang", "") != lang:
        _ocr_basic = PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            show_log=False,
            use_gpu=False
        )
        _ocr_basic._lang = lang
    return _ocr_basic

def get_ocr_struct(lang: str) -> PPStructureV3:
    global _ocr_struct
    # Un seul pipeline struct (langue multi-Latin ok). Si tu veux isoler par langue, dupliques comme basic.
    if _ocr_struct is None:
        _ocr_struct = PPStructureV3(
            use_gpu=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )
    return _ocr_struct

# =====================
# Conversions : TEXT-ONLY
# =====================
def pdf_text_only_md(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        parts = []
        for i, page in enumerate(doc):
            if i >= OCR_MAX_PAGES:
                break
            txt = page.get_text("text")
            if txt.strip():
                parts.append(txt.strip())
        return _md_cleanup("\n\n".join(parts))
    finally:
        doc.close()

def docx_text_only_md(docx_bytes: bytes) -> str:
    if docx is None:
        # fallback MarkItDown si dispo
        if MarkItDown:
            md = MarkItDown()
            res = md.convert_stream(io.BytesIO(docx_bytes), file_name="file.docx")
            return _md_cleanup(getattr(res, "text_content", "") or "")
        raise HTTPException(status_code=500, detail="python-docx n'est pas installé")
    d = docx.Document(io.BytesIO(docx_bytes))
    out = []
    for p in d.paragraphs:
        text = p.text or ""
        if not text.strip():
            continue
        style = (p.style.name or "") if p.style else ""
        md_line = text
        if style.startswith("Heading"):
            try:
                level = int(style.split()[-1])
            except Exception:
                level = 1
            level = max(1, min(level, 6))
            md_line = "#" * level + " " + text
        out.append(md_line.strip())
    # tables -> markdown
    for t in d.tables:
        rows = t.rows
        if not rows:
            continue
        # header
        header = [c.text.strip() for c in rows[0].cells]
        lines = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join([" --- "] * len(header)) + "|")
        for row in rows[1:]:
            cells = [c.text.strip().replace("\n", "<br>") for c in row.cells]
            lines.append("| " + " | ".join(cells) + " |")
        out.append("\n".join(lines))
    return _md_cleanup("\n\n".join(out))

def xlsx_csv_text_only_md(file_bytes: bytes, ext: str) -> str:
    if ext == "csv":
        # encodage robuste
        txt = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                txt = file_bytes.decode(enc)
                break
            except Exception:
                pass
        if txt is None:
            raise HTTPException(status_code=400, detail="Impossible de décoder le CSV")
        df = pd.read_csv(io.StringIO(txt), dtype=str).fillna("")
        return df.to_markdown(index=False)
    # XLSX/XLS
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        out = []
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet, dtype=str).fillna("")
            if len(xl.sheet_names) > 1:
                out.append(f"## {sheet}")
            out.append(df.to_markdown(index=False))
        return "\n\n".join(out)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lecture tableur échouée: {e}")

# ===================================
# Conversions : OCR / IMAGES & STRUCT
# ===================================
def pdf_quality_with_ocr(pdf_bytes: bytes, lang: str) -> str:
    """
    Mode qualité : structure + images en base64 + OCR des captures.
    Utilise PPStructureV3 puis intègre les images en data URI.
    """
    # Ecrire sur disque (PPStructure marche mieux avec un chemin)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        pdf_path = f.name
    try:
        pipeline = get_ocr_struct(lang)
        results = pipeline.predict(input=pdf_path)  # liste de pages
        # Concaténer markdown + collecter images
        md_parts: List[Dict[str, Any]] = []
        imgs: Dict[str, Image.Image] = {}
        for res in results:
            md_info = getattr(res, "markdown", None)
            if isinstance(md_info, dict):
                md_parts.append(md_info)
                for k, v in (md_info.get("markdown_images") or {}).items():
                    imgs[k] = v
        try:
            full_md = pipeline.concatenate_markdown_pages(md_parts)
        except Exception:
            full_md = "\n\n".join(p.get("markdown_text", "") for p in md_parts)
        # Remplacer chemins d'images par data URI
        for path, pil_img in imgs.items():
            if pil_img is None:
                continue
            pil_img = _pil_resize_max(pil_img, IMG_MAX_WIDTH)
            data_uri = _pil_to_data_uri(pil_img, IMG_FORMAT, IMG_JPEG_QUALITY)
            full_md = full_md.replace(f"]({path})", f"]({data_uri})")
        return _md_cleanup(full_md)
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass

def pdf_fast_ocr_text(pdf_bytes: bytes, lang: str) -> str:
    """
    Mode rapide : OCR ligne à ligne (texte), sans images intégrées.
    Si la page a déjà du texte (couche texte PDF), on le garde.
    """
    ocr = get_ocr_basic(lang)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        out = []
        for i, page in enumerate(doc):
            if i >= OCR_MAX_PAGES:
                break
            # Si texte natif dispo -> on prend
            t = page.get_text("text")
            if t.strip():
                out.append(t.strip())
                continue
            # Sinon OCR de la page
            pix = page.get_pixmap(matrix=fitz.Matrix(OCR_DPI/72, OCR_DPI/72))
            im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            arr = np.array(im)
            res = ocr.ocr(arr, cls=False) or []
            lines = []
            # PaddleOCR (v2.7.x) renvoie [[(box),(text,conf)],...]
            for item in res:
                if isinstance(item, list):
                    for line in item:
                        if isinstance(line, list) and len(line) > 1:
                            txt = line[1][0]
                            lines.append(txt)
                else:
                    # fallback
                    try:
                        lines.append(item[1][0])
                    except Exception:
                        pass
            if lines:
                out.append("\n".join(lines))
        return _md_cleanup("\n\n".join(out))
    finally:
        doc.close()

def docx_with_images_and_ocr(docx_bytes: bytes, lang: str, embed_images: bool = True, add_ocr_text: bool = True) -> str:
    """
    DOCX -> Markdown, images intégrées (base64), OCR des images si add_ocr_text=True (texte sous l'image).
    """
    if docx is None:
        if MarkItDown:
            md = MarkItDown()
            res = md.convert_stream(io.BytesIO(docx_bytes), file_name="file.docx")
            return _md_cleanup(getattr(res, "text_content", "") or "")
        raise HTTPException(status_code=500, detail="python-docx non installé")
    d = docx.Document(io.BytesIO(docx_bytes))
    # Extraire toutes les images du package (fallback si relations manquantes)
    image_data_map = {}
    with ZipFile(io.BytesIO(docx_bytes)) as z:
        for name in z.namelist():
            if name.startswith("word/media/"):
                image_data_map[name] = z.read(name)

    ocr = get_ocr_basic(lang) if add_ocr_text else None
    out: List[str] = []

    # Paragraphes (avec bold/italic basique)
    def fmt_run(run) -> str:
        txt = run.text or ""
        if not txt:
            return ""
        if run.bold and run.italic:
            return f"***{txt}***"
        if run.bold:
            return f"**{txt}**"
        if run.italic:
            return f"*{txt}*"
        return txt

    # Parcours du corps (paragraphes + tables)
    body = d.element.body
    table_idx = 0
    for child in body.iterchildren():
        if child.tag.endswith('}tbl'):
            # Table
            if table_idx >= len(d.tables):
                continue
            table = d.tables[table_idx]
            table_idx += 1
            rows = table.rows
            if not rows:
                continue
            ncols = len(rows[0].cells)
            lines = []
            # header
            head = []
            for c in rows[0].cells:
                cell_text = []
                for p in c.paragraphs:
                    for run in p.runs:
                        cell_text.append(fmt_run(run))
                head.append("".join(cell_text).strip())
            lines.append("| " + " | ".join(head) + " |")
            lines.append("|" + "|".join([" --- "] * ncols) + "|")
            # data
            for r in rows[1:]:
                row_cells = []
                for c in r.cells:
                    parts = []
                    for p in c.paragraphs:
                        for run in p.runs:
                            # image dans cellule ?
                            if embed_images and run._element.xpath('.//w:drawing'):
                                blips = run._element.xpath('.//a:blip', namespaces={'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
                                if blips:
                                    rId = blips[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                                    img_bytes = None
                                    if rId:
                                        part = d.part.related_parts.get(rId)
                                        if part and hasattr(part, 'blob'):
                                            img_bytes = part.blob
                                    if img_bytes is None and image_data_map:
                                        img_bytes = next(iter(image_data_map.values()))
                                    if img_bytes:
                                        im = Image.open(io.BytesIO(img_bytes))
                                        im = _pil_resize_max(im, IMG_MAX_WIDTH)
                                        data_uri = _pil_to_data_uri(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                                        parts.append(f"![{IMG_ALT_PREFIX}]({data_uri})")
                                        if add_ocr_text and ocr:
                                            res = ocr.ocr(np.array(im), cls=False) or []
                                            lines_ocr = []
                                            for item in res:
                                                if isinstance(item, list):
                                                    for line in item:
                                                        if isinstance(line, list) and len(line) > 1:
                                                            lines_ocr.append(line[1][0])
                                            if lines_ocr:
                                                parts.append("\n" + "\n".join(lines_ocr))
                            parts.append(fmt_run(run))
                    row_cells.append("".join(parts).replace("\n", "<br>").strip())
                lines.append("| " + " | ".join(row_cells) + " |")
            out.append("\n".join(lines))
        elif child.tag.endswith('}p'):
            # Paragraphe
            para = docx.text.paragraph.Paragraph(child, d)
            style = para.style.name if para.style else ""
            text_parts = []
            prefix = ""
            if style.startswith("Heading"):
                try:
                    lv = int(style.split()[-1])
                except Exception:
                    lv = 1
                prefix = "#" * max(1, min(lv, 6)) + " "
            elif "List" in style or "Bullet" in style or "Number" in style:
                prefix = "- "
            # runs (texte + images inline)
            for run in para.runs:
                if embed_images and run._element.xpath('.//w:drawing'):
                    blips = run._element.xpath('.//a:blip', namespaces={'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
                    if blips:
                        rId = blips[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        img_bytes = None
                        if rId:
                            part = d.part.related_parts.get(rId)
                            if part and hasattr(part, 'blob'):
                                img_bytes = part.blob
                        if img_bytes is None and image_data_map:
                            img_bytes = next(iter(image_data_map.values()))
                        if img_bytes:
                            im = Image.open(io.BytesIO(img_bytes))
                            im = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_data_uri(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            text_parts.append(f"![{IMG_ALT_PREFIX}]({data_uri})")
                            if add_ocr_text:
                                ocr = get_ocr_basic(lang)
                                res = ocr.ocr(np.array(im), cls=False) or []
                                lines_ocr = []
                                for item in res:
                                    if isinstance(item, list):
                                        for line in item:
                                            if isinstance(line, list) and len(line) > 1:
                                                lines_ocr.append(line[1][0])
                                if lines_ocr:
                                    text_parts.append("\n" + "\n".join(lines_ocr))
                text_parts.append(fmt_run(run))
            text = (prefix + "".join(text_parts)).strip()
            if text:
                out.append(text)
    return _md_cleanup("\n\n".join(out))

# ===========================
# Route UI simple (inchangée)
# ===========================
HTML = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Doc → Markdown</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:24px;background:#0b0f14;color:#e6edf3}
.card{max-width:980px;margin:0 auto;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:16px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
label{opacity:.9}
input[type=file]{display:none}
.drop{border:1px dashed rgba(255,255,255,.2);padding:16px;border-radius:12px;text-align:center;margin:8px 0;cursor:pointer}
select,input[type=text]{background:#0e141b;color:#e6edf3;border:1px solid rgba(255,255,255,.1);border-radius:6px;padding:6px 8px}
button{background:#63b3ff;color:#0b0f14;border:none;border-radius:8px;padding:8px 12px;cursor:pointer}
textarea{width:100%;min-height:260px;background:#0e141b;color:#e6edf3;border-radius:8px;border:1px solid rgba(255,255,255,.1);padding:8px}
.small{opacity:.7}
.switch{position:relative;display:inline-block;width:40px;height:20px}
.switch input{opacity:0;width:0;height:0}
.slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:rgba(255,255,255,.15);border-radius:20px;transition:.2s}
.slider:before{position:absolute;content:"";height:14px;width:14px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.2s}
input:checked + .slider{background:#63b3ff}
input:checked + .slider:before{transform:translateX(20px)}
</style>
</head>
<body>
<div class="card">
  <h2>Conversion de documents en Markdown</h2>
  <div class="row">
    <label>Fichier :</label>
    <input id="file" type="file">
    <div id="meta" class="small"></div>
  </div>
  <div class="drop" id="drop">Glissez-déposez ici (ou cliquez)</div>

  <div class="row">
    <label>Activer plugins MarkItDown (texte uniquement)</label>
    <label class="switch"><input id="mkd" type="checkbox" checked><span class="slider"></span></label>

    <label>Forcer OCR + images (base64)</label>
    <label class="switch"><input id="ocr" type="checkbox"><span class="slider"></span></label>

    <label>Mode</label>
    <select id="mode">
      <option value="fast">Rapide (texte prioritaire)</option>
      <option value="quality">Qualité (structure + images)</option>
    </select>

    <label>Langue OCR</label>
    <input id="lang" type="text" placeholder="fra, eng, fra+eng" value="{DEFAULT_LANG}" style="width:140px">
  </div>

  <div class="row" style="margin-top:8px">
    <button id="go">Convertir</button>
    <button id="copy" type="button">Copier</button>
    <a id="dl" style="display:none">Télécharger</a>
    <div id="status" class="small"></div>
  </div>

  <div style="margin-top:8px">
    <textarea id="md" placeholder="Le Markdown apparaîtra ici..."></textarea>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const drop=$("drop"), fi=$("file"), meta=$("meta");
drop.onclick=()=>fi.click();
drop.ondragover=e=>{e.preventDefault();drop.style.background="rgba(255,255,255,.06)";}
drop.ondragleave=()=>drop.style.background="";
drop.ondrop=e=>{e.preventDefault();drop.style.background=""; if(e.dataTransfer.files[0]) fi.files=e.dataTransfer.files; show();}
fi.onchange=show;
function show(){ if(!fi.files[0]){meta.textContent="";return;} const f=fi.files[0]; const s=f.size<1024?f.size+" B":(f.size<1048576?(f.size/1024).toFixed(1)+" KB":(f.size/1048576).toFixed(1)+" MB"); meta.textContent=f.name+" — "+s; }

$("copy").onclick=async()=>{ try{ await navigator.clipboard.writeText($("md").value||""); $("status").textContent="Copié !"; setTimeout(()=>$("status").textContent="",1000);}catch{} };

$("go").onclick=async()=>{
  const f=fi.files[0]; if(!f){alert("Choisis un fichier");return;}
  $("status").textContent="Conversion en cours...";
  const fd=new FormData();
  fd.append("file",f);
  fd.append("activer_plugin_markitdown", $("mkd").checked ? "true":"false");
  fd.append("forcer_ocr", $("ocr").checked ? "true":"false");
  fd.append("mode", $("mode").value);
  fd.append("lang", $("lang").value||"");
  const res=await fetch("/convert",{method:"POST",body:fd});
  if(!res.ok){ $("md").value="ERREUR: "+res.status+" "+(await res.text()); $("status").textContent=""; return;}
  const text=await res.text();
  $("md").value=text||"";
  const a=$("dl");
  a.href=URL.createObjectURL(new Blob([text],{type:"text/markdown;charset=utf-8"}));
  a.download=(f.name.replace(/\.[^.]+$/,"")||"sortie")+".md";
  a.textContent="Télécharger";
  a.style.display="inline-block";
  $("status").textContent="OK";
};
</script>
</body></html>
""".replace("{DEFAULT_LANG}", DEFAULT_OCR_LANGS)

@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(HTML)

# ===========================
# Point d'entrée /convert
# ===========================
@app.post("/convert", response_class=PlainTextResponse)
async def convert(
    file: UploadFile = File(...),
    activer_plugin_markitdown: bool = Form(True),
    forcer_ocr: bool = Form(False),
    mode: str = Form("fast"),  # "fast" | "quality"
    lang: Optional[str] = Form(None)
):
    """
    - activer_plugin_markitdown=True  -> texte uniquement (si OCR OFF), sinon texte + OCR captures + images base64
    - forcer_ocr=True                 -> OCR activé + images base64 + texte OCR sous les captures
    - mode: fast (texte prioritaire), quality (structure + images)
    - lang: fra | eng | fra+eng ...
    Retour: Markdown (text/plain)
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Fichier manquant")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Fichier vide")

    # Sauvegarde upload si demandé
    if SAVE_UPLOADS:
        _save_if_needed(os.path.join(UPLOAD_DIR, file.filename), content)

    ext = _guess_ext(file.filename)
    lang_code = _determine_lang_code(lang)

    try:
        # ==========
        # PDF
        # ==========
        if ext == "pdf":
            if activer_plugin_markitdown and not forcer_ocr:
                # Texte uniquement
                md = pdf_text_only_md(content)
            elif forcer_ocr:
                # OCR + images + texte OCR (captures) — qualité ou rapide
                if mode == "quality":
                    md = pdf_quality_with_ocr(content, lang_code)
                else:
                    # rapide : priorité texte, sinon OCR texte (pas d’images)
                    md = pdf_fast_ocr_text(content, lang_code)
            else:
                # MarkItDown si dispo (texte propre), sinon texte PyMuPDF
                if MarkItDown:
                    md_engine = MarkItDown()
                    res = md_engine.convert_stream(io.BytesIO(content), file_name=file.filename)
                    md = _md_cleanup(getattr(res, "text_content", "") or "")
                else:
                    md = pdf_text_only_md(content)

        # ==========
        # DOCX
        # ==========
        elif ext in ("docx", "doc"):
            if ext == "doc":
                raise HTTPException(status_code=400, detail=".doc non supporté (convertis en .docx)")
            if activer_plugin_markitdown and not forcer_ocr:
                # texte uniquement
                md = docx_text_only_md(content)
            elif forcer_ocr:
                # images base64 + OCR sous l'image (mode qualité/rapide agit peu ici)
                md = docx_with_images_and_ocr(
                    content,
                    lang=lang_code,
                    embed_images=True,
                    add_ocr_text=True
                )
            else:
                # MarkItDown si dispo
                if MarkItDown:
                    md_engine = MarkItDown()
                    res = md_engine.convert_stream(io.BytesIO(content), file_name=file.filename)
                    md = _md_cleanup(getattr(res, "text_content", "") or "")
                else:
                    md = docx_text_only_md(content)

        # ==========
        # CSV / XLSX / XLS
        # ==========
        elif ext in ("csv", "xlsx", "xls"):
            md = xlsx_csv_text_only_md(content, ext)

        # ==========
        # Images (png/jpg...) -> OCR si forcer_ocr, sinon image embed
        # ==========
        elif ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"):
            im = Image.open(io.BytesIO(content)).convert("RGB")
            if forcer_ocr:
                ocr = get_ocr_basic(lang_code)
                res = ocr.ocr(np.array(im), cls=False) or []
                lines = []
                for item in res:
                    if isinstance(item, list):
                        for line in item:
                            if isinstance(line, list) and len(line) > 1:
                                lines.append(line[1][0])
                text_md = "\n".join(lines) if lines else ""
                # Intégrer aussi l'image (quality) selon préférence
                im = _pil_resize_max(im, IMG_MAX_WIDTH)
                data_uri = _pil_to_data_uri(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                if lines:
                    md = f"![{IMG_ALT_PREFIX}]({data_uri})\n\n{text_md}"
                else:
                    md = f"![{IMG_ALT_PREFIX}]({data_uri})"
            else:
                # plugin texte uniquement => on n'intègre pas d'image si explicitement texte-only
                if activer_plugin_markitdown:
                    md = ""  # image seule = pas de texte
                else:
                    im = _pil_resize_max(im, IMG_MAX_WIDTH)
                    data_uri = _pil_to_data_uri(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                    md = f"![{IMG_ALT_PREFIX}]({data_uri})"

        else:
            raise HTTPException(status_code=400, detail=f"Type non supporté: .{ext}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de conversion: {type(e).__name__}: {e}")

    md = _md_cleanup(md or "")

    # Sauvegarde output si demandé
    if SAVE_OUTPUTS:
        out_name = os.path.splitext(file.filename)[0] + ".md"
        _save_if_needed(os.path.join(OUTPUT_DIR, out_name), md.encode("utf-8"))

    return PlainTextResponse(md)
