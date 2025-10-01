import os
import io
import re
import time
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
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "50"))
OCR_MIN_CHARS      = int(os.getenv("OCR_MIN_CHARS", "500"))
OCR_KEEP_SPACES    = os.getenv("OCR_KEEP_SPACES", "true").lower() == "true"
OCR_PSMS           = [p.strip() for p in os.getenv("OCR_PSMS", "6,4,3,11").split(",")]
OCR_DPI_CANDIDATES = [int(x) for x in os.getenv("OCR_DPI_CANDIDATES", "300,350,400").split(",")]
OCR_SCORE_GOOD_ENOUGH = float(os.getenv("OCR_SCORE_GOOD_ENOUGH", "0.6"))

# Politique OCR images & fallback
IMAGE_OCR_MODE = os.getenv("IMAGE_OCR_MODE", "smart").strip()  # smart | conservative | always | never
IMAGE_OCR_MIN_WORDS = int(os.getenv("IMAGE_OCR_MIN_WORDS", "10"))
OCR_TEXT_QUALITY_MIN = float(os.getenv("OCR_TEXT_QUALITY_MIN", "0.30"))
OCR_DISABLE_PAGE_FALLBACK = os.getenv("OCR_DISABLE_PAGE_FALLBACK", "false").lower() == "true"  # Fallback OCR pages sans texte activé par défaut

# Embedding images base64
# none | ocr_only | all
EMBED_IMAGES       = os.getenv("EMBED_IMAGES", "ocr_only").strip()
IMG_FORMAT         = os.getenv("IMG_FORMAT", "jpeg").strip().lower()
IMG_JPEG_QUALITY   = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH      = int(os.getenv("IMG_MAX_WIDTH", "1600"))
IMG_ALT_PREFIX     = os.getenv("IMG_ALT_PREFIX", "Capture").strip()

# (Optionnel) Azure Document Intelligence
DEFAULT_DOCINTEL_ENDPOINT = os.getenv("DEFAULT_DOCINTEL_ENDPOINT", "").strip()

# Dossiers persistants
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="3.7-ocr-fallback+cleanup")

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
# Helpers génériques & markdown
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
    if not md:
        return md
    lines = []
    for L in md.replace("\r", "").split("\n"):
        l = re.sub(r"[ \t]+$", "", L)
        l = re.sub(r"^\s*[•·●◦▪]\s+", "- ", l)         # bullets unicode -> '- '
        l = re.sub(r"^\s*(\d+)[\)\]]\s+", r"\1. ", l)  # "1)" ou "1]" -> "1. "
        lines.append(l)
    txt = "\n".join(lines)
    # Encadrer les grilles ASCII dans ```text
    txt = re.sub(
        r"(?:^|\n)((?:[|+\-=_].*\n){2,})",
        lambda m: "```text\n" + m.group(1).strip() + "\n```",
        txt, flags=re.S
    )
    return txt.strip()

# ---------------------------
# OCR utils
# ---------------------------
_table_chars = re.compile(r"[|+\-=_]{3,}")

def _tess_config(psm: str) -> str:
    cfg = f"--psm {psm} --oem 1"
    if OCR_KEEP_SPACES:
        cfg += " -c preserve_interword_spaces=1"
    return cfg

def _preprocess_for_ocr(im: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.MedianFilter(size=3))
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
    return (pipes*1.0 + plus*0.6 + dashes*0.3 + ascii_blocks*2.0)/n + len(txt)/5000.0

def _ocr_quality_score(txt: str) -> float:
    if not txt:
        return 0.0
    t = txt.replace("\n", " ").strip()
    if not t:
        return 0.0
    n = len(t)
    alnum_ratio = sum(ch.isalnum() for ch in t) / n
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,12}", t)
    return max(0.0, min(1.0, 0.5 * alnum_ratio + 0.5 * min(1.0, n/600) + 0.02 * len(words)))

def _fast_has_text(im: Image.Image, langs: str, min_words: int) -> bool:
    try:
        data = pytesseract.image_to_data(im, lang=langs, config="--psm 11 --oem 1", output_type=pytesseract.Output.DICT)
        words = [w for w in (data.get("text") or []) if w and re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{2,}", w)]
        return len(words) >= min_words
    except Exception:
        return False

def _classify_ocr_block(txt: str) -> str:
    t = (txt or "").strip()
    if not t:
        return ""
    # Tables ASCII détectées -> bloc de code
    if re.search(r"^\s*\+.*[-+].*\+\s*$", t, flags=re.M) or re.search(r"^\s*\|.*\|\s*$", t, flags=re.M):
        return f"```text\n{t}\n```"
    # Lignes type console / SQL détectées -> bloc de code approprié
    if re.search(r"^\s*(\$|#)\s", t, flags=re.M) or "mysql>" in t:
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|SHOW|GRANT|CHANGE\s+MASTER|START\s+(SLAVE|REPLICA))\b", t, flags=re.I):
            return f"```sql\n{t}\n```"
        return f"```bash\n{t}\n```"
    return t

def _ocr_image_best(im: Image.Image, langs: str) -> Tuple[str, float]:
    # Appliquer OCR (plusieurs PSM, prétraitements) et renvoyer le meilleur texte + score qualité
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im2 = _preprocess_for_ocr(im)
    best_txt, best_score = "", -1e9
    for psm in OCR_PSMS:
        cfg = _tess_config(psm)
        t1 = pytesseract.image_to_string(im, lang=langs, config=cfg) or ""
        s1 = _score_text_for_table(t1)
        cand_txt, cand_score = t1, s1
        t2 = pytesseract.image_to_string(im2, lang=langs, config=cfg) or ""
        s2 = _score_text_for_table(t2)
        if s2 > cand_score:
            cand_txt, cand_score = t2, s2
        if cand_score > best_score:
            best_txt, best_score = cand_txt, cand_score
        if best_score >= OCR_SCORE_GOOD_ENOUGH:
            break
    q = _ocr_quality_score(best_txt)
    if q < OCR_TEXT_QUALITY_MIN:
        return "", q
    return best_txt.strip(), q

def ocr_image_bytes(img_bytes: bytes, langs: str) -> Tuple[str, float]:
    # OCR d'une image isolée (renvoie texte OCRisé formaté + score qualité)
    with Image.open(io.BytesIO(img_bytes)) as im:
        txt, score = _ocr_image_best(im, langs)
        return _classify_ocr_block(txt), score

def _raster_pdf_page(page: fitz.Page, dpi: int) -> Image.Image:
    # Rendu raster d'une page PDF entière à la résolution souhaitée
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# ---------------------------
# Images utilitaires
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

def crop_bbox_image(page: fitz.Page, bbox: Tuple[float, float, float, float], dpi: int) -> Optional[Image.Image]:
    try:
        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0, y0, x1, y1)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None

# ---------------------------
# PDF inline : texte + images (OCR smart) + fallback contrôlé
# ---------------------------
def _is_bold(flags: int) -> bool:
    return bool(flags & 1 or flags & 32)  # BOLD | FAKEBOLD

def _median_font_size(page_raw: Dict[str, Any]) -> float:
    sizes = []
    for b in page_raw.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if s.get("text", "").strip():
                    sizes.append(float(s.get("size", 0)))
    if not sizes:
        return 0.0
    sizes.sort()
    mid = len(sizes) // 2
    return sizes[mid] if len(sizes) % 2 == 1 else (sizes[mid-1] + sizes[mid]) / 2.0

def _classify_heading(size: float, median_size: float) -> Optional[str]:
    if median_size <= 0:
        return None
    if size >= median_size * 1.8: return "#"
    if size >= median_size * 1.5: return "##"
    if size >= median_size * 1.25: return "###"
    return None

_bullet_re = re.compile(r"^\s*(?:[-–—•·●◦▪]|\d+[.)])\s+")

def _line_to_md(spans: List[Dict[str, Any]], median_size: float) -> str:
    parts = []
    max_size = 0.0
    for sp in spans:
        t = sp.get("text", "")
        if not t:
            continue
        size = float(sp.get("size", 0))
        max_size = max(max_size, size)
        parts.append(f"**{t}**" if _is_bold(int(sp.get("flags", 0))) else t)
    raw = "".join(parts).strip()
    if not raw:
        return ""
    h = _classify_heading(max_size, median_size)
    if h and len(raw) < 180:
        return f"{h} {raw}"
    if _bullet_re.match(raw):
        return raw
    return raw

def _wrap_tables_as_code(txt: str) -> str:
    if not txt:
        return txt
    out, buf, in_blk = [], [], False
    for line in txt.splitlines():
        is_tbl = _table_chars.search(line) is not None or line.strip().startswith("|")
        if is_tbl and not in_blk:
            in_blk = True; out.append("```text"); buf = []
        if in_blk and not is_tbl and buf:
            out.extend(buf); out.append("```")
            in_blk = False; out.append(line); buf = []; continue
        if in_blk:
            buf.append(line)
        else:
            out.append(line)
    if in_blk:
        out.extend(buf); out.append("```")
    return "\n".join(out)

def render_pdf_markdown_inline(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_lines: List[str] = []
    meta: Dict[str, Any] = {"engine": "pymupdf_inline", "pages": doc.page_count}
    try:
        total_pages = min(doc.page_count, OCR_MAX_PAGES)
        for p in range(total_pages):
            page = doc.load_page(p)
            raw = page.get_text("rawdict") or {}
            median_size = _median_font_size(raw)
            line_h_med = 12.0 if median_size == 0 else median_size * 1.0
            band_h = max(8.0, line_h_med * 1.2)

            atoms = []
            page_w = page.rect.width
            page_h = page.rect.height
            page_area = max(1.0, page_w * page_h)

            # 1) TEXTE (par LIGNE) + marquage images (traitées ensuite)
            for b in raw.get("blocks", []):
                btype = b.get("type", 0)
                bbox = tuple(b.get("bbox", (0, 0, 0, 0)))
                x0, y0, x1, y1 = bbox
                x0 = max(0.0, min(x0, page_w)); x1 = max(0.0, min(x1, page_w))
                y0 = max(0.0, min(y0, page_h)); y1 = max(0.0, min(y1, page_h))
                bbox = (x0, y0, x1, y1)

                if btype == 0:  # bloc texte
                    for line in b.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        lx0 = min(s.get("bbox", [x0, y0, x1, y1])[0] for s in spans if s.get("bbox"))
                        ly0 = min(s.get("bbox", [x0, y0, x1, y1])[1] for s in spans if s.get("bbox"))
                        lx1 = max(s.get("bbox", [x0, y0, x1, y1])[2] for s in spans if s.get("bbox"))
                        ly1 = max(s.get("bbox", [x0, y0, x1, y1])[3] for s in spans if s.get("bbox"))
                        line_bbox = (lx0, ly0, lx1, ly1)

                        md_line = _line_to_md(spans, median_size).strip()
                        if not md_line:
                            continue
                        atoms.append({
                            "bbox": line_bbox,
                            "md": md_line,
                            "kind": "text",
                            "text_len": len(md_line),
                            "area_ratio": ((lx1 - lx0) * (ly1 - ly0)) / page_area
                        })

                elif btype == 1:  # bloc image
                    atoms.append({
                        "bbox": bbox,
                        "md": None,
                        "kind": "image_raw",
                        "text_len": 0,
                        "area_ratio": ((x1 - x0) * (y1 - y0)) / page_area
                    })

            has_vector_text = any(a["kind"] == "text" for a in atoms)

            # 2) IMAGES -> OCR (selon mode) ou embed base64
            processed = []
            for a in atoms:
                if a["kind"] != "image_raw":
                    processed.append(a)
                    continue

                im = crop_bbox_image(page, a["bbox"], OCR_DPI)
                if im is None:
                    continue

                area_ratio = a["area_ratio"]
                is_background_like = area_ratio > 0.85

                # Mode "never": pas d'OCR
                # Mode "smart": OCR seulement si l'image contient du texte
                # Modes "always" et "conservative": OCR sur toutes les images (y compris plein-page)
                do_img_ocr = False
                if IMAGE_OCR_MODE == "never":
                    do_img_ocr = False
                elif IMAGE_OCR_MODE == "smart":
                    do_img_ocr = _fast_has_text(im, OCR_LANGS, IMAGE_OCR_MIN_WORDS)
                else:
                    do_img_ocr = True

                md_img = ""
                txt, q = "", 0.0
                if do_img_ocr:
                    txt, q = _ocr_image_best(im, OCR_LANGS)

                if txt and q >= OCR_TEXT_QUALITY_MIN:
                    md_img = _classify_ocr_block(txt.strip())
                else:
                    if EMBED_IMAGES in ("all", "ocr_only"):
                        im = _pil_resize_max(im, IMG_MAX_WIDTH)
                        data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                        md_img = f'![{IMG_ALT_PREFIX} – page {p+1}]({data_uri})'

                if md_img:
                    processed.append({
                        "bbox": a["bbox"],
                        "md": md_img,
                        "kind": "image",
                        "text_len": len(txt or ""),
                        "area_ratio": area_ratio
                    })

            atoms = processed

            # 3) Tri de lecture simple (haut → bas, gauche → droite)
            def sort_key(a):
                x0, y0, x1, y1 = a["bbox"]
                y_center = 0.5 * (y0 + y1)
                return (int(y_center / band_h), x0, y0)
            atoms.sort(key=sort_key)

            # 4) Concaténation lignes + wrap tableaux
            page_buf: List[str] = []
            para_buf: List[str] = []
            last_band = None

            def flush_para():
                if not para_buf:
                    return
                block_txt = "\n".join(para_buf)
                if _table_chars.search(block_txt):
                    page_buf.append("```text"); page_buf.append(block_txt); page_buf.append("```")
                else:
                    page_buf.append(block_txt)
                para_buf.clear()

            for a in atoms:
                x0, y0, x1, y1 = a["bbox"]
                y_center = 0.5 * (y0 + y1)
                band = int(y_center / band_h)
                md = a["md"]
                if last_band is not None and band != last_band:
                    flush_para()
                last_band = band
                if a["kind"] == "text":
                    para_buf.append(md)
                else:
                    flush_para()
                    page_buf.append(md)
            flush_para()

            # Indicateur texte extrait sur la page (vectoriel ou OCR)
            has_text_content = any(a["text_len"] > 0 for a in atoms)
            # 5) Fallback OCR page : uniquement si aucun texte extrait et fallback activé
            if not has_text_content and not OCR_DISABLE_PAGE_FALLBACK:
                try:
                    best_txt, best_score = "", -1e9
                    for d in OCR_DPI_CANDIDATES:
                        im_page = _raster_pdf_page(page, d)
                        txt, score = _ocr_image_best(im_page, OCR_LANGS)
                        if score > best_score:
                            best_txt, best_score = txt, score
                        if best_score >= OCR_SCORE_GOOD_ENOUGH:
                            break
                    if best_txt.strip():
                        page_buf.append("<!-- OCR page fallback -->")
                        page_buf.append(_wrap_tables_as_code(best_txt.strip()))
                except Exception:
                    pass

            if page_buf:
                md_lines.append("\n\n".join(page_buf))

        final_md = "\n\n".join([l for l in md_lines if l.strip()]).strip()
        return (final_md, meta)
    finally:
        doc.close()

# ---------------------------
# UI web (exactement comme ton origine, avec sliders .switch)
# ---------------------------
HTML_PAGE = r'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>MarkItDown UI</title>
  <style>
    :root{
      color-scheme: dark;
      --bg: #0b0f14;
      --bg2: #0e141b;
      --card: rgba(255,255,255,0.06);
      --card-border: rgba(255,255,255,0.08);
      --text: #e6edf3;
      --muted: #9fb0bf;
      --accent: #63b3ff;
      --accent-2: #8f7aff;
      --ok: #58d68d;
      --err: #ff6b6b;
      --shadow: 0 6px 24px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03);
      --radius: 16px;
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      color:var(--text);
      background:
        radial-gradient(1000px 600px at 20% -10%, #183a58 0%, transparent 60%),
        radial-gradient(900px 500px at 120% 10%, #3b2d6a 0%, transparent 55%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
      background-attachment: fixed;
      line-height:1.55;
      padding:32px 20px 40px;
    }
    h1{margin:0 0 .35rem 0; font-size:1.65rem; letter-spacing:.2px}
    .sub{color:var(--muted); font-size:.95rem; margin-bottom:18px}
    .container{max-width:1060px; margin:0 auto}
    .card{
      background:var(--card);
      border:1px solid var(--card-border);
      border-radius:var(--radius);
      box-shadow:var(--shadow);
      padding:18px;
      margin-top:16px;
      backdrop-filter: blur(8px);
    }
    .row{display:flex; gap:12px; align-items:center; flex-wrap:wrap}
    label{font-weight:600}
    input[type="text"], input[type="file"], textarea{
      background:rgba(255,255,255,0.03);
      color:var(--text);
      border:1px solid rgba(255,255,255,0.12);
      border-radius:12px;
      padding:10px 12px;
      outline:none;
      transition:border .15s ease, box-shadow .15s ease;
    }
    input[type="text"]:focus, textarea:focus{
      border-color:var(--accent);
      box-shadow:0 0 0 3px rgba(99,179,255,.15);
    }
    textarea{
      width:100%;
      min-height:280px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size:.95rem;
      resize: vertical;
    }
    button{
      padding:10px 16px;
      border-radius:12px;
      border:1px solid rgba(255,255,255,0.12);
      color:#0b0f14;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      cursor:pointer;
      font-weight:700;
      letter-spacing:.2px;
      transition:transform .06s ease, filter .15s ease, opacity .2s ease;
    }
    button:hover{ filter:brightness(1.08) }
    button:active{ transform:translateY(1px) }
    button:disabled{ opacity:.55; cursor:not-allowed; filter:none }
    .btn-ghost{
      background:transparent;
      color:var(--text);
      border-color:rgba(255,255,255,0.16);
    }
    a#download{
      display:inline-flex; align-items:center; gap:8px;
      padding:10px 16px; border-radius:12px;
      border:1px solid rgba(255,255,255,0.16);
      text-decoration:none; color:var(--text);
      background:rgba(255,255,255,0.03);
    }
    .muted{color:var(--muted)}

    /* Dropzone */
    .drop{
      border:1.5px dashed rgba(255,255,255,0.18);
      border-radius:14px;
      padding:18px;
      text-align:center;
      cursor:pointer;
      transition:.15s border-color ease, background .15s ease;
      background:rgba(255,255,255,0.02);
    }
    .drop:hover{border-color:rgba(255,255,255,0.35)}
    .drop.active{border-color:var(--accent); background:rgba(99,179,255,.06)}

    .filemeta{font-size:.95rem; color:var(--text); opacity:.9}

    /* Switches (sliders) */
    .switch{position:relative; display:inline-block; width:44px; height:24px}
    .switch input{opacity:0; width:0; height:0}
    .slider{
      position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0;
      background:rgba(255,255,255,0.12); transition:.2s; border-radius:999px; border:1px solid rgba(255,255,255,0.2);
    }
    .slider:before{
      position:absolute; content:""; height:18px; width:18px; left:2px; top:2.5px;
      background:white; transition:.2s; border-radius:50%;
    }
    .switch input:checked + .slider{
      background:linear-gradient(135deg, var(--accent), var(--accent-2));
      border-color:transparent;
    }
    .switch input:checked + .slider:before{ transform:translateX(20px) }

    .progress{height:10px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden; display:none; margin-top:10px}
    .bar{height:100%; width:40%; background:linear-gradient(135deg, var(--accent), var(--accent-2)); border-radius:999px; animation:slide 1.2s infinite}
    @keyframes slide{0%{transform:translateX(-100%)}50%{transform:translateX(50%)}100%{transform:translateX(150%)}}

    .stats{display:flex; gap:12px; align-items:center; margin-top:10px; flex-wrap:wrap}
    .tag{
      display:inline-flex; gap:6px; align-items:center;
      background:rgba(255,255,255,0.05);
      border:1px solid rgba(255,255,255,0.15);
      border-radius:999px; padding:6px 10px; font-size:.9rem
    }
    .tag b{font-weight:800}
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>MarkItDown — Conversion</h1>
      <div class="sub">Plugins seul : MarkItDown → Markdown. Plugins + Forcer OCR (PDF) : texte + OCR des captures (smart). Résumé Azure LLM en option.</div>
    </div>

    <div class="card">
      <div class="row">
        <label for="file">Fichier :</label>
        <input id="file" type="file" />
        <div class="filemeta" id="filemeta"></div>
      </div>

      <div class="drop" id="dropzone" tabindex="0" aria-label="Déposez un fichier ici">
        Glissez-déposez votre fichier ici (ou utilisez le champ ci-dessus)
      </div>

      <div class="row" style="margin-top:12px">
        <label for="plugins">Activer plugins MarkItDown</label>
        <label class="switch">
          <input id="plugins" type="checkbox" />
          <span class="slider"></span>
        </label>

        <label for="llm">Résumé Azure LLM</label>
        <label class="switch">
          <input id="llm" type="checkbox" />
          <span class="slider"></span>
        </label>

        <label for="forceocr">Forcer OCR</label>
        <label class="switch">
          <input id="forceocr" type="checkbox" />
          <span class="slider"></span>
        </label>
      </div>

      <div class="row" style="margin-top:12px; gap:8px; align-items:baseline;">
        <label for="di">Endpoint Azure Document Intelligence</label>
        <input id="di" type="text" placeholder="https://<resource>.cognitiveservices.azure.com/"/>
        <span class="muted">Optionnel (MarkItDown)</span>
      </div>

      <div class="row" style="margin-top:12px; gap:10px">
        <button id="convert">Convertir</button>
        <a id="download" download="sortie.md" style="display:none">Télécharger Markdown</a>
        <button id="copy" class="btn-ghost" title="Copier le Markdown">Copier</button>
        <button id="clear" class="btn-ghost" title="Vider les zones">Vider</button>
      </div>

      <div class="stats">
        <span class="tag">Durée: <b id="timer">0.00 s</b></span>
        <span class="tag">Lignes MD: <b id="linecount">0</b></span>
        <span class="tag">Caractères: <b id="charcount">0</b></span>
        <span id="status" class="muted" style="margin-left:auto"></span>
      </div>

      <div class="progress" id="progress"><div class="bar"></div></div>
    </div>

    <div class="card">
      <label>Markdown</label>
      <textarea id="md" spellcheck="false"></textarea>
    </div>

    <div class="card">
      <label>Métadonnées (JSON)</label>
      <textarea id="meta" style="min-height:160px" spellcheck="false"></textarea>
    </div>
  </div>

<script>
const $ = (id) => document.getElementById(id);
const endpoint = "/convert";

// drag & drop + meta
(function(){
  const dz = $("dropzone"), fi = $("file"), fm = $("filemeta");
  function prettySize(bytes){ if(bytes < 1024) return bytes + " B"; if(bytes < 1048576) return (bytes/1024).toFixed(1) + " KB"; return (bytes/1048576).toFixed(1) + " MB"; }
  function showMeta(f){ fm.textContent = f ? `${f.name} — ${prettySize(f.size)}` : ""; }
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("active"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("active"));
  dz.addEventListener("drop", e => { e.preventDefault(); dz.classList.remove("active"); if(e.dataTransfer.files && e.dataTransfer.files[0]){ fi.files = e.dataTransfer.files; showMeta(fi.files[0]); } });
  fi.addEventListener("change", () => showMeta(fi.files[0]));
})();

// Timer + counters
let timerId = null, t0 = 0;
function startTimer(){ stopTimer(); t0 = performance.now(); timerId = setInterval(() => { const dt = (performance.now() - t0) / 1000; $("timer").textContent = dt < 60 ? dt.toFixed(2) + " s" : (Math.floor(dt/60) + "m " + (dt % 60).toFixed(1) + "s"); }, 100); }
function stopTimer(final = false){ if(timerId){ clearInterval(timerId); timerId = null; } if(final){ const dt = (performance.now() - t0) / 1000; $("timer").textContent = dt < 60 ? dt.toFixed(2) + " s" : (Math.floor(dt/60) + "m " + (dt % 60).toFixed(1) + "s"); } }
function updateCounters(){ const txt = $("md").value || ""; $("charcount").textContent = txt.length.toString(); $("linecount").textContent = (txt ? txt.split(/\r?\n/).length : 0).toString(); }
$("copy").onclick = async () => { 
  try { 
    await navigator.clipboard.writeText($("md").value || ""); 
    $("status").textContent = "Markdown copié"; 
    setTimeout(() => { $("status").textContent = ""; }, 1200); 
  } catch { 
    $("status").textContent = "Impossible de copier."; 
  } 
};

$("clear").onclick = () => { 
  $("md").value = ""; 
  $("meta").value = ""; 
  $("download").style.display = "none"; 
  updateCounters(); 
  $("status").textContent = "Zones effacées."; 
  setTimeout(() => { $("status").textContent = ""; }, 1200); 
};

// Convertir le document
$("convert").onclick = async () => {
  const f = $("file").files[0];
  if(!f){ alert("Choisissez un fichier."); return; }
  $("convert").disabled = true;
  $("status").textContent = "Conversion en cours...";
  $("md").value = "";
  $("meta").value = "";
  $("download").style.display = "none";
  $("progress").style.display = "block";
  startTimer();

  const fd = new FormData();
  fd.append("file", f);
  fd.append("use_plugins", $("plugins").checked ? "true" : "false");
  fd.append("use_llm", $("llm").checked ? "true" : "false");
  fd.append("docintel_endpoint", $("di").value || "");
  fd.append("force_ocr", $("forceocr").checked ? "true" : "false");

  try {
    const res = await fetch(endpoint, { method: "POST", body: fd });
    if(!res.ok){ throw new Error("HTTP " + res.status); }
    const json = await res.json();
    $("md").value = json.markdown || "";
    $("meta").value = JSON.stringify(json.metadata || {}, null, 2);
    updateCounters();

    const blob = new Blob([$("md").value], { type: "text/markdown;charset=utf-8" });
    const url  = URL.createObjectURL(blob);
    const a = $("download");
    a.href = url;
    a.download = (json.output_filename || "sortie.md");
    a.style.display = "inline-flex";
    $("status").textContent = "OK";
  } catch(e) {
    $("status").textContent = "Erreur : " + (e && e.message ? e.message : e);
  } finally {
    $("convert").disabled = false;
    $("progress").style.display = "none";
    stopTimer(true);
  }
};

$("md").addEventListener("input", updateCounters);
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
    - Plugins OFF : MarkItDown simple (+ cleanup).
    - Plugins ON (PDF + OCR forcé) : PyMuPDF inline → texte vectoriel + OCR “smart” des captures, fallback page si pas de texte, Markdown nettoyé.
    - Image seule : OCR + base64 si OCR pauvre.
    """
    try:
        t_start = time.perf_counter()

        if not docintel_endpoint:
            docintel_endpoint = DEFAULT_DOCINTEL_ENDPOINT

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Fichier vide")

        # Sauvegarde du fichier source si demandé
        in_path = None
        if SAVE_UPLOADS:
            in_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(in_path, "wb") as f:
                f.write(content)

        is_pdf = guess_is_pdf(file.filename, file.content_type)
        is_img = guess_is_image(file.filename, file.content_type)

        metadata: Dict[str, Any] = {}

        if is_pdf and use_plugins and OCR_ENABLED and force_ocr:
            # Pipeline PDF inline : texte + images (OCR “smart”) in‑place
            markdown, meta_pdf = render_pdf_markdown_inline(content)
            metadata.update(meta_pdf)
            # Nettoyage du Markdown final
            markdown = _md_cleanup(markdown)

        else:
            # Conversion MarkItDown générique
            md_engine = MarkItDown(enable_plugins=use_plugins, docintel_endpoint=docintel_endpoint)
            result = md_engine.convert_stream(io.BytesIO(content), file_name=file.filename)

            markdown = getattr(result, "text_content", "") or ""
            metadata.update(getattr(result, "metadata", {}) or {})
            warnings = getattr(result, "warnings", None)
            if warnings:
                metadata["warnings"] = warnings

            # Post-traitement Markdown
            markdown = _md_cleanup(markdown)

            # Image isolée + OCR local
            if OCR_ENABLED and is_img:
                ocr_text, score = ocr_image_bytes(content, OCR_LANGS)
                if ocr_text.strip():
                    markdown += "\n\n# OCR (extrait)\n" + ocr_text
                if EMBED_IMAGES in ("all", "ocr_only") and (not ocr_text or score < OCR_TEXT_QUALITY_MIN):
                    try:
                        with Image.open(io.BytesIO(content)) as im:
                            im = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            markdown += f'\n\n![{IMG_ALT_PREFIX}]({data_uri})\n'
                    except Exception:
                        pass

        # Sauvegarde du résultat Markdown si demandé
        out_name = f"{os.path.splitext(file.filename)[0]}.md"
        out_path = None
        if SAVE_OUTPUTS:
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(markdown)

        # Résumé Azure OpenAI (optionnel)
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

        # Durée de traitement côté serveur
        metadata["duration_sec"] = round(time.perf_counter() - t_start, 3)

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
