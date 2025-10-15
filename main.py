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

from markitdown import MarkItDown

# OCR & Images
import numpy as np
from paddleocr import PaddleOCR
from paddleocr import PPStructureV3 as PPStructure
from PIL import Image, ImageOps, ImageFilter
import fitz  # PyMuPDF
from html.parser import HTMLParser
import html

# ---------------------------
# Config via variables d'env
# ---------------------------
SAVE_UPLOADS  = os.getenv("SAVE_UPLOADS", "false").lower() == "true"
SAVE_OUTPUTS  = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"
UPLOAD_DIR    = os.getenv("UPLOAD_DIR", "/data/uploads")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "/data/outputs")

# Modes de traitement: "fast" | "quality"
PROCESSING_MODE = os.getenv("PROCESSING_MODE", "quality").strip().lower()
ALLOWED_MODES = {"fast", "quality"}

# OCR (PaddleOCR)
OCR_ENABLED        = os.getenv("OCR_ENABLED", "true").lower() == "true"
_OCR_LANG_RAW      = os.getenv("OCR_LANGS", "fra+eng").strip().lower()

# PDF raster
OCR_DPI            = int(os.getenv("OCR_DPI", "350"))
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "50"))
OCR_DPI_CANDIDATES = [int(x) for x in os.getenv("OCR_DPI_CANDIDATES", "300,350,400").split(",")]
OCR_SCORE_GOOD_ENOUGH = float(os.getenv("OCR_SCORE_GOOD_ENOUGH", "0.6"))

# Politique OCR images & fallback
# En “plugins ON”, on lance l’OCR image systématiquement (comme Marker).
IMAGE_OCR_MODE = os.getenv("IMAGE_OCR_MODE", "smart").strip()  # smart | conservative | always | never
IMAGE_OCR_MIN_WORDS = int(os.getenv("IMAGE_OCR_MIN_WORDS", "10"))
OCR_TEXT_QUALITY_MIN = float(os.getenv("OCR_TEXT_QUALITY_MIN", "0.25"))
OCR_MIN_ACCEPT_LEN = int(os.getenv("OCR_MIN_ACCEPT_LEN", "10"))
OCR_DISABLE_PAGE_FALLBACK = os.getenv("OCR_DISABLE_PAGE_FALLBACK", "false").lower() == "true"

# Intégration images base64
EMBED_IMAGES       = os.getenv("EMBED_IMAGES", "ocr_only").strip()  # none | ocr_only | all
IMG_FORMAT         = os.getenv("IMG_FORMAT", "png").strip().lower()
IMG_JPEG_QUALITY   = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH      = int(os.getenv("IMG_MAX_WIDTH", "1600"))
IMG_ALT_PREFIX     = os.getenv("IMG_ALT_PREFIX", "Capture").strip()
EMBED_SKIP_BG_IF_TEXT = os.getenv("EMBED_SKIP_BG_IF_TEXT", "true").lower() == "true"

# Dossiers persistents
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="4.9-ppstructure-tables+paddleocr-inline")

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
def guess_is_pdf(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower() in ("application/pdf", "pdf"):
        return True
    return filename.lower().endswith(".pdf")

def guess_is_image(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    return any(filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))


def guess_is_html(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower() in ("text/html", "application/xhtml+xml", "html"):
        return True
    return filename.lower().endswith((".html", ".htm"))

def _md_cleanup(md: str) -> str:
    if not md:
        return md
    lines = []
    for L in md.replace("\r", "").split("\n"):
        l = re.sub(r"[ \t]+$", "", L)
        l = re.sub(r"^\s*[•·●◦▪]\s+", "- ", l)
        l = re.sub(r"^\s*(\d+)[\)\]]\s+", r"\1. ", l)
        lines.append(l)
    txt = "\n".join(lines)
    txt = re.sub(
        r"(?:^|\n)((?:[|+\-=_].*\n){2,})",
        lambda m: "```text\n" + m.group(1).strip() + "\n```",
        txt, flags=re.S
    )
    return txt.strip()

# ---------------------------
# OCR utils (PaddleOCR + PPStructure)
# ---------------------------
_table_chars = re.compile(r"[|+\-=_]{3,}")

def _preprocess_for_ocr(im: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = im.filter(ImageFilter.MedianFilter(size=3))
    return g

def _classify_ocr_block(txt: str) -> str:
    t = (txt or "").strip()
    if not t:
        return ""
    if re.search(r"^\s*\+.*[-+].*\+\s*$", t, flags=re.M) or re.search(r"^\s*\|.*\|\s*$", t, flags=re.M):
        return f"```text\n{t}\n```"
    if re.search(r"^\s*(\$|#)\s", t, flags=re.M) or "mysql>" in t:
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|SHOW|GRANT|CHANGE\s+MASTER|START\s+(SLAVE|REPLICA))\b", t, flags=re.I):
            return f"```sql\n{t}\n```"
        return f"```bash\n{t}\n```"
    return t

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

def _parse_langs(raw: str) -> List[str]:
    parts = [p for p in re.split(r"[+,/|\s]+", raw) if p]
    out: List[str] = []
    for p in parts:
        p = p.lower()
        if p in ("fra", "fr"): out.append("fr")
        elif p in ("eng", "en"): out.append("en")
        elif p in ("deu", "ger", "german"): out.append("german")
        elif p in ("spa", "es", "spanish"): out.append("latin")
        else: out.append(p)
    seen = set(); unique = []
    for l in out:
        if l not in seen:
            unique.append(l); seen.add(l)
    if "latin" not in seen:
        unique.append("latin")
    return unique

OCR_LANGS_LIST = _parse_langs(_OCR_LANG_RAW)

def _paddle_key(mode: str, lang: str) -> str:
    return f"{mode}:{lang}"

def _build_paddle(mode: str, lang: str) -> PaddleOCR:
    return PaddleOCR(use_angle_cls=(mode == "quality"), lang=lang, show_log=False)

_PADDLE_INSTANCES: Dict[str, PaddleOCR] = {}
_PPSTRUCT_INSTANCES: Dict[str, PPStructure] = {}

def _get_ocr(mode: str, lang: str) -> PaddleOCR:
    key = _paddle_key(mode, lang)
    if key not in _PADDLE_INSTANCES:
        _PADDLE_INSTANCES[key] = _build_paddle(mode, lang)
    return _PADDLE_INSTANCES[key]

def _get_table_engine(mode: str) -> PPStructure:
    # PPStructure: on reste sur 'en' (latin) pour la structure/table.
    key = f"ppstruct:{mode}"
    if key not in _PPSTRUCT_INSTANCES:
        _PPSTRUCT_INSTANCES[key] = PPStructure(show_log=False, lang='en')
    return _PPSTRUCT_INSTANCES[key]

def _paddle_ocr_text_best(im: Image.Image, mode: str, langs: Optional[List[str]] = None) -> Tuple[str, float, str]:
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if mode == "quality":
        im = _preprocess_for_ocr(im)
    img_nd = np.array(im)

    candidates = langs or OCR_LANGS_LIST
    best_txt, best_score, best_lang = "", -1e9, (candidates[0] if candidates else "en")

    for lang in candidates:
        try:
            ocr = _get_ocr(mode, lang)
            res = ocr.ocr(img_nd, cls=(mode == "quality"))
            lines = []
            if res and isinstance(res, list):
                for page in res:
                    if not page:
                        continue
                    for det in page:
                        try:
                            txt = det[1][0]
                            if txt:
                                lines.append(txt)
                        except Exception:
                            continue
            text = "\n".join(lines).strip()
            score = _ocr_quality_score(text)
            if score > best_score:
                best_txt, best_score, best_lang = text, score, lang
            if best_score >= OCR_SCORE_GOOD_ENOUGH:
                break
        except Exception:
            continue

    return best_txt, best_score, best_lang


def _ppstruct_tables_to_md(img: Image.Image) -> Optional[str]:
    """Use PPStructure to detect tables and return Markdown-ish representation.
    Returns None if no table is detected or engine unavailable.
    """
    try:
        engine = _get_table_engine("quality")
        if not engine:
            return None
        nd = np.array(img.convert("RGB"))
        result = engine(nd)
        tables_md: List[str] = []
        for item in result:
            if item.get("type") != "table":
                continue
            res = item.get("res", {}) or {}
            html_str = res.get("html", "") or ""
            if not html_str:
                continue
            # Fallback: keep HTML table as-is; consumer may render it, or you can improve with HTML→MD later.
            tables_md.append(html_str.strip())
        return "\n\n".join(tables_md) if tables_md else None
    except Exception:
        return None

class _ImgStripper(HTMLParser):
    """Replace <img> with tokens to preserve position and record their attributes."""
    def __init__(self):
        super().__init__()
        self.out = []
        self.images = []  # list of dicts: {src, alt, width, height}
    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "img":
            d = {k.lower(): (v or "") for k, v in attrs}
            self.images.append({
                "src": d.get("src", ""),
                "alt": (d.get("alt", "") or "").strip(),
                "width": d.get("width", ""),
                "height": d.get("height", ""),
            })
            self.out.append(f"<<<MDIMG_{len(self.images)-1}>>>")
        else:
            self.out.append("<" + t)
            for k, v in attrs:
                if v is None:
                    v = ""
                self.out.append(f' {k}="{html.escape(v, quote=True)}"')
            self.out.append(">")
    def handle_startendtag(self, tag, attrs):
        t = tag.lower()
        if t == "img":
            d = {k.lower(): (v or "") for k, v in attrs}
            self.images.append({
                "src": d.get("src", ""),
                "alt": (d.get("alt", "") or "").strip(),
                "width": d.get("width", ""),
                "height": d.get("height", ""),
            })
            self.out.append(f"<<<MDIMG_{len(self.images)-1}>>>")
        else:
            self.out.append("<" + t)
            for k, v in attrs:
                if v is None:
                    v = ""
                self.out.append(f' {k}="{html.escape(v, quote=True)}"')
            self.out.append(" />")
    def handle_endtag(self, tag):
        if tag.lower() != "img":
            self.out.append(f"</{tag.lower()}>")
    def handle_data(self, data):
        self.out.append(data)
    def get_html(self):
        return "".join(self.out)

def _data_uri_to_bytes(data_uri: str):
    try:
        if not data_uri.startswith("data:"):
            return None, None
        header, b64 = data_uri.split(",", 1)
        mime = "image/png"
        if ";base64" in header:
            if ":" in header and ";" in header:
                mime = header.split(":")[1].split(";")[0] or mime
            raw = base64.b64decode(b64)
            return raw, mime
        return None, None
    except Exception:
        return None, None

def _convert_html_with_inline_images(
    html_bytes: bytes,
    use_plugins: bool,
    img_fmt: str,
    img_quality: int,
    img_max_w: int,
    alt_prefix: str
) -> Tuple[str, Dict[str, Any]]:
    # 1) Decode bytes
    try:
        html_txt = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        html_txt = html_bytes.decode("latin-1", errors="ignore")

    # 2) Replace <img> by tokens
    stripper = _ImgStripper()
    stripper.feed(html_txt)
    stripped_html = stripper.get_html()
    imgs = stripper.images

    # 3) Convert stripped HTML via MarkItDown
    md_engine = MarkItDown(enable_plugins=use_plugins)
    result = md_engine.convert_stream(io.BytesIO(stripped_html.encode('utf-8')), file_name='input.html')
    md = getattr(result, 'text_content', '') or ''
    meta: Dict[str, Any] = getattr(result, "metadata", {}) or {}
    warns = getattr(result, "warnings", None)
    if warns:
        meta["warnings"] = warns

    # 4) Re-inject images
    from PIL import Image
    embedded = 0
    linked = 0
    tokens: Dict[str, str] = {}
    for i, info in enumerate(imgs):
        src = (info.get("src") or "").strip()
        alt = (info.get("alt") or "").strip() or alt_prefix
        token = f"<<<MDIMG_{i}>>>"
        md_img = ""
        if src.startswith("data:"):
            raw, _mime = _data_uri_to_bytes(src)
            if raw:
                try:
                    with Image.open(io.BytesIO(raw)) as im:
                        if img_max_w and im.width > img_max_w:
                            r = img_max_w / im.width
                            im = im.resize((img_max_w, int(im.height * r)))
                        buf = io.BytesIO()
                        if img_fmt.lower() in ("jpeg", "jpg"):
                            im = im.convert("RGB")
                            im.save(buf, format="JPEG", quality=img_quality, optimize=True)
                            mime2 = "image/jpeg"
                        else:
                            im.save(buf, format="PNG", optimize=True)
                            mime2 = "image/png"
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        data_uri = f"data:{mime2};base64,{b64}"
                        md_img = f"![{alt}]({data_uri})"
                        embedded += 1
                except Exception:
                    linked += 1
            else:
                linked += 1
        else:
            # Do not fetch external URLs here; leave as linked image.
            md_img = f"![{alt}]({src})"
            linked += 1
        tokens[token] = md_img

    
    # If MarkItDown returned empty content (e.g., image-only HTML), fall back to listing images.
    if not md.strip() and tokens:
        md = "\n\n".join([v for k, v in tokens.items() if v])

    for token, repl in tokens.items():
        md = md.replace(token, repl)

    md = _md_cleanup(md)
    meta.setdefault("html_image_stats", {})
    meta["html_image_stats"].update({
        "found": len(imgs),
        "embedded_base64": embedded,
        "linked_only": linked
    })
    meta["engine_html"] = f"markitdown+inline_images({'plugins' if use_plugins else 'core'})"
    return md, meta

class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_tr = False
        self.in_cell = False
        self.headers: List[str] = []
        self.rows: List[List[str]] = []
        self.current_row: List[str] = []
        self.buf: List[str] = []
        self.is_th = False

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "table":
            self.in_table = True
        elif t == "tr" and self.in_table:
            self.in_tr = True
            self.current_row = []
        elif t in ("td", "th") and self.in_tr:
            self.in_cell = True
            self.buf = []
            self.is_th = (t == "th")

    def handle_data(self, data):
        if self.in_cell:
            self.buf.append(data)

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in ("td", "th") and self.in_cell:
            txt = html.unescape("".join(self.buf)).strip()
            txt = re.sub(r"\s*\|\s*", " ", txt)
            self.current_row.append(txt)
            self.in_cell = False
            self.is_th = False
        elif t == "tr" and self.in_tr:
            if self.current_row:
                # s'il y a une ligne d'en-têtes évidente (th), PP-Structure l'a déjà mise au début
                self.rows.append(self.current_row)
            self.in_tr = False
        elif t == "table":
            self.in_table = False


def _html_table_to_markdown(html_str: str) -> str:
    """Very simple fallback: return the HTML table unchanged.
    You can improve by parsing to Markdown if needed."""
    return html_str or ""

def _data_uri_to_bytes(data_uri: str) -> tuple[bytes, str] | tuple[None, None]:
    try:
        if not data_uri.startswith("data:"):
            return None, None
        header, b64 = data_uri.split(",", 1)
        mime = "image/png"
        if ";base64" in header:
            # extract mime like data:image/png;base64
            if ":" in header and ";" in header:
                mime = header.split(":")[1].split(";")[0] or mime
            raw = base64.b64decode(b64)
            return raw, mime
        return None, None
    except Exception:
        return None, None

def _convert_html_with_inline_images(html_bytes: bytes, use_plugins: bool, img_fmt: str, img_quality: int, img_max_w: int, alt_prefix: str) -> tuple[str, dict]:
    # Decode
    try:
        html_txt = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        html_txt = html_bytes.decode("latin-1", errors="ignore")

    # Replace <img> with tokens
    stripper = _ImgStripper()
    stripper.feed(html_txt)
    stripped_html = stripper.get_html()
    imgs = stripper.images

    # Convert text via MarkItDown (on sanitized HTML without <img> tags)
    md_engine = MarkItDown(enable_plugins=use_plugins)
    result = md_engine.convert(stripped_html, "text/html")
    md = getattr(result, "text_content", "") or ""
    meta = getattr(result, "metadata", {}) or {}
    warns = getattr(result, "warnings", None)
    if warns:
        meta["warnings"] = warns

    # Build md for each image token
    embedded = 0
    skipped = 0
    processed = {}
    for i, info in enumerate(imgs):
        src = info.get("src","").strip()
        alt = info.get("alt","").strip() or alt_prefix
        token = f"<<<MDIMG_{i}>>>"
        md_img = ""
        if src.startswith("data:"):
            raw, mime = _data_uri_to_bytes(src)
            if raw:
                try:
                    from PIL import Image, ImageOps, ImageFilter  # ensure available
                    import io as _io
                    with Image.open(_io.BytesIO(raw)) as im:
                        # resize, re-encode to configured format
                        if img_max_w and im.width > img_max_w:
                            ratio = img_max_w / im.width
                            im = im.resize((img_max_w, int(im.height*ratio)))
                        # re-encode
                        buf = _io.BytesIO()
                        if img_fmt.lower() in ("jpeg","jpg"):
                            im = im.convert("RGB")
                            im.save(buf, format="JPEG", quality=img_quality, optimize=True)
                            mime2 = "image/jpeg"
                        else:
                            im.save(buf, format="PNG", optimize=True)
                            mime2 = "image/png"
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        data_uri = f"data:{mime2};base64,{b64}"
                        md_img = f'![{alt}]({data_uri})'
                        embedded += 1
                except Exception:
                    skipped += 1
            else:
                skipped += 1
        else:
            # For non-data URIs, leave a normal markdown pointing to original src (no fetch to avoid side-effects)
            md_img = f'![{alt}]({src})'
            skipped += 1  # not embedded as base64

        processed[token] = md_img or ""

    # Replace tokens in markdown, preserving positions
    for token, repl in processed.items():
        md = md.replace(token, repl)

    # Cleanup
    md = _md_cleanup(md)

    meta.setdefault("html_image_stats", {})
    meta["html_image_stats"].update({
        "found": len(imgs),
        "embedded_base64": embedded,
        "linked_only": skipped
    })
    meta["engine_html"] = f"markitdown+inline_images({ 'plugins' if use_plugins else 'core'})"

    return md, meta

def _fast_has_text_with_paddle(im: Image.Image, mode: str, min_words: int) -> bool:
    try:
        txt, _, _ = _paddle_ocr_text_best(im, mode)
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{2,}", txt)
        return len(words) >= min_words
    except Exception:
        return False

def ocr_image_bytes(img_bytes: bytes, mode: str) -> Tuple[str, float, str]:
    with Image.open(io.BytesIO(img_bytes)) as im:
        # D’abord tenter une table via PPStructure
        md_table = _ppstruct_tables_to_md(im)
        if md_table:
            return md_table, 1.0, "en(struct)"  # on considère “qualité OK”
        txt, score, used_lang = _paddle_ocr_text_best(im, mode)
        return _classify_ocr_block(txt), score, used_lang

def _raster_pdf_page(page: fitz.Page, dpi: int) -> Image.Image:
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

def _hash_image_for_dedup(im: Image.Image) -> str:
    small_w = min(256, im.width)
    small_h = max(1, int(im.height * small_w / max(1, im.width)))
    small = im.resize((small_w, small_h), Image.BILINEAR)
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

# ---------------------------
# PDF inline : texte + images (OCR + PPStructure) + fallback
# ---------------------------
def _is_bold(flags: int) -> bool:
    return bool(flags & 1 or flags & 32)

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

def _classify_heading(size: float, median_size: float, has_bold: bool) -> Optional[str]:
    if median_size <= 0:
        return None
    if size >= median_size * 1.8: return "#"
    if size >= median_size * 1.5: return "##"
    if has_bold and size >= median_size * 1.1: return "###"
    if size >= median_size * 1.25: return "###"
    return None

_bullet_re = re.compile(r"^\s*(?:[-–—•·●◦▪]|\d+[.)])\s+")

def _line_to_md(spans: List[Dict[str, Any]], median_size: float) -> str:
    parts = []
    max_size = 0.0
    any_bold = False
    for sp in spans:
        t = sp.get("text", "")
        if not t:
            continue
        size = float(sp.get("size", 0))
        max_size = max(max_size, size)
        bold = _is_bold(int(sp.get("flags", 0)))
        any_bold = any_bold or bold
        parts.append(f"**{t}**" if bold else t)
    raw = "".join(parts).strip()
    if not raw:
        return ""
    h = _classify_heading(max_size, median_size, any_bold)
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

def render_pdf_markdown_inline(
    pdf_bytes: bytes,
    mode: str,
    meta_out: Dict[str, Any],
    force_ocr_images: bool = False
) -> Tuple[str, Dict[str, Any]]:
    if mode == "fast":
        dpi_page = min(OCR_DPI, 300)
        dpi_candidates = [min(x, 300) for x in OCR_DPI_CANDIDATES[:2]]
    else:
        dpi_page = OCR_DPI
        dpi_candidates = OCR_DPI_CANDIDATES

    # En plugins ON, on OCR TOUJOURS les images
    img_ocr_policy = "always" if IMAGE_OCR_MODE != "never" else "never"

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_lines: List[str] = []
    meta: Dict[str, Any] = {"engine": f"pymupdf_inline+paddleocr+ppstructure({mode})", "pages": doc.page_count}
    try:
        total_pages = min(doc.page_count, OCR_MAX_PAGES)
        for p in range(total_pages):
            page = doc.load_page(p)
            raw = page.get_text("dict") or {}

            median_size = _median_font_size(raw)
            line_h_med = 12.0 if median_size == 0 else median_size * 1.0
            band_h = max(8.0, line_h_med * 1.2)

            atoms = []
            page_w = page.rect.width
            page_h = page.rect.height
            page_area = max(1.0, page_w * page_h)

            # 1) TEXTE (lignes) + marquage images
            for b in raw.get("blocks", []):
                btype = b.get("type", 0)
                bbox = tuple(b.get("bbox", (0, 0, 0, 0)))
                x0, y0, x1, y1 = bbox
                x0 = max(0.0, min(x0, page_w)); x1 = max(0.0, min(x1, page_w))
                y0 = max(0.0, min(y0, page_h)); y1 = max(0.0, min(y1, page_h))
                bbox = (x0, y0, x1, y1)

                if btype == 0:
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
                            md_line = "".join(s.get("text", "") for s in spans).strip()
                        if not md_line:
                            continue

                        atoms.append({
                            "bbox": line_bbox,
                            "md": md_line,
                            "kind": "text",
                            "text_len": len(md_line),
                            "area_ratio": ((lx1 - lx0) * (ly1 - ly0)) / page_area
                        })

                elif btype == 1:
                    atoms.append({
                        "bbox": bbox,
                        "md": None,
                        "kind": "image_raw",
                        "text_len": 0,
                        "area_ratio": ((x1 - x0) * (y1 - y0)) / page_area
                    })

            # 2) IMAGES -> PPStructure (tables) puis OCR texte, sinon embed
            processed = []
            seen_hashes: set = set()
            page_has_vector_text = any(a["kind"] == "text" for a in atoms)

            for a in atoms:
                if a["kind"] != "image_raw":
                    processed.append(a)
                    continue

                im = crop_bbox_image(page, a["bbox"], dpi_page)
                if im is None:
                    continue

                h = _hash_image_for_dedup(im)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                area_ratio = a["area_ratio"]
                is_background_like = area_ratio > 0.85

                # Politique d'OCR (always en plugins ON)
                if img_ocr_policy == "never":
                    do_img_ocr = False
                else:
                    do_img_ocr = True

                md_img = ""
                txt, q, used_lang = "", 0.0, None

                if do_img_ocr and OCR_ENABLED:
                    # 2.1 – d’abord PPStructure (tables)
                    md_table = _ppstruct_tables_to_md(im)
                    if md_table:
                        md_img = md_table
                        txt = md_table
                        q = 1.0
                        used_lang = "en(struct)"
                    else:
                        # 2.2 – texte classique
                        try:
                            txt, q, used_lang = _paddle_ocr_text_best(im, mode)
                        except Exception as e:
                            meta_out.setdefault("ocr_errors", []).append(
                                f"img_page_{p+1}: {type(e).__name__}: {e}"
                            )

                accept_low_quality = (force_ocr_images and txt and len(txt) >= OCR_MIN_ACCEPT_LEN)

                if (txt and q >= OCR_TEXT_QUALITY_MIN) or accept_low_quality:
                    md_img = _classify_ocr_block(txt.strip())
                    a_kind = "image_text"
                else:
                    if EMBED_IMAGES in ("all", "ocr_only"):
                        if not (EMBED_SKIP_BG_IF_TEXT and is_background_like and page_has_vector_text):
                            im = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            md_img = f'![{IMG_ALT_PREFIX} – page {p+1}]({data_uri})'
                            a_kind = "image_embed"
                        else:
                            md_img = ""
                            a_kind = "image_skipped_bg"
                    else:
                        md_img = ""
                        a_kind = "image_skipped"

                if used_lang:
                    meta_out.setdefault("ocr_used_langs", set()).add(used_lang)

                if md_img:
                    processed.append({
                        "bbox": a["bbox"],
                        "md": md_img,
                        "kind": "image",
                        "text_len": len(txt or ""),
                        "area_ratio": area_ratio
                    })
                meta_out.setdefault("image_stats", {"ocr_text":0,"embedded":0,"skipped":0})
                if a_kind == "image_text":
                    meta_out["image_stats"]["ocr_text"] += 1
                elif a_kind == "image_embed":
                    meta_out["image_stats"]["embedded"] += 1
                else:
                    meta_out["image_stats"]["skipped"] += 1

            atoms = processed

            # 3) Tri de lecture (haut→bas, gauche→droite)
            def sort_key(a):
                x0, y0, x1, y1 = a["bbox"]
                y_center = 0.5 * (y0 + y1)
                return (int(y_center / band_h), x0, y0)
            atoms.sort(key=sort_key)

            # 4) Concaténation lignes + wrap tableaux ascii
            page_buf: List[str] = []
            para_buf: List[str] = []
            last_band = None

            def flush_para():
                if not para_buf:
                    return
                block_txt = "\n".join(para_buf).strip()
                if block_txt:
                    if _table_chars.search(block_txt):
                        page_buf.append("```text")
                        page_buf.append(block_txt)
                        page_buf.append("```")
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

            # 5) Fallback OCR page si rien d’extrait
            has_any_text = any(a.get("kind") == "text" for a in atoms) or \
                           any(a.get("kind") == "image" and a.get("text_len",0) > 0 for a in atoms)
            if not has_any_text and not OCR_DISABLE_PAGE_FALLBACK and OCR_ENABLED:
                try:
                    best_txt, best_score, used_lang = "", -1e9, None
                    for d in dpi_candidates:
                        im_page = _raster_pdf_page(page, d)
                        txt, score, lang = _paddle_ocr_text_best(im_page, mode)
                        if lang:
                            meta_out.setdefault("ocr_used_langs", set()).add(lang)
                        if score > best_score:
                            best_txt, best_score = txt, score
                        if best_score >= OCR_SCORE_GOOD_ENOUGH:
                            break
                    if best_txt.strip():
                        page_buf.append("<!-- OCR page fallback -->")
                        page_buf.append(_wrap_tables_as_code(best_txt.strip()))
                        meta_out.setdefault("page_fallback_ocr", 0)
                        meta_out["page_fallback_ocr"] += 1
                except Exception as e:
                    meta_out.setdefault("ocr_errors", []).append(
                        f"page_fallback_{p+1}: {type(e).__name__}: {e}"
                    )

            if page_buf:
                md_lines.append("\n\n".join(page_buf))

        # Suppression des en-têtes/pieds de page répétés
        if len(md_lines) > 1:
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
            for i, content in enumerate(md_lines):
                lines = content.splitlines()
                if lines and lines[0].strip() and lines[0] in headers_to_remove:
                    lines = lines[1:]
                if lines and lines[-1].strip() and lines[-1] in footers_to_remove:
                    lines = lines[:-1]
                md_lines[i] = "\n".join(lines).strip()

        final_md = "\n\n".join([l for l in md_lines if l.strip()]).strip()

        if "ocr_used_langs" in meta_out and isinstance(meta_out["ocr_used_langs"], set):
            meta_out["ocr_used_langs"] = sorted(list(meta_out["ocr_used_langs"]))

        return (final_md, meta)
    finally:
        doc.close()

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
      width:100%;
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
      <div class="sub">Conversion Markdown + OCR et Base64</div>
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

        <label for="forceocr">Forcer OCR</label>
        <label class="switch">
          <input id="forceocr" type="checkbox" />
          <span class="slider"></span>
        </label>
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
  try { await navigator.clipboard.writeText($("md").value || ""); $("status").textContent = "Markdown copié"; setTimeout(() => { $("status").textContent = ""; }, 1200); }
  catch { $("status").textContent = "Impossible de copier."; }
};
$("clear").onclick = () => { $("md").value = ""; $("meta").value = ""; $("download").style.display = "none"; updateCounters(); $("status").textContent = "Zones effacées."; setTimeout(() => { $("status").textContent = ""; }, 1200); };

// Convertir
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

# ---------------------------
# Endpoint API de conversion
# ---------------------------
@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    use_plugins: bool = Form(False),
    force_ocr: bool = Form(False),
    mode: Optional[str] = Form(None),  # "fast" | "quality"
):
    """
    - Plugins OFF : MarkItDown simple (+ cleanup).
    - Plugins ON (PDF) : pipeline inline (texte vectoriel + OCR images + PPStructure), avec 'force_ocr' qui assouplit l'acceptation.
    - Image seule : OCR/PPStructure + base64 si OCR pauvre.
    """
    try:
        t_start = time.perf_counter()

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Fichier vide")

        in_path = None
        if SAVE_UPLOADS:
            in_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(in_path, "wb") as f:
                f.write(content)

        is_pdf = guess_is_pdf(file.filename, file.content_type)
        is_img = guess_is_image(file.filename, file.content_type)
        is_html = guess_is_html(file.filename, file.content_type)

        selected_mode = (mode or PROCESSING_MODE).lower()
        if selected_mode not in ALLOWED_MODES:
            selected_mode = "quality"

        metadata: Dict[str, Any] = {"processing_mode": selected_mode}

        if is_pdf and use_plugins and OCR_ENABLED:
            ocr_meta: Dict[str, Any] = {}
            markdown, meta_pdf = render_pdf_markdown_inline(
                content,
                selected_mode,
                ocr_meta,
                force_ocr_images=force_ocr
            )
            metadata.update(meta_pdf)
            if ocr_meta:
                if isinstance(ocr_meta.get("ocr_used_langs"), set):
                    ocr_meta["ocr_used_langs"] = sorted(list(ocr_meta["ocr_used_langs"]))
                metadata.update(ocr_meta)
            markdown = _md_cleanup(markdown)

        elif is_html:
                # Convert HTML with inline <img> embedded as base64 (like PDF policy)
                markdown, meta_html = _convert_html_with_inline_images(
                    content,
                    use_plugins=use_plugins,
                    img_fmt=IMG_FORMAT,
                    img_quality=IMG_JPEG_QUALITY,
                    img_max_w=IMG_MAX_WIDTH,
                    alt_prefix=IMG_ALT_PREFIX
                )
                metadata.update(meta_html)

        else:
            md_engine = MarkItDown(enable_plugins=use_plugins)
            result = md_engine.convert_stream(io.BytesIO(content), file_name=file.filename)
            markdown = getattr(result, "text_content", "") or ""
            metadata.update(getattr(result, "metadata", {}) or {})
            warnings = getattr(result, "warnings", None)
            if warnings:
                metadata["warnings"] = warnings
            markdown = _md_cleanup(markdown)

            if OCR_ENABLED and is_img:
                try:
                    ocr_text, score, used_lang = ocr_image_bytes(content, selected_mode)
                    if used_lang:
                        metadata.setdefault("ocr_used_langs", [])
                        if used_lang not in metadata["ocr_used_langs"]:
                            metadata["ocr_used_langs"].append(used_lang)
                    if ocr_text.strip():
                        markdown += "\n\n# OCR (extrait)\n" + ocr_text
                    if EMBED_IMAGES in ("all", "ocr_only") and (not ocr_text or score < OCR_TEXT_QUALITY_MIN):
                        with Image.open(io.BytesIO(content)) as im:
                            im = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            markdown += f'\n\n![{IMG_ALT_PREFIX}]({data_uri})\n'
                except Exception as e:
                    metadata.setdefault("ocr_errors", []).append(f"image: {type(e).__name__}: {e}")
                    if EMBED_IMAGES in ("all", "ocr_only"):
                        try:
                            with Image.open(io.BytesIO(content)) as im:
                                im = _pil_resize_max(im, IMG_MAX_WIDTH)
                                data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                                markdown += f'\n\n![{IMG_ALT_PREFIX}]({data_uri})\n'
                        except Exception:
                            pass

        out_name = f"{os.path.splitext(file.filename)[0]}.md"
        out_path = None
        if SAVE_OUTPUTS:
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(markdown)

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
        detail = f"Conversion error: {type(e).__name__}: {e}"
        trace = traceback.format_exc(limit=3)
        raise HTTPException(status_code=500, detail=f"{detail}\n{trace}")

# ---------------------------
# Healthcheck
# ---------------------------
@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"
