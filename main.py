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

# === NEW: data utils
import pandas as pd
try:
    import cv2  # OpenCV (headless)
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

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
OCR_MODE           = os.getenv("OCR_MODE", "append").strip()
OCR_KEEP_SPACES    = os.getenv("OCR_KEEP_SPACES", "true").lower() == "true"
OCR_TWO_PASS       = os.getenv("OCR_TWO_PASS", "true").lower() == "true"
OCR_TABLE_MODE     = os.getenv("OCR_TABLE_MODE", "true").lower() == "true"
OCR_PSMS           = [p.strip() for p in os.getenv("OCR_PSMS", "6,3,4,11").split(",")]  # === NEW add psm 3
OCR_DPI_CANDIDATES = [int(x) for x in os.getenv("OCR_DPI_CANDIDATES", "300,350,400").split(",")]
OCR_SCORE_GOOD_ENOUGH = float(os.getenv("OCR_SCORE_GOOD_ENOUGH", "0.6"))
# === NEW: deskew + binarisation
OCR_DESKEW         = os.getenv("OCR_DESKEW", "true").lower() == "true"
OCR_BINARIZE       = os.getenv("OCR_BINARIZE", "true").lower() == "true"

# Images base64
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
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="3.0")  # bump version

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

def guess_is_docx(filename: str) -> bool:
    return filename.lower().endswith(".docx")

def guess_is_xlsx(filename: str) -> bool:
    return filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls")

def guess_is_csv(filename: str) -> bool:
    return filename.lower().endswith(".csv")

def _md_cleanup(md: str) -> str:
    """Post-format léger pour la sortie MarkItDown: listes/titres/tableaux ASCII."""
    if not md:
        return md
    lines = []
    for L in md.replace("\r","").split("\n"):
        l = re.sub(r"[ \t]+$", "", L)
        l = re.sub(r"^\s*[•·●◦▪]\s+", "- ", l)         # bullets unicode -> '- '
        l = re.sub(r"^\s*(\d+)[\)\]]\s+", r"\1. ", l)  # "1)" -> "1. "
        lines.append(l)
    txt = "\n".join(lines)
    # encadre blocs ASCII
    txt = re.sub(
        r"(?:^|\n)((?:[|+\-=_].*\n){2,})",
        lambda m: "```text\n" + m.group(1).strip() + "\n```",
        txt,
        flags=re.S
    )
    # === NEW: tuer les NaN éventuels générés par pandas
    txt = txt.replace(" NaN", " ").replace("NaN", " ")
    return txt.strip()

# ---------------------------
# OCR utils
# ---------------------------
_table_chars = re.compile(r"[|+\-=_]{3,}")  # heuristique ASCII

def _tess_config(psm: str, keep_spaces: bool, table_mode: bool) -> str:
    cfg = f"--psm {psm} --oem 1"
    if keep_spaces:
        cfg += " -c preserve_interword_spaces=1"
    if table_mode:
        cfg += " -c tessedit_write_images=false"
    return cfg

# === NEW: OpenCV deskew & binarization
def _opencv_preprocess_for_ocr(im: Image.Image) -> Image.Image:
    if not _HAS_CV2:
        return im
    # PIL -> OpenCV (grayscale)
    cv = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY)

    # Binarize adaptative (optionnel)
    if OCR_BINARIZE:
        cv = cv2.adaptiveThreshold(cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 35, 11)

    # Deskew (optionnel)
    if OCR_DESKEW:
        coords = cv2.findNonZero(255 - cv)  # texte = noir
        if coords is not None:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = cv.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            cv = cv2.warpAffine(cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Retour PIL
    return Image.fromarray(cv)

def _preprocess_for_ocr(im: Image.Image) -> Image.Image:
    # Pipeline léger PIL (toujours sûr)
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    # Seuil doux
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
    noise = sum(1 for l in lines if "nnn" in l or "$-----" in l)
    return (pipes*1.0 + plus*0.6 + dashes*0.3 + ascii_blocks*2.0)/n - noise*0.25 + len(txt)/5000.0

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
        if in_blk: buf.append(line)
        else: out.append(line)
    if in_blk:
        out.extend(buf); out.append("```")
    return "\n".join(out)

def _ocr_image_best(im: Image.Image, langs: str) -> Tuple[str, float]:
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    # === NEW: OpenCV preprocessing first (if available)
    try:
        if _HAS_CV2:
            im2 = _opencv_preprocess_for_ocr(im)
        else:
            im2 = _preprocess_for_ocr(im)
    except Exception:
        im2 = _preprocess_for_ocr(im)

    best_txt, best_score = "", -1e9
    for psm in OCR_PSMS:
        cfg = _tess_config(psm, OCR_KEEP_SPACES, OCR_TABLE_MODE)
        t1 = pytesseract.image_to_string(im, lang=langs, config=cfg) or ""
        s1 = _score_text_for_table(t1)
        cand_txt, cand_score = t1, s1
        if OCR_TWO_PASS:
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
# PDF inline : texte + images + OCR + hiérarchie & listes
# ---------------------------

def _is_bold(flags: int) -> bool:
    return bool(flags & 1 or flags & 32)

def _classify_heading(size: float, size_levels: List[float]) -> Optional[str]:
    """Map taille → #, ##, ### selon quantiles mesurés sur la page/document."""
    if not size_levels:
        return None
    # size_levels triées décroissant : [H1, H2, H3 ...]
    for idx, s in enumerate(size_levels[:6], start=1):
        if size >= s * 0.98:  # tolérance
            return "#" * idx
    return None

_bullet_re = re.compile(r"^\s*(?:[-–—•·●◦▪]|\d+[.)])\s+")
_table_border_re = re.compile(r"^\s*[+].*[-+].*[+]\s*$")

def _median_font_size(page_raw: Dict[str,Any]) -> float:
    sizes = []
    for b in page_raw.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = s.get("text","").strip()
                if t:
                    sizes.append(float(s.get("size",0)))
    if not sizes:
        return 0.0
    sizes.sort()
    mid = len(sizes)//2
    return sizes[mid] if len(sizes)%2==1 else (sizes[mid-1]+sizes[mid])/2.0

def _collect_sizes(doc: fitz.Document, sample_pages: int = 5) -> List[float]:
    """Scan quelques pages pour extraire les plus grosses tailles, sert de base aux niveaux Hx."""
    sizes = []
    n = min(sample_pages, doc.page_count)
    for p in range(n):
        raw = doc.load_page(p).get_text("rawdict") or {}
        for b in raw.get("blocks", []):
            if b.get("type", 0) != 0:
                continue
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    t = s.get("text","").strip()
                    if t:
                        sizes.append(float(s.get("size",0)))
    if not sizes:
        return []
    sizes = sorted(set(sizes), reverse=True)
    return sizes[:6]  # top tailles

def _line_to_md(spans: List[Dict[str,Any]], size_levels: List[float]) -> str:
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
    # list bullets/ordered untouched (laisser "- " ou "1. " si déjà présent)
    if _bullet_re.match(raw):
        return raw
    # headings by size levels
    h = _classify_heading(max_size, size_levels)
    if h and len(raw) < 180:
        return f"{h} {raw}"
    return raw

def _median_line_height(page_raw: Dict[str, Any]) -> float:
    heights = []
    for b in page_raw.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            ymin = min(s.get("bbox", [0,0,0,0])[1] for s in l.get("spans", []) if s.get("bbox"))
            ymax = max(s.get("bbox", [0,0,0,0])[3] for s in l.get("spans", []) if s.get("bbox"))
            if ymax > ymin:
                heights.append(ymax - ymin)
    if not heights:
        return 12.0
    heights.sort()
    mid = len(heights)//2
    return heights[mid] if len(heights)%2==1 else (heights[mid-1]+heights[mid])/2.0

def _band_key(y_center: float, band_h: float) -> int:
    if band_h <= 0:
        band_h = 12.0
    return int(y_center / band_h)

def _merge_wrapped_paragraphs(lines: List[str]) -> List[str]:
    """
    Fusionne les retours à la ligne cassés par l'extraction en paragraphes continus.
    Heuristique : si une ligne se termine sans ponctuation forte et que la suivante
    commence par minuscule, fusionne avec espace.
    """
    out = []
    buf = ""
    def flush():
        nonlocal buf
        if buf.strip():
            out.append(buf.strip())
        buf = ""
    for l in lines:
        s = l.rstrip()
        if not s:
            flush()
            continue
        if buf:
            # ponctuation forte ?
            if re.search(r"[.!?;:]$", buf) or s[:1].isupper():
                # nouveau paragraphe
                flush()
                buf = s
            else:
                # suite de paragraphe
                buf += " " + s.lstrip()
        else:
            buf = s
    flush()
    return out

def _ascii_table_to_md(block: str) -> Optional[str]:
    """
    Convertit un tableau ASCII (+---+ ... | ... |) en tableau Markdown via pandas.
    Retourne None si ce n'est clairement pas un tableau.
    """
    lines = [l for l in block.splitlines() if l.strip()]
    border_lines = [i for i, l in enumerate(lines) if _table_border_re.match(l)]
    if len(border_lines) < 2:
        return None
    # Garder uniquement les lignes de contenu `| col | col |`
    content = [l for l in lines if l.strip().startswith("|") and l.strip().endswith("|")]
    if not content:
        return None
    # Split columns
    rows = []
    for row in content:
        cells = [c.strip() for c in row.strip()[1:-1].split("|")]
        rows.append(cells)
    if not rows:
        return None
    # Première ligne comme header si plausible
    header = rows[0]
    data = rows[1:] if len(rows) > 1 else []
    # DataFrame -> markdown (github table)
    df = pd.DataFrame(data=data, columns=header)
    df = df.where(df.notna(), "")
    return df.to_markdown(index=False, tablefmt="github")

def render_pdf_markdown_inline(pdf_bytes: bytes) -> Tuple[str, Dict[str,Any]]:
    """
    Parcourt chaque page et insère texte/images à leur place.
    Améliorations:
      - Détection titres par niveaux (tailles de police globales)
      - Fusion intelligente de lignes en paragraphes
      - Détection listes
      - OCR local images + fallback page
      - Conversion tableaux ASCII → vrais tableaux Markdown
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_lines: List[str] = []
    meta: Dict[str,Any] = {"engine": "pymupdf_inline", "pages": doc.page_count}
    try:
        size_levels = _collect_sizes(doc, sample_pages=5)
        total_pages = min(doc.page_count, OCR_MAX_PAGES)

        for p in range(total_pages):
            page = doc.load_page(p)
            raw = page.get_text("rawdict") or {}
            median_size = _median_font_size(raw)
            line_h_med = _median_line_height(raw)
            band_h = max(8.0, line_h_med * 1.2)

            atoms = []
            page_w = page.rect.width
            page_h = page.rect.height
            page_area = max(1.0, page_w * page_h)

            for b in raw.get("blocks", []):
                btype = b.get("type", 0)
                bbox = tuple(b.get("bbox", (0,0,0,0)))
                x0,y0,x1,y1 = bbox
                x0 = max(0.0, min(x0, page_w)); x1 = max(0.0, min(x1, page_w))
                y0 = max(0.0, min(y0, page_h)); y1 = max(0.0, min(y1, page_h))
                bbox = (x0,y0,x1,y1)

                if btype == 0:
                    # Lignes de texte → meilleure granularité
                    for line in b.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        lx0 = min(s.get("bbox", [x0,y0,x1,y1])[0] for s in spans if s.get("bbox"))
                        ly0 = min(s.get("bbox", [x0,y0,x1,y1])[1] for s in spans if s.get("bbox"))
                        lx1 = max(s.get("bbox", [x0,y0,x1,y1])[2] for s in spans if s.get("bbox"))
                        ly1 = max(s.get("bbox", [x0,y0,x1,y1])[3] for s in spans if s.get("bbox"))
                        line_bbox = (lx0, ly0, lx1, ly1)

                        md_line = _line_to_md(spans, size_levels).strip()
                        if not md_line:
                            continue
                        atoms.append({
                            "bbox": line_bbox,
                            "md": md_line,
                            "kind": "text",
                            "text_len": len(md_line),
                            "area_ratio": ((lx1-lx0)*(ly1-ly0)) / page_area
                        })

                elif btype == 1:
                    # IMAGE
                    im = crop_bbox_image(page, bbox, OCR_DPI)
                    if im is None:
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
                    area_ratio = ((x1-x0)*(y1-y0))/page_area
                    is_background_like = area_ratio > 0.85

                    if txt and score >= OCR_SCORE_GOOD_ENOUGH and not is_background_like:
                        md_img = _wrap_tables_as_code(txt.strip())
                    else:
                        md_img = ""
                        if EMBED_IMAGES in ("all", "ocr_only"):
                            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            md_img = f'![{IMG_ALT_PREFIX} – page {p+1}]({data_uri})'

                    if md_img:
                        atoms.append({
                            "bbox": bbox,
                            "md": md_img,
                            "kind": "image",
                            "text_len": len(txt or ""),
                            "area_ratio": area_ratio
                        })

            # Tri lecture
            def sort_key(a):
                x0,y0,x1,y1 = a["bbox"]
                y_center = 0.5*(y0+y1)
                return (_band_key(y_center, band_h), x0, y0)
            atoms.sort(key=sort_key)

            # Concat avec fusion de paragraphes + conversion tableaux ASCII
            page_buf: List[str] = []
            para_buf: List[str] = []
            last_band = None

            def flush_para():
                if not para_buf:
                    return
                merged = _merge_wrapped_paragraphs(para_buf)
                # Conversion ASCII table → Markdown table si applicable
                for block in merged:
                    if _table_chars.search(block) and ("|" in block or "+" in block):
                        md_tbl = _ascii_table_to_md(block)
                        if md_tbl:
                            page_buf.append(md_tbl)
                            continue
                    page_buf.append(block)
                para_buf.clear()

            for a in atoms:
                x0,y0,x1,y1 = a["bbox"]
                y_center = 0.5*(y0+y1)
                band = _band_key(y_center, band_h)
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

            # Fallback OCR pleine page si texte pauvre
            page_text_chars = sum(len(x) for x in page_buf if x and not x.strip().startswith("!["))  # ignore images
            if page_text_chars < OCR_MIN_CHARS:
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
                        # tenter conversion ASCII→MD table à l'intérieur aussi
                        blocks = _merge_wrapped_paragraphs(best_txt.splitlines())
                        for block in blocks:
                            if _table_chars.search(block) and ("|" in block or "+" in block):
                                md_tbl = _ascii_table_to_md(block)
                                if md_tbl:
                                    page_buf.append(md_tbl)
                                    continue
                            page_buf.append(_wrap_tables_as_code(block))
                except Exception:
                    pass

            if page_buf:
                md_lines.append("\n\n".join(page_buf))

        final_md = "\n\n".join([l for l in md_lines if l.strip()]).strip()
        return (_md_cleanup(final_md), meta)
    finally:
        doc.close()

# ---------------------------
# DOCX / XLSX / CSV — conversions dédiées (propres)
# ---------------------------

def docx_to_md_bytes(data: bytes) -> str:
    """
    Utilise MarkItDown. Si images, MarkItDown les inline déjà via plugins.
    Si besoin d'un contrôle plus fin, on pourrait passer par Mammoth directement.
    """
    md = MarkItDown(enable_plugins=True).convert_stream(io.BytesIO(data), file_name="file.docx")
    text = getattr(md, "text_content", "") or ""
    return _md_cleanup(text)

def xlsx_to_md_bytes(data: bytes) -> str:
    """
    Pandas → vraies tables Markdown (une table par feuille).
    Remplace NaN, ajoute le titre de feuille.
    """
    bio = io.BytesIO(data)
    xls = pd.ExcelFile(bio)
    out = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, dtype=str)  # str pour ne rien perdre
        df = df.where(df.notna(), "")
        tbl = df.to_markdown(index=False, tablefmt="github")
        out.append(f"### Feuille : {sheet}\n\n{tbl}\n")
    return _md_cleanup("\n".join(out))

def csv_to_md_bytes(data: bytes) -> str:
    """
    Détecteur simple et Pandas → table Markdown.
    """
    bio = io.BytesIO(data)
    try:
        df = pd.read_csv(bio, sep=None, engine="python", dtype=str)
    except Exception:
        bio.seek(0)
        df = pd.read_csv(bio, sep=",", dtype=str)
    df = df.where(df.notna(), "")
    tbl = df.to_markdown(index=False, tablefmt="github")
    return _md_cleanup(tbl)

# ---------------------------
# Mini interface (inchangée, juste version)
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
      margin:0; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica Neue,Arial;
      color:var(--text);
      background:
        radial-gradient(1000px 600px at 20% -10%, #183a58 0%, transparent 60%),
        radial-gradient(900px 500px at 120% 10%, #3b2d6a 0%, transparent 55%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
      background-attachment: fixed;
      line-height:1.55; padding:32px 20px 40px;
    }
    h1{margin:0 0 .35rem 0; font-size:1.65rem; letter-spacing:.2px}
    .sub{color:var(--muted); font-size:.95rem; margin-bottom:18px}
    .container{max-width:1060px; margin:0 auto}
    .card{background:var(--card); border:1px solid var(--card-border); border-radius:var(--radius); box-shadow:var(--shadow); padding:18px; margin-top:16px; backdrop-filter: blur(8px);}
    .row{display:flex; gap:12px; align-items:center; flex-wrap:wrap}
    label{font-weight:600}
    input[type="text"], input[type="file"], textarea{background:rgba(255,255,255,0.03); color:var(--text); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:10px 12px; outline:none; transition:border .15s, box-shadow .15s}
    input[type="text"]:focus, textarea:focus{border-color:var(--accent); box-shadow:0 0 0 3px rgba(99,179,255,.15)}
    textarea{width:100%; min-height:280px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size:.95rem; resize: vertical;}
    button{padding:10px 16px; border-radius:12px; border:1px solid rgba(255,255,255,0.12); color:#0b0f14;
      background: linear-gradient(135deg, var(--accent), var(--accent-2)); cursor:pointer; font-weight:700; letter-spacing:.2px; transition:transform .06s, filter .15s, opacity .2s}
    button:hover{ filter:brightness(1.08) } button:active{ transform:translateY(1px) } button:disabled{ opacity:.55; cursor:not-allowed; filter:none }
    .btn-ghost{ background:transparent; color:var(--text); border-color:rgba(255,255,255,0.16) }
    a#download{ display:inline-flex; align-items:center; gap:8px; padding:10px 16px; border-radius:12px; border:1px solid rgba(255,255,255,0.16); text-decoration:none; color:var(--text); background:rgba(255,255,255,0.03); transition:background .15s, border-color .15s }
    a#download:hover{ background:rgba(255,255,255,0.06); border-color:rgba(255,255,255,0.28) }
    .muted{color:var(--muted)}
    .drop{ border:1.5px dashed rgba(255,255,255,0.18); border-radius:14px; padding:18px; text-align:center; cursor:pointer; transition:.15s border-color, background .15s; background:rgba(255,255,255,0.02) }
    .drop:hover{border-color:rgba(255,255,255,0.35)} .drop.active{border-color:var(--accent); background:rgba(99,179,255,.06)}
    .filemeta{font-size:.95rem; color:var(--text); opacity:.9}
    .switch{position:relative; display:inline-block; width:44px; height:24px}
    .switch input{opacity:0; width:0; height:0}
    .slider{position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0; background:rgba(255,255,255,0.12); transition:.2s; border-radius:999px; border:1px solid rgba(255,255,255,0.2)}
    .slider:before{position:absolute; content:""; height:18px; width:18px; left:2px; top:2.5px; background:white; transition:.2s; border-radius:50%}
    .switch input:checked + .slider{background:linear-gradient(135deg, var(--accent), var(--accent-2)); border-color:transparent}
    .switch input:checked + .slider:before{ transform:translateX(20px) }
    .progress{height:10px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden; display:none; margin-top:10px}
    .bar{height:100%; width:40%; background:linear-gradient(135deg, var(--accent), var(--accent-2)); border-radius:999px; animation:slide 1.2s infinite}
    @keyframes slide{0%{transform:translateX(-100%)}50%{transform:translateX(50%)}100%{transform:translateX(150%)}}
    .stats{display:flex; gap:12px; align-items:center; margin-top:10px; flex-wrap:wrap}
    .tag{display:inline-flex; gap:6px; align-items:center; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15); border-radius:999px; padding:6px 10px; font-size:.9rem}
    .tag b{font-weight:800}
    .hero{ background: linear-gradient(135deg, rgba(99,179,255,.14), rgba(143,122,255,.18));
      border:1px solid rgba(255,255,255,0.12); border-radius: 18px; padding: 16px 18px; box-shadow: var(--shadow); margin-bottom: 18px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>MarkItDown — Conversion</h1>
      <div class="sub">PDF/Images → Markdown hiérarchisé (+ OCR). DOCX/XLSX/CSV → Markdown propre (tables, listes). Résumé Azure en option.</div>
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
const $ = (id)=>document.getElementById(id);
const endpoint = "/convert";

fetch("/config").then(r=>r.ok?r.json():null).then(j=>{
  if(j && j.docintel_default){ $("di").value = j.docintel_default; }
}).catch(()=>{});

// Drag & drop
(function(){
  const dz = $("dropzone");
  const fi = $("file");
  const fm = $("filemeta");
  function prettySize(bytes){
    if(bytes < 1024) return bytes + " B";
    if(bytes < 1024*1024) return (bytes/1024).toFixed(1) + " KB";
    return (bytes/1024/1024).toFixed(1) + " MB";
  }
  function showMeta(f){
    if(!f){ fm.textContent = ""; return; }
    fm.textContent = `${f.name} — ${prettySize(f.size)}`;
  }
  dz.addEventListener("click", ()=> fi.click());
  dz.addEventListener("dragover", e=>{ e.preventDefault(); dz.classList.add("active"); });
  dz.addEventListener("dragleave", ()=> dz.classList.remove("active"));
  dz.addEventListener("drop", e=>{
    e.preventDefault(); dz.classList.remove("active");
    if(e.dataTransfer.files && e.dataTransfer.files[0]){
      fi.files = e.dataTransfer.files;
      showMeta(fi.files[0]);
    }
  });
  fi.addEventListener("change", ()=> showMeta(fi.files[0]));
})();

// Timer/counters/ui unchanged...
let timerId = null, t0 = 0;
function startTimer(){ stopTimer(); t0 = performance.now(); timerId = setInterval(()=>{
  const secs = (performance.now() - t0)/1000;
  $("timer").textContent = (secs < 60) ? secs.toFixed(2) + " s" : (Math.floor(secs/60) + "m " + (secs%60).toFixed(1) + "s");
}, 100);}
function stopTimer(final=false){ if(timerId){ clearInterval(timerId); timerId=null; }
  if(final){ const secs=(performance.now()-t0)/1000;
    $("timer").textContent = (secs < 60) ? secs.toFixed(2) + " s" : (Math.floor(secs/60) + "m " + (secs%60).toFixed(1) + "s");
  }}
function updateCounters(){
  const txt = $("md").value || "";
  $("charcount").textContent = txt.length.toString();
  $("linecount").textContent = (txt ? txt.split(/\r?\n/).length : 0).toString();
}
$("copy").onclick = async ()=>{
  try{ await navigator.clipboard.writeText($("md").value || "");
    $("status").textContent = "Markdown copié";
    setTimeout(()=>{$("status").textContent="";}, 1400);
  }catch(_){ $("status").textContent = "Impossible de copier."; }
};
$("clear").onclick = ()=>{
  $("md").value = ""; $("meta").value = ""; $("download").style.display = "none"; updateCounters();
  $("status").textContent = "Zones effacées."; setTimeout(()=>{$("status").textContent="";}, 1200);
};
$("convert").onclick = async () => {
  const f = $("file").files[0];
  if(!f){ alert("Choisis un fichier."); return; }
  $("convert").disabled = true; $("status").textContent = "Conversion en cours...";
  $("md").value = ""; $("meta").value = ""; $("download").style.display = "none";
  $("progress").style.display = "block"; startTimer();
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
    updateCounters();
    const blob = new Blob([$("md").value], {type:"text/markdown;charset=utf-8"});
    const url  = URL.createObjectURL(blob);
    const a = $("download");
    a.href = url;
    a.download = (json.output_filename || "sortie.md");
    a.style.display = "inline-flex";
    $("status").textContent = "OK";
  }catch(e){
    $("status").textContent = "Erreur : " + (e && e.message ? e.message : e);
  }finally{
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
    llm_model: Optional[str] = Form(None),   # compat
    use_llm: bool = Form(False),
    force_ocr: bool = Form(False),
):
    """
    Règles:
    - PDF:
      * Plugins + Forcer OCR (et OCR activé) -> pipeline PyMuPDF inline amélioré (titres/listes/tables/ocr/images)
      * Sinon -> MarkItDown comme avant (+ cleanup)
    - IMG:
      * OCR image + base64 si besoin
    - DOCX / XLSX / CSV:
      * Conversions dédiées (Pandas/Mammoth) → Markdown propre
    """
    try:
        t_start = time.perf_counter()

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
        is_docx = guess_is_docx(file.filename)
        is_xlsx = guess_is_xlsx(file.filename)
        is_csv  = guess_is_csv(file.filename)

        metadata: Dict[str,Any] = {}

        # === DOCX/XLSX/CSV — conversions dédiées
        if is_docx:
            markdown = docx_to_md_bytes(content)

        elif is_xlsx:
            markdown = xlsx_to_md_bytes(content)

        elif is_csv:
            markdown = csv_to_md_bytes(content)

        # === PDF
        elif is_pdf and use_plugins and force_ocr and OCR_ENABLED:
            markdown, meta_pdf = render_pdf_markdown_inline(content)
            metadata.update(meta_pdf)

        else:
            # MarkItDown générique (PDF ou autres) + cleanup
            md_engine = MarkItDown(enable_plugins=use_plugins, docintel_endpoint=docintel_endpoint)
            result = md_engine.convert_stream(io.BytesIO(content), file_name=file.filename)

            markdown = getattr(result, "text_content", "") or ""
            metadata.update(getattr(result, "metadata", {}) or {})
            warnings = getattr(result, "warnings", None)
            if warnings:
                metadata["warnings"] = warnings
            markdown = _md_cleanup(markdown)

            # Image seule + Forcer OCR
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

        # Durée côté serveur
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
