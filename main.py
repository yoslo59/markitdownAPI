import os
import io
import re
import base64
import time
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
OCR_MODE           = os.getenv("OCR_MODE", "append").strip()
OCR_KEEP_SPACES    = os.getenv("OCR_KEEP_SPACES", "true").lower() == "true"
OCR_TWO_PASS       = os.getenv("OCR_TWO_PASS", "true").lower() == "true"
OCR_TABLE_MODE     = os.getenv("OCR_TABLE_MODE", "true").lower() == "true"
OCR_PSMS           = [p.strip() for p in os.getenv("OCR_PSMS", "6,4,11").split(",")]  # 6=block,4=cols,11=sparse
OCR_DPI_CANDIDATES = [int(x) for x in os.getenv("OCR_DPI_CANDIDATES", "300,350,400").split(",")]
OCR_SCORE_GOOD_ENOUGH = float(os.getenv("OCR_SCORE_GOOD_ENOUGH", "0.6"))

# Embedding images base64
# none | ocr_only | all
EMBED_IMAGES       = os.getenv("EMBED_IMAGES", "ocr_only").strip()
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
app = FastAPI(title="MarkItDown API", version="2.6")

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
# PDF inline : texte + images à la bonne place (+ fallback page OCR)
# ---------------------------
def _is_bold(flags: int) -> bool:
    return bool(flags & 1 or flags & 32)  # BOLD | FAKEBOLD

def _classify_heading(size: float, median_size: float) -> Optional[str]:
    if median_size <= 0:
        return None
    if size >= median_size * 1.8: return "#"
    if size >= median_size * 1.5: return "##"
    if size >= median_size * 1.25: return "###"
    return None

_bullet_re = re.compile(r"^\s*(?:[-–—•·●◦▪]|\d+[.)])\s+")

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
    return sizes[mid] if len(sizes)%2==1 else (sizes[mid-1]+sizes[mid])/2.0

def _line_to_md(spans: List[Dict[str,Any]], median_size: float) -> str:
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
    h = _classify_heading(max_size, median_size)
    if h and len(raw) < 180:
        return f"{h} {raw}"
    if _bullet_re.match(raw):
        return f"{raw}"
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

def render_pdf_markdown_inline(pdf_bytes: bytes) -> Tuple[str, Dict[str,Any]]:
    """
    Parcourt chaque page et insère texte/images exactement à leur place (lecture visuelle).
    Stratégie :
      - On crée des 'atoms' (par LIGNE de texte + par IMAGE) avec bbox.
      - On convertit chaque ligne en MD (gras, titres, listes), chaque image en OCR texte
        sinon en base64 si OCR insuffisant.
      - On trie par 'bandes' verticales (y) puis gauche→droite (x) pour gérer multi-colonnes.
      - Fallback: si page encore pauvre en texte → OCR pleine page INSÉRÉ ICI (pas en fin).
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
            line_h_med = _median_line_height(raw)
            band_h = max(8.0, line_h_med * 1.2)

            # 1) Construire les atoms (texte par LIGNE + images)
            atoms = []  # dicts: {"bbox":(x0,y0,x1,y1), "md":str, "kind":"text"|"image", "text_len":int, "area_ratio":float}
            page_w = page.rect.width
            page_h = page.rect.height
            page_area = max(1.0, page_w * page_h)

            for b in raw.get("blocks", []):
                btype = b.get("type", 0)
                bbox = tuple(b.get("bbox", (0,0,0,0)))
                x0,y0,x1,y1 = bbox
                # Sanity: garder dans page
                x0 = max(0.0, min(x0, page_w)); x1 = max(0.0, min(x1, page_w))
                y0 = max(0.0, min(y0, page_h)); y1 = max(0.0, min(y1, page_h))
                bbox = (x0,y0,x1,y1)

                if btype == 0:
                    # Pour le texte, on fabrique un atom PAR LIGNE (meilleure granularité de position)
                    for line in b.get("lines", []):
                        spans = line.get("spans", [])
                        # bbox de la ligne (union des spans)
                        if not spans:
                            continue
                        lx0 = min(s.get("bbox", [x0,y0,x1,y1])[0] for s in spans if s.get("bbox"))
                        ly0 = min(s.get("bbox", [x0,y0,x1,y1])[1] for s in spans if s.get("bbox"))
                        lx1 = max(s.get("bbox", [x0,y0,x1,y1])[2] for s in spans if s.get("bbox"))
                        ly1 = max(s.get("bbox", [x0,y0,x1,y1])[3] for s in spans if s.get("bbox"))
                        line_bbox = (lx0, ly0, lx1, ly1)

                        md_line = _line_to_md(spans, median_size).strip()
                        if not md_line:
                            continue
                        # Emballage ASCII table si besoin sera fait plus tard au blocage; ici, on laisse la ligne brute.
                        atoms.append({
                            "bbox": line_bbox,
                            "md": md_line,
                            "kind": "text",
                            "text_len": len(md_line),
                            "area_ratio": ( (lx1-lx0) * (ly1-ly0) ) / page_area
                        })

                elif btype == 1:
                    # IMAGE: OCR localisé, sinon base64
                    im = crop_bbox_image(page, bbox, OCR_DPI)
                    if im is None:
                        # fallback xref si dispo
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

                    # Heuristique: ignorer très grandes images "fond" si on aura suffisamment de texte par ailleurs.
                    is_background_like = area_ratio > 0.85

                    if txt and score >= OCR_SCORE_GOOD_ENOUGH and not is_background_like:
                        md_img = _wrap_tables_as_code(txt.strip())
                    else:
                        # base64 si demandé
                        md_img = ""
                        if EMBED_IMAGES in ("all", "ocr_only"):
                            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            md_img = f'![{IMG_ALT_PREFIX} p{p+1}]({data_uri})'

                    if md_img:
                        atoms.append({
                            "bbox": bbox,
                            "md": md_img,
                            "kind": "image",
                            "text_len": len(txt or ""),
                            "area_ratio": area_ratio
                        })

            # 2) Tri lecture : bandes verticales (centre Y // band_h) puis X
            def sort_key(a):
                x0,y0,x1,y1 = a["bbox"]
                y_center = 0.5*(y0+y1)
                return (_band_key(y_center, band_h), x0, y0)
            atoms.sort(key=sort_key)

            # 3) Concat, avec regroupement ‘paragraphe’ + emballage ASCII tables
            page_buf: List[str] = []
            para_buf: List[str] = []
            last_band = None

            def flush_para():
                if not para_buf:
                    return
                block_txt = "\n".join(para_buf)
                # si présence nette de tableaux ASCII → encadrer
                if _table_chars.search(block_txt):
                    page_buf.append("```text")
                    page_buf.append(block_txt)
                    page_buf.append("```")
                else:
                    page_buf.append(block_txt)
                para_buf.clear()

            for a in atoms:
                x0,y0,x1,y1 = a["bbox"]
                y_center = 0.5*(y0+y1)
                band = _band_key(y_center, band_h)
                md = a["md"]

                # changement de bande = probablement retour à la ligne/bloc
                if last_band is not None and band != last_band:
                    flush_para()
                last_band = band

                if a["kind"] == "text":
                    # Coller la ligne au paragraphe en conservant le format (titres/listes déjà gérés)
                    para_buf.append(md)
                else:
                    # image/ocr: la poser "inline" → on coupe le paragraphe en cours
                    flush_para()
                    page_buf.append(md)

            # vider le dernier paragraphe
            flush_para()

            # 4) Fallback OCR pleine page si toujours pauvre en texte
            page_text_chars = sum(len(x) for x in page_buf if x and not x.strip().startswith("!["))  # ne compte pas l'image MD
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
# Mini interface web
# ---------------------------
HTML_PAGE = r'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>MarkItDown — Conversion</title>
  <style>
    :root{color-scheme: light dark;}
    *{box-sizing:border-box}
    body{
      margin:0;padding:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
      background:
        radial-gradient(1200px 600px at -10% -10%, #7dd3fc33 0, transparent 60%),
        radial-gradient(1200px 600px at 110% -10%, #a78bfa33 0, transparent 60%),
        radial-gradient(1200px 600px at 50% 120%, #34d39922 0, transparent 60%),
        linear-gradient(180deg, #0b0f18, #0b0f18);
      color:#e5e7eb;
    }
    .wrap{max-width:1100px;margin:0 auto;padding:32px}
    header{
      display:flex;gap:16px;align-items:center;justify-content:space-between;margin-bottom:20px
    }
    .brand{
      display:flex;gap:12px;align-items:center
    }
    .logo{
      width:40px;height:40px;border-radius:12px;
      background: linear-gradient(135deg,#22d3ee,#a78bfa);
      box-shadow:0 6px 30px #22d3ee55, inset 0 0 20px #ffffff22;
    }
    h1{margin:0;font-weight:800;letter-spacing:.3px}
    .sub{opacity:.8;margin-top:4px}

    .grid{
      display:grid;gap:18px;
      grid-template-columns: 1fr;
    }
    @media(min-width:1000px){ .grid{ grid-template-columns: 420px 1fr; } }

    .card{
      background: linear-gradient(180deg, #0f1422dd, #0a0f1add);
      border:1px solid #334155;
      border-radius:16px; padding:18px;
      box-shadow:
        0 20px 60px #00000066,
        inset 0 1px 0 #ffffff0f;
      backdrop-filter: blur(6px);
    }
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    label{font-weight:600}
    .muted{color:#94a3b8;font-size:12px}

    .btn{
      appearance:none;border:0;border-radius:12px;padding:10px 16px;
      font-weight:700;letter-spacing:.3px;cursor:pointer;
      color:#0b0f18;background:#e5e7eb;
      transition: transform .06s ease, box-shadow .2s ease, opacity .2s ease;
      box-shadow: 0 10px 30px #22d3ee33, inset 0 -2px 0 #00000011;
    }
    .btn:hover{transform: translateY(-1px)}
    .btn:disabled{opacity:.5;cursor:not-allowed}

    input[type="checkbox"]{transform: scale(1.15); accent-color:#22d3ee}

    input[type="text"]{
      color:#e5e7eb;background:#0b1220;border:1px solid #334155;
      border-radius:12px;padding:10px 12px;min-width:360px;outline:none
    }
    input[type="text"]::placeholder{color:#64748b}

    textarea{
      width:100%;min-height:280px;padding:12px;border-radius:12px;
      background:#0b1220;border:1px solid #334155;color:#e5e7eb;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      line-height:1.5;
    }

    .drop{
      border:1.5px dashed #334155;border-radius:14px;padding:14px;
      background:#0b1220; color:#cbd5e1; text-align:center;
      transition:border-color .2s ease, background .2s ease;
    }
    .drop.drag{ border-color:#22d3ee; background:#0b122055; }

    .kpi{
      display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-top:10px
    }
    .chip{
      border:1px solid #334155;border-radius:999px;padding:6px 10px;
      background:#0b1220; font-size:12px; color:#cbd5e1
    }

    .timer{
      font-variant-numeric: tabular-nums; font-weight:800; letter-spacing:.5px;
      padding:8px 10px;border-radius:10px;
      background: #0b1220; border:1px solid #334155; color:#e2e8f0;
      box-shadow: inset 0 1px 0 #ffffff0f;
      min-width: 120px; text-align:center
    }

    .status{
      display:flex; align-items:center; gap:8px; margin-top:10px
    }
    .dot{
      width:10px;height:10px;border-radius:999px;background:#22d3ee;box-shadow:0 0 14px #22d3ee99
    }
    .spin{
      width:16px;height:16px;border:2px solid #94a3b855;border-top-color:#22d3ee;border-radius:50%;
      animation:spin 1s linear infinite
    }
    @keyframes spin{ to{ transform:rotate(360deg)} }

    .download{
      display:inline-flex;align-items:center;gap:8px;margin-left:8px;
      padding:10px 14px;border-radius:12px;border:1px solid #334155;
      color:#e5e7eb;text-decoration:none;background:#0b1220;
    }

    .area-title{font-weight:700;margin-bottom:8px;opacity:.9}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="brand">
        <div class="logo"></div>
        <div>
          <h1>MarkItDown — Conversion</h1>
          <div class="sub">Convertit (PDF, DOCX, PPTX, XLSX, HTML, images) → Markdown. Plugins, OCR, images base64.</div>
        </div>
      </div>
    </header>

    <div class="grid">
      <!-- Panneau gauche (upload + options) -->
      <div class="card">
        <div class="area-title">Document</div>
        <div id="drop" class="drop">
          <div><strong>Glisse-dépose</strong> un fichier ici ou</div>
          <div style="margin-top:10px">
            <input id="file" type="file" />
          </div>
          <div id="fname" class="muted" style="margin-top:8px"></div>
        </div>

        <div style="height:10px"></div>
        <div class="area-title">Options</div>
        <div class="row" style="gap:14px">
          <label for="plugins">Activer plugins MarkItDown</label>
          <input id="plugins" type="checkbox" />
        </div>
        <div class="row" style="gap:14px;margin-top:6px">
          <label for="forceocr">Forcer OCR</label>
          <input id="forceocr" type="checkbox" />
        </div>
        <div class="row" style="gap:14px;margin-top:6px">
          <label for="llm">Résumé Azure LLM</label>
          <input id="llm" type="checkbox" />
        </div>
        <div class="row" style="gap:10px;align-items:baseline;margin-top:10px">
          <label for="di" style="min-width:250px">Endpoint Azure Document Intelligence</label>
          <input id="di" type="text" placeholder="https://<resource>.cognitiveservices.azure.com/"/>
        </div>

        <div class="kpi">
          <div class="timer" id="timer">00:00.0</div>
          <div id="serverTime" class="chip" style="display:none"></div>
          <div id="sizeInfo" class="chip" style="display:none"></div>
        </div>

        <div class="row" style="margin-top:14px">
          <button class="btn" id="convert">Convertir</button>
          <a class="download" id="download" download="sortie.md" style="display:none">Télécharger Markdown</a>
        </div>

        <div class="status">
          <div id="spin" class="spin" style="display:none"></div>
          <div id="status" class="muted"></div>
        </div>
      </div>

      <!-- Panneau droit (résultats) -->
      <div class="card">
        <div class="area-title">Markdown</div>
        <textarea id="md"></textarea>
      </div>

      <div class="card" style="grid-column:1/-1">
        <div class="area-title">Métadonnées (JSON)</div>
        <textarea id="meta" style="min-height:160px"></textarea>
      </div>
    </div>
  </div>

<script>
const $ = id => document.getElementById(id);
const endpoint = "/convert";

// préremplir l’endpoint DI si dispo
fetch("/config").then(r=>r.ok?r.json():null).then(j=>{
  if(j && j.docintel_default){ $("di").value = j.docintel_default; }
}).catch(()=>{});

// drag & drop
(() => {
  const drop = $("drop"), input = $("file"), fname = $("fname");
  const showName = f => { if(!f) return; fname.textContent = `${f.name} — ${(f.size/1024/1024).toFixed(2)} Mo`; $("sizeInfo").style.display="inline-block"; $("sizeInfo").textContent = `${(f.size/1024/1024).toFixed(2)} Mo`; };
  drop.addEventListener("dragover", e=>{e.preventDefault(); drop.classList.add("drag");});
  drop.addEventListener("dragleave", ()=>drop.classList.remove("drag"));
  drop.addEventListener("drop", e=>{
    e.preventDefault(); drop.classList.remove("drag");
    if(e.dataTransfer.files && e.dataTransfer.files[0]){
      input.files = e.dataTransfer.files;
      showName(input.files[0]);
    }
  });
  input.addEventListener("change", ()=> showName(input.files[0]));
})();

// chrono
let tStart=0, tTick=null;
function resetTimer(){ $("timer").textContent="00:00.0"; if(tTick){ clearInterval(tTick); tTick=null; } }
function startTimer(){
  tStart = performance.now();
  resetTimer();
  tTick = setInterval(()=>{
    const ms = performance.now()-tStart;
    const s = Math.floor(ms/1000);
    const d = (ms%1000)/100;
    const m = Math.floor(s/60);
    const ss = (s%60).toString().padStart(2,'0');
    $("timer").textContent = `${m.toString().padStart(2,'0')}:${ss}.${Math.floor(d)}`;
  }, 100);
}
function stopTimer(){ if(tTick){ clearInterval(tTick); tTick=null; } }

$("convert").onclick = async () => {
  const f = $("file").files[0];
  if(!f){ alert("Choisis un fichier."); return; }

  // reset UI
  $("convert").disabled = true;
  $("spin").style.display = "inline-block";
  $("status").textContent = "Conversion en cours…";
  $("md").value = "";
  $("meta").value = "";
  $("download").style.display = "none";
  $("serverTime").style.display = "none";

  // chrono client ON
  startTimer();

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
    $("status").textContent = "OK";

    // Download du MD
    const blob = new Blob([$("md").value], {type:"text/markdown;charset=utf-8"});
    const url  = URL.createObjectURL(blob);
    const a = $("download");
    a.href = url;
    a.download = (json.output_filename || "sortie.md");
    a.style.display = "inline-flex";

    // chrono client OFF
    stopTimer();

    // durée serveur si dispo
    if(typeof json.duration_ms === "number"){
      $("serverTime").style.display = "inline-block";
      $("serverTime").textContent = `Serveur: ${(json.duration_ms/1000).toFixed(2)} s`;
    }else if(json.metadata && typeof json.metadata.duration_ms==="number"){
      $("serverTime").style.display = "inline-block";
      $("serverTime").textContent = `Serveur: ${(json.metadata.duration_ms/1000).toFixed(2)} s`;
    }
  }catch(e){
    stopTimer();
    $("status").textContent = "Erreur : " + (e && e.message ? e.message : e);
  }finally{
    $("spin").style.display = "none";
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
    t0 = time.time()
    file: UploadFile = File(...),
    use_plugins: bool = Form(False),
    docintel_endpoint: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),   # ignoré pour Azure; gardé pour compat
    use_llm: bool = Form(False),
    force_ocr: bool = Form(False),
):
    """
    - Rien coché ou Plugins seul : MarkItDown (+ post-format).
    - Plugins + Forcer OCR (PDF) : pipeline PyMuPDF inline (texte + OCR / base64 in-place + fallback page).
    - Image seule + Forcer OCR : OCR + image base64 si OCR pauvre.
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

        # === Cas PDF + Plugins + Forcer OCR => PyMuPDF inline ===
        if is_pdf and use_plugins and force_ocr and OCR_ENABLED:
            markdown, meta_pdf = render_pdf_markdown_inline(content)
            metadata.update(meta_pdf)

        else:
            # === Pipeline MarkItDown (fonctionne même si plugins = False)
            md_engine = MarkItDown(enable_plugins=use_plugins, docintel_endpoint=docintel_endpoint)
            result = md_engine.convert_stream(io.BytesIO(content), file_name=file.filename)

            markdown = getattr(result, "text_content", "") or ""
            metadata.update(getattr(result, "metadata", {}) or {})
            warnings = getattr(result, "warnings", None)
            if warnings:
                metadata["warnings"] = warnings

            # post-traitement léger Markdown
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

        if SAVE_UPLOADS and in_path:
            metadata["saved_input_path"] = in_path
        if SAVE_OUTPUTS and out_path:
            metadata["saved_output_path"] = out_path
        duration_ms = int((time.time() - t0) * 1000)
        metadata["duration_ms"] = duration_ms
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
