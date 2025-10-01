import os, io, re, base64
from typing import Optional, Tuple, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Conversion library MarkItDown (Open-source Microsoft tool for docs → Markdown)
from markitdown import MarkItDown

# OCR and image libs
import fitz  # PyMuPDF
import cv2
from PIL import Image

# PaddleOCR pipeline (for OCR modes)
from paddleocr import PaddleOCR, PPStructure

# ---------------------------
# Config via variables d'environnement
# ---------------------------
SAVE_UPLOADS   = os.getenv("SAVE_UPLOADS", "false").lower() == "true"
SAVE_OUTPUTS   = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"
UPLOAD_DIR     = os.getenv("UPLOAD_DIR", "/data/uploads")
OUTPUT_DIR     = os.getenv("OUTPUT_DIR", "/data/outputs")

OCR_ENABLED    = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_MAX_PAGES  = int(os.getenv("OCR_MAX_PAGES", "50"))
OCR_DPI        = int(os.getenv("OCR_DPI", "350"))

# Stratégie d'inclusion des images : none | ocr_only | all
EMBED_IMAGES   = os.getenv("EMBED_IMAGES", "ocr_only").strip()
IMG_FORMAT     = os.getenv("IMG_FORMAT", "png").strip().lower()
IMG_JPEG_QUALITY = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH  = int(os.getenv("IMG_MAX_WIDTH", "1400"))
IMG_ALT_PREFIX = os.getenv("IMG_ALT_PREFIX", "Capture").strip()

# Préparation des dossiers de stockage
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API (PaddleOCR)", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["POST", "GET"], allow_headers=["*"],
)

# ---------------------------
# OCR Engine Initialization (lazy)
# ---------------------------
paddleocr_model: Optional[PaddleOCR] = None      # Mode rapide
paddlestruct_model: Optional[PPStructure] = None  # Mode qualité

def get_paddle_ocr():
    """Return a PaddleOCR (fast OCR) instance (initialize if needed)."""
    global paddleocr_model
    if paddleocr_model is None:
        # Déterminer la langue du modèle d'après OCR_LANGS (fra+eng -> french)
        ocr_langs = os.getenv("OCR_LANGS", "fra+eng")
        if ocr_langs.lower().startswith("fra"):
            lang = "french"
        elif ocr_langs.lower().startswith("en"):
            lang = "en"
        elif ocr_langs.lower().startswith("de"):
            lang = "german"
        elif ocr_langs.lower().startswith("ko"):
            lang = "korean"
        elif ocr_langs.lower().startswith("ja"):
            lang = "japan"
        else:
            lang = "en"
        paddleocr_model = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)
    return paddleocr_model

def get_paddle_structure():
    """Return a PPStructure (quality OCR) instance (initialize if needed)."""
    global paddlestruct_model
    if paddlestruct_model is None:
        # Même logique de langue que ci-dessus
        ocr_langs = os.getenv("OCR_LANGS", "fra+eng")
        if ocr_langs.lower().startswith("fra"):
            lang = "french"
        elif ocr_langs.lower().startswith("en"):
            lang = "en"
        elif ocr_langs.lower().startswith("de"):
            lang = "german"
        elif ocr_langs.lower().startswith("ko"):
            lang = "korean"
        elif ocr_langs.lower().startswith("ja"):
            lang = "japan"
        else:
            lang = "en"
        paddlestruct_model = PPStructure(show_log=False, 
                                         recovery=True, 
                                         return_ocr_result_in_table=True, 
                                         lang=lang)
    return paddlestruct_model

# ---------------------------
# Helpers génériques & Markdown
# ---------------------------
def guess_is_pdf(filename: str, content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return filename.lower().endswith(".pdf")

def guess_is_image(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))

def _md_cleanup(md: str) -> str:
    """Nettoyage du Markdown : trim espaces, normalisation listes, tableaux ASCII -> blocs de code."""
    if not md:
        return md
    lines = []
    for L in md.replace("\r", "").split("\n"):
        l = re.sub(r"[ \t]+$", "", L)                           # trim espaces fin
        l = re.sub(r"^\s*[•·●◦▪]\s+", "- ", l)                  # puces unicode -> "- "
        l = re.sub(r"^\s*(\d+)[\)\]]\s+", r"\1. ", l)           # "1)" ou "1]" -> "1. "
        lines.append(l)
    txt = "\n".join(lines)
    # Encadrer les tableaux ASCII détectés dans ```text
    txt = re.sub(
        r"(?:^|\n)((?:[|+\-=_].*\n){2,})",
        lambda m: "```text\n" + m.group(1).strip() + "\n```",
        txt, flags=re.S
    )
    return txt.strip()

def _pil_resize_max(im: Image.Image, max_w: int) -> Image.Image:
    """Redimensionne l'image PIL pour qu'elle ne dépasse pas max_w en largeur."""
    if max_w and im.width > max_w:
        ratio = max_w / im.width
        new_h = int(im.height * ratio)
        return im.resize((max_w, new_h), Image.LANCZOS)
    return im

def _pil_to_base64(im: Image.Image, fmt: str = "png", quality: int = 85) -> str:
    """Encode une image PIL en base64 data URI (format PNG ou JPEG)."""
    buf = io.BytesIO()
    if fmt in ("jpeg", "jpg"):
        im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _crop_page_region(page: fitz.Page, bbox: Tuple[float,float,float,float], dpi: int) -> Optional[Image.Image]:
    """Extrait une région de page PDF (bbox en coord. page) en image PIL à la résolution donnée."""
    try:
        x0, y0, x1, y1 = bbox
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1), alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None

# ---------------------------
# Pipeline OCR rapide (PaddleOCR plein page)
# ---------------------------
def ocr_page_fast(page: fitz.Page) -> Tuple[str, bool]:
    """
    OCR d'une page PDF en mode rapide.
    Retourne le markdown de la page et un indicateur si du texte a été extrait.
    """
    # Rasteriser la page entière en image
    pil_page = Image.frombytes("RGB", [page.rect.width, page.rect.height],
                               page.get_pixmap(matrix=fitz.Matrix(OCR_DPI/72, OCR_DPI/72), alpha=False).samples)
    # Conversion en image OpenCV (BGR)
    cv_page = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR) if 'np' in globals() else cv2.imread(page)  # ensure numpy is imported
    ocr = get_paddle_ocr()
    result = ocr.ocr(cv_page, cls=False)  # liste de [ [poly, (text, conf)], ... ]
    items: List[Dict[str, Any]] = []
    page_area = page.rect.width * page.rect.height
    # Ajouter résultats texte OCR
    for line in result:
        poly = line[0]
        text = line[1][0] if line[1] else ""
        if not text:
            continue
        # calculer bbox rectangulaire du poly
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x0,y0,x1,y1 = min(xs), min(ys), max(xs), max(ys)
        items.append({
            "bbox": (x0, y0, x1, y1),
            "md": text.strip(),
            "kind": "text",
            "text_len": len(text.strip())
        })
    # Ajouter les images natives du PDF
    rawdict = page.get_text("rawdict") or {}
    for b in rawdict.get("blocks", []):
        if b.get("type", 0) == 1:  # image block
            bbox = tuple(b.get("bbox", (0,0,0,0)))
            x0,y0,x1,y1 = bbox
            items.append({
                "bbox": bbox,
                "md": None,
                "kind": "image_raw",
                "text_len": 0,
                "area_ratio": ((x1-x0)*(y1-y0)) / (page_area if page_area>0 else 1)
            })
    # Traiter chaque image : tenter OCR localisé sinon embed base64
    processed: List[Dict[str, Any]] = []
    for item in items:
        if item["kind"] != "image_raw":
            processed.append(item)
            continue
        bbox = item["bbox"]
        im_crop = _crop_page_region(page, bbox, OCR_DPI)
        if im_crop is None:
            continue
        # Décider si on fait un OCR sur l'image (mode "ocr_only" => seulement si texte détecté)
        do_ocr = True
        if EMBED_IMAGES == "never" or EMBED_IMAGES == "none":
            do_ocr = False
        # Mode "ocr_only" ou "all" -> on tente OCR sur toutes les images (comportement par défaut rapide)
        # On pourrait implémenter un mode "smart" : détecter d'abord s'il y a du texte, mais ignoré ici pour simplicité.
        txt = ""
        if do_ocr:
            # OCR sur le crop d'image
            cv_crop = cv2.cvtColor(np.array(im_crop), cv2.COLOR_RGB2BGR)
            result_crop = ocr.ocr(cv_crop, cls=False)
            # Concaténer tout le texte détecté dans cette image
            txt = " ".join([res[1][0] for res in result_crop if res[1]]) if result_crop else ""
        # Si du texte a été extrait de l'image et semble non vide
        md_img = ""
        if txt.strip():
            md_img = txt.strip()
        else:
            # Pas de texte, ou texte vide -> insérer l'image en base64 si configuré
            if EMBED_IMAGES in ("all", "ocr_only"):
                im_resized = _pil_resize_max(im_crop, IMG_MAX_WIDTH)
                data_uri = _pil_to_base64(im_resized, IMG_FORMAT, IMG_JPEG_QUALITY)
                # Texte alternatif avec numéro de page
                md_img = f"![{IMG_ALT_PREFIX} – page {page.number+1}]({data_uri})"
        if md_img:
            processed.append({
                "bbox": bbox,
                "md": md_img,
                "kind": "image",
                "text_len": len(txt.strip())
            })
    # Trier éléments par position (approche haut-bas puis gauche-droite)
    if not processed:
        return ("", False)
    # Calculer hauteur médiane de ligne de texte pour regrouper paragraphes
    text_heights = []
    for it in processed:
        x0,y0,x1,y1 = it["bbox"]
        if it["kind"] == "text":
            text_heights.append(y1 - y0)
    median_h = sorted(text_heights)[len(text_heights)//2] if text_heights else 12.0
    band_h = max(8.0, median_h * 1.2)
    processed.sort(key=lambda a: (int(((a["bbox"][1]+a["bbox"][3])/2) / band_h), a["bbox"][0], a["bbox"][1]))
    # Concaténer en Markdown : regrouper lignes continues en paragraphes, insérer images sur lignes séparées
    page_lines: List[str] = []
    para_buf: List[str] = []
    last_band = None
    for it in processed:
        x0,y0,x1,y1 = it["bbox"]
        band = int(((y0+y1)/2) / band_h)
        md = it["md"]
        if last_band is not None and band != last_band:
            # nouvelle "bande" → fin de paragraphe courant
            if para_buf:
                page_lines.append("\n".join(para_buf))
                para_buf = []
        last_band = band
        if it["kind"] == "text":
            para_buf.append(md)
        else:  # image ou autre bloc
            if para_buf:
                page_lines.append("\n".join(para_buf))
                para_buf = []
            page_lines.append(md)
    # Flush le dernier paragraphe
    if para_buf:
        page_lines.append("\n".join(para_buf))
    page_md = "\n\n".join([line for line in page_lines if line.strip()])
    return (page_md, any(it.get("text_len",0) > 0 for it in processed))

# ---------------------------
# Pipeline OCR qualité (PaddleOCR PP-Structure)
# ---------------------------
def ocr_page_quality(page: fitz.Page) -> Tuple[str, bool]:
    """
    OCR d'une page PDF en mode qualité (PP-Structure).
    Retourne le markdown de la page et un indicateur texte (texte trouvé ou non).
    """
    # Conversion de la page en image (PIL) à la résolution spécifiée
    pix = page.get_pixmap(matrix=fitz.Matrix(OCR_DPI/72, OCR_DPI/72), alpha=False)
    pil_page = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    cv_page = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
    structure = get_paddle_structure()
    # Exécuter la pipeline PP-Structure sur l'image de page
    result = structure(cv_page)
    # `result` est une liste de dicts avec 'type' (Text/Title/List/Table/Image):contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}
    page_blocks: List[str] = []
    text_found = False
    for block in result:
        btype = block.get("type")
        # Extraire le texte OCR reconnu du bloc, selon son type
        if btype in ("Text", "Title", "List"):
            # Récupérer toutes les lignes de texte OCR du bloc
            ocr_lines = []
            res = block.get("res")
            if res:
                if isinstance(res, tuple):
                    # Forme (coords, [(text, conf), ...]) pour Text/Title/List
                    lines = res[1] if len(res) > 1 else []
                elif isinstance(res, list):
                    # Certains retours peuvent être liste de lignes (selon version)
                    lines = res
                else:
                    lines = []
            else:
                lines = []
            for line in lines:
                if not line: 
                    continue
                text = line[0] if isinstance(line, tuple) else line
                if text and text.strip():
                    ocr_lines.append(text.strip())
            if not ocr_lines:
                continue
            text_found = True
            # Joindre les lignes : 
            joiner = "\n" if btype == "List" else " "
            block_text = joiner.join(ocr_lines)
            # Ajouter préfixe titre si applicable
            if btype == "Title":
                # Premier titre = H1, suivants = H2
                if not any(t.startswith("#") for t in page_blocks):
                    block_text = "# " + block_text
                else:
                    block_text = "## " + block_text
            page_blocks.append(block_text)
        elif btype == "Table":
            # Extraire HTML du tableau et convertir en Markdown simple
            table_res = block.get("res")
            if table_res and isinstance(table_res, dict):
                html = table_res.get("html", "")
            else:
                html = block.get("res", "")  # parfois res peut être directement HTML
            if not html:
                continue
            text_found = True
            # Remplacer les <br> par des espaces pour ne pas coller les mots
            html = html.replace("<br/>", " ").replace("<br>", " ")
            # Extraire les lignes de tableau
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S|re.I)
            table_data = []
            for row_html in rows:
                # Capturer toutes les cellules (th ou td) de la ligne
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.S|re.I)
                cell_texts = []
                for cell in cells:
                    # Supprimer tout tag HTML restant dans la cellule
                    text = re.sub(r"<[^>]+>", "", cell)
                    # Échapper les '|' qui pourraient gêner le Markdown
                    text = text.replace("|", "\\|").strip()
                    cell_texts.append(text)
                table_data.append(cell_texts)
            if not table_data:
                continue
            # Construire le Markdown du tableau (format à pipes)
            md_lines = []
            # En-tête si première ligne du tableau original contenait <th> ou est marquée explicitement
            header_idx = 0
            first_row = rows[0] if rows else ""
            has_header = bool(re.search(r"<th", first_row, flags=re.I))
            if has_header:
                # Utiliser la première ligne comme en-tête
                header = table_data[0]
                md_lines.append("| " + " | ".join(header) + " |")
                md_lines.append("|" + "|".join([" --- "]*len(header)) + "|")
                header_idx = 1
            # Lignes de données
            for r in table_data[header_idx:]:
                md_lines.append("| " + " | ".join(r) + " |")
            table_md = "\n".join(md_lines)
            page_blocks.append(table_md)
        elif btype == "Image":
            # Bloc image sans texte : insérer image base64 si configuré
            img = block.get("img")
            if not img:
                continue
            im_pil = img if isinstance(img, Image.Image) else Image.fromarray(img)
            if EMBED_IMAGES == "none":
                continue  # images ignorées
            # Encoder en base64 et insérer
            im_resized = _pil_resize_max(im_pil, IMG_MAX_WIDTH)
            data_uri = _pil_to_base64(im_resized, IMG_FORMAT, IMG_JPEG_QUALITY)
            img_md = f"![{IMG_ALT_PREFIX} – page {page.number+1}]({data_uri})"
            page_blocks.append(img_md)
    # Séparer les blocs par deux sauts de ligne pour structurer Markdown
    page_md = "\n\n".join([blk for blk in page_blocks if blk.strip()])
    # Nettoyage final Markdown de la page
    page_md = _md_cleanup(page_md)
    return (page_md, text_found)

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    # Renvoie l'UI HTML intégrée (formulaire de conversion)
    return HTMLResponse(r'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>MarkItDown UI</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0f14; --bg2: #0e141b;
      --card: rgba(255,255,255,0.06);
      --card-border: rgba(255,255,255,0.08);
      --text: #e6edf3; --muted: #9fb0bf;
      --accent: #63b3ff; --accent-2: #8f7aff;
      --ok: #58d68d; --err: #ff6b6b;
      --shadow: 0 6px 24px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03);
      --radius: 16px;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; }
    body {
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      color: var(--text); background: 
        radial-gradient(1000px 600px at 20% -10%, #183a58 0%, transparent 60%),
        radial-gradient(900px 500px at 120% 10%, #3b2d6a 0%, transparent 55%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
      background-attachment: fixed;
      line-height: 1.55; padding: 32px 20px 40px;
    }
    h1 { margin: 0 0 .35rem 0; font-size: 1.65rem; }
    .sub { color: var(--muted); font-size: .95rem; margin-bottom: 18px; }
    .container { max-width: 1060px; margin: 0 auto; }
    .card {
      background: var(--card); border: 1px solid var(--card-border);
      border-radius: var(--radius); box-shadow: var(--shadow);
      padding: 18px; margin-top: 16px; backdrop-filter: blur(8px);
    }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    label { font-weight: 600; }
    input[type="text"], input[type="file"], textarea {
      background: rgba(255,255,255,0.03); color: var(--text);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px; padding: 10px 12px; outline: none;
      transition: border .15s ease, box-shadow .15s ease;
    }
    input[type="text"]:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(99,179,255,.15);
    }
    textarea {
      width: 100%; min-height: 280px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: .95rem; resize: vertical;
    }
    button {
      padding: 10px 16px; border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      color: #0b0f14;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      cursor: pointer; font-weight: 700;
      transition: transform .06s ease, filter .15s ease, opacity .2s ease;
    }
    button:hover { filter: brightness(1.08); }
    button:active { transform: translateY(1px); }
    button:disabled { opacity: .55; cursor: not-allowed; filter: none; }
    .btn-ghost {
      background: transparent; color: var(--text);
      border-color: rgba(255,255,255,0.16);
    }
    a#download {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 10px 16px; border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.16);
      text-decoration: none; color: var(--text);
    }
    .tag { background: var(--card); padding: 4px 8px; border-radius: 8px; }
    .filemeta { font-size: .9rem; color: var(--muted); }
    .drop {
      border: 2px dashed rgba(255,255,255,0.16);
      border-radius: 12px; padding: 12px; text-align: center;
      color: var(--muted); font-size: .95rem; cursor: pointer;
      margin-top: 8px;
    }
    .drop.active { background: rgba(255,255,255,0.06); border-color: var(--accent); color: var(--accent); }
    .switch { position: relative; display: inline-block; width: 48px; height: 24px; margin-left: 8px; margin-right: 4px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #777; transition: .2s;
      border-radius: 34px;
    }
    .slider:before {
      position: absolute; content: "";
      height: 16px; width: 16px;
      left: 4px; bottom: 4px;
      background-color: white;
      transition: .2s; border-radius: 50%;
    }
    input:checked + .slider { background-color: var(--accent); }
    input:checked + .slider:before { transform: translateX(24px); }
    .stats { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
    .progress { display: none; width: 100%; background: var(--card); border-radius: 8px; margin-top: 12px; overflow: hidden; }
    .progress .bar { height: 6px; width: 100%; background: var(--accent); animation: progress 2s infinite; }
    @keyframes progress {
      0% { transform: translateX(-100%); }
      50% { transform: translateX(-50%); }
      100% { transform: translateX(0); }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>MarkItDown – Conversion Markdown (OCR PaddleOCR)</h1>
    <p class="sub">Convertit PDF, DOCX, XLSX, CSV en texte Markdown structuré (OCR intégré pour PDF/images scannés).</p>
    <div class="card">
      <label for="file">Fichier à convertir</label>
      <div class="row">
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
        <label for="qualite">Mode qualité (PP-Structure)</label>
        <label class="switch">
          <input id="qualite" type="checkbox" />
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
// Gestion drag & drop
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
// Conversion du document
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
  fd.append("activer_plugins_markitdown", $("plugins").checked ? "true" : "false");
  fd.append("forcer_ocr", $("forceocr").checked ? "true" : "false");
  fd.append("mode_qualite", $("qualite").checked ? "true" : "false");
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
    $("status").textContent = "Conversion terminée.";
  } catch(e) {
    $("status").textContent = "Erreur: " + e.message;
  } finally {
    stopTimer(true);
    $("convert").disabled = false;
    $("progress").style.display = "none";
  }
};
</script>
</body>
</html>''')

@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    activer_plugins_markitdown: bool = Form(False),
    forcer_ocr: bool = Form(False),
    mode_qualite: bool = Form(False)
):
    """
    Conversion d'un fichier en Markdown.
    - Si `forcer_ocr` est activé pour PDF/images, utilise PaddleOCR (mode rapide ou qualité).
    - Sinon, utilise MarkItDown pour conversion standard (docs, PDF textuels).
    """
    try:
        t_start = time.perf_counter()
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Fichier vide")

        # Sauvegarder le fichier uploadé si demandé
        in_path = None
        if SAVE_UPLOADS:
            in_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(in_path, "wb") as f_in:
                f_in.write(content)

        is_pdf = guess_is_pdf(file.filename, file.content_type)
        is_img = guess_is_image(file.filename, file.content_type)
        metadata: Dict[str, Any] = {}

        markdown = ""
        if OCR_ENABLED and forcer_ocr and (is_pdf or is_img):
            # Pipeline OCR (PaddleOCR)
            engine = None
            if is_pdf:
                # Ouvrir le PDF avec PyMuPDF
                doc = fitz.open(stream=content, filetype="pdf")
                try:
                    total_pages = min(doc.page_count, OCR_MAX_PAGES)
                    pages_md: List[str] = []
                    text_extracted = False
                    for p in range(total_pages):
                        page = doc.load_page(p)
                        if mode_qualite:
                            # Mode qualité (PP-Structure)
                            page_md, has_text = ocr_page_quality(page)
                            metadata["engine"] = "paddleocr_ppstructure"
                        else:
                            # Mode rapide (OCR brut)
                            page_md, has_text = ocr_page_fast(page)
                            metadata["engine"] = "paddleocr_fast"
                        pages_md.append(page_md)
                        if has_text:
                            text_extracted = True
                    markdown = "\n\n".join([md for md in pages_md if md.strip()])
                    metadata["pages"] = total_pages
                    # Si aucune page n'a fourni de texte (document purement image sans texte détectable)
                    if not text_extracted:
                        metadata["warnings"] = "Aucun texte OCR n'a été extrait."
                finally:
                    doc.close()
            else:
                # Fichier image isolée (PNG, JPG, etc.)
                pil_img = Image.open(io.BytesIO(content))
                # Mode qualité et rapide identiques pour image unique (on peut appliquer PP-Structure également)
                if mode_qualite:
                    # Créer une page PDF temporaire avec l'image ?
                    # Simplification: utiliser OCR direct car PP-Structure sur image isolée équivaut à OCR direct
                    ocr = get_paddle_ocr()
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    result = ocr.ocr(cv_img, cls=False)
                    text = "\n".join([res[1][0] for res in result if res[1] and res[1][0].strip()])
                    markdown = text.strip()
                    metadata["engine"] = "paddleocr_fast"
                else:
                    ocr = get_paddle_ocr()
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    result = ocr.ocr(cv_img, cls=False)
                    markdown = "\n".join([res[1][0] for res in result if res[1] and res[1][0].strip()]).strip()
                    metadata["engine"] = "paddleocr_fast"
                metadata["pages"] = 1
                # Si du texte a été extrait, on peut ajouter une section OCR
                if markdown:
                    markdown = markdown.strip()
                # Si config images -> embed image si OCR absent ou insuffisant
                if EMBED_IMAGES in ("all", "ocr_only") and not markdown:
                    pil_resized = _pil_resize_max(pil_img, IMG_MAX_WIDTH)
                    data_uri = _pil_to_base64(pil_resized, IMG_FORMAT, IMG_JPEG_QUALITY)
                    markdown += f'\n\n![{IMG_ALT_PREFIX}]({data_uri})\n'
        else:
            # Conversion standard via MarkItDown (hors OCR)
            md_engine = MarkItDown(enable_plugins=activer_plugins_markitdown)
            result = md_engine.convert_stream(io.BytesIO(content), file_name=file.filename)
            markdown = getattr(result, "text_content", "") or ""
            # Récupérer métadonnées ou avertissements éventuels
            metadata.update(getattr(result, "metadata", {}) or {})
            if getattr(result, "warnings", None):
                metadata["warnings"] = result.warnings
            metadata["engine"] = "markitdown"
            # Post-traitement du Markdown (nettoyage)
            markdown = _md_cleanup(markdown)
            # Si fichier image isolé sans OCR forcé, tenter quand même un OCR sur l'image
            if OCR_ENABLED and is_img:
                ocr = get_paddle_ocr()
                pil_img = Image.open(io.BytesIO(content))
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                result = ocr.ocr(cv_img, cls=False)
                ocr_text = "\n".join([res[1][0] for res in result if res[1] and res[1][0].strip()]).strip()
                if ocr_text:
                    markdown += "\n\n# OCR (extrait)\n" + ocr_text
                if EMBED_IMAGES in ("all", "ocr_only") and not ocr_text:
                    try:
                        pil_resized = _pil_resize_max(pil_img, IMG_MAX_WIDTH)
                        data_uri = _pil_to_base64(pil_resized, IMG_FORMAT, IMG_JPEG_QUALITY)
                        markdown += f'\n\n![{IMG_ALT_PREFIX}]({data_uri})\n'
                    except Exception:
                        pass

        # Sauvegarder le Markdown de sortie si demandé
        out_name = f"{os.path.splitext(file.filename)[0]}.md"
        out_path = None
        if SAVE_OUTPUTS:
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write(markdown or "")

        # Durée de traitement
        metadata["duration_sec"] = round(time.perf_counter() - t_start, 3)
        if SAVE_UPLOADS and in_path:
            metadata["saved_input_path"] = in_path
        if SAVE_OUTPUTS and out_path:
            metadata["saved_output_path"] = out_path

        return JSONResponse({
            "filename": file.filename,
            "output_filename": out_name if SAVE_OUTPUTS else None,
            "markdown": markdown,
            "metadata": metadata
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur conversion: {type(e).__name__}: {e}")
    
@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"
