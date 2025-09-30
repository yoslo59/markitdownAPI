import os
import io
import re
import base64
import time
from typing import Optional, Tuple, List

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
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "25"))
OCR_MIN_CHARS      = int(os.getenv("OCR_MIN_CHARS", "500"))
OCR_MODE           = os.getenv("OCR_MODE", "append").strip()
OCR_KEEP_SPACES    = os.getenv("OCR_KEEP_SPACES", "true").lower() == "true"
OCR_TWO_PASS       = os.getenv("OCR_TWO_PASS", "true").lower() == "true"
OCR_TABLE_MODE     = os.getenv("OCR_TABLE_MODE", "true").lower() == "true"
OCR_PSMS           = [p.strip() for p in os.getenv("OCR_PSMS", "6,4,11").split(",")]     # 6=block, 4=columns, 11=sparse
OCR_DPI_CANDIDATES = [int(x) for x in os.getenv("OCR_DPI_CANDIDATES", "300,350,400").split(",")]

# Short-circuit : si on atteint ce score, on arrête la recherche multi-DPI/PSM pour gagner du temps
OCR_SCORE_GOOD_ENOUGH = float(os.getenv("OCR_SCORE_GOOD_ENOUGH", "0.6"))

# Embedding images base64
# none      : n’embarque aucune image
# ocr_only  : embarque les images uniquement si l’OCR n’est pas exploitable (court/bruyant)
# all       : embarque toutes les images du PDF
EMBED_IMAGES       = os.getenv("EMBED_IMAGES", "ocr_only").strip()  # none | ocr_only | all
IMG_FORMAT         = os.getenv("IMG_FORMAT", "png").strip().lower()  # png | jpeg
IMG_JPEG_QUALITY   = int(os.getenv("IMG_JPEG_QUALITY", "85"))
IMG_MAX_WIDTH      = int(os.getenv("IMG_MAX_WIDTH", "1400"))         # resize max (px), 0 = no limit
IMG_ALT_PREFIX     = os.getenv("IMG_ALT_PREFIX", "Capture").strip()

# (Optionnel) Azure Document Intelligence
DEFAULT_DOCINTEL_ENDPOINT = os.getenv("DEFAULT_DOCINTEL_ENDPOINT", "").strip()

# Dossiers persistants
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="1.9")

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
# Helpers OCR
# ---------------------------
_table_chars = re.compile(r"[|+\-=_]{3,}")  # heuristique ASCII

def _tess_config(psm: str, keep_spaces: bool, table_mode: bool) -> str:
    cfg = f"--psm {psm} --oem 1"  # LSTM
    if keep_spaces:
        cfg += " -c preserve_interword_spaces=1"
    if table_mode:
        cfg += " -c tessedit_write_images=false"
    return cfg

def _preprocess_for_ocr(im: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    # binarisation douce
    g = g.point(lambda p: 255 if p > 190 else (0 if p < 110 else p))
    return g

def _score_text_for_table(txt: str) -> float:
    """Score simple favorisant tableaux/monospace, pénalise bruit évident."""
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
    """Emballe blocs ASCII en ```text``` pour garder l’alignement Markdown."""
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

def _ocr_image_best(im: Image.Image, langs: str) -> Tuple[str, float]:
    """Essaie plusieurs PSM, 2 passes (brute + prétraitée). Short-circuit si score OK."""
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    best_txt, best_score = "", -1e9
    for psm in OCR_PSMS:
        cfg = _tess_config(psm, OCR_KEEP_SPACES, OCR_TABLE_MODE)
        # Pass brute
        t1 = pytesseract.image_to_string(im, lang=langs, config=cfg) or ""
        s1 = _score_text_for_table(t1)
        cand_txt, cand_score = t1, s1
        # Pass prétraitée
        if OCR_TWO_PASS:
            im2 = _preprocess_for_ocr(im)
            t2 = pytesseract.image_to_string(im2, lang=langs, config=cfg) or ""
            s2 = _score_text_for_table(t2)
            if s2 > cand_score:
                cand_txt, cand_score = t2, s2
        if cand_score > best_score:
            best_txt, best_score = cand_txt, cand_score
        # short-circuit si suffisant
        if best_score >= OCR_SCORE_GOOD_ENOUGH:
            break
    return best_txt.strip(), best_score

def ocr_image_bytes(img_bytes: bytes, langs: str) -> Tuple[str, float]:
    with Image.open(io.BytesIO(img_bytes)) as im:
        txt, score = _ocr_image_best(im, langs)
        return _wrap_tables_as_code(txt), score

def _raster_pdf_page(page, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_pdf_bytes(pdf_bytes: bytes, langs: str, dpi: int, max_pages: int) -> Tuple[str, int, List[float]]:
    """Multi-DPI/PSM per page, returns (markdown_text, pages_done, scores_list)."""
    out = []
    pages_done = 0
    scores: List[float] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = doc.page_count
        for i in range(min(total_pages, max_pages)):
            page = doc.load_page(i)
            best_txt, best_score = "", -1e9
            # multi-DPI par page avec short-circuit
            for d in OCR_DPI_CANDIDATES:
                im = _raster_pdf_page(page, d)
                txt, score = _ocr_image_best(im, langs)
                if score > best_score:
                    best_txt, best_score = txt, score
                if best_score >= OCR_SCORE_GOOD_ENOUGH:
                    break
            scores.append(best_score)
            if best_txt.strip():
                out.append(f"\n\n## Page {i+1}\n\n{_wrap_tables_as_code(best_txt)}")
            pages_done += 1
    finally:
        doc.close()
    return ("\n".join(out).strip(), pages_done, scores)

# ---------------------------
# Extraction/embedding images
# ---------------------------
def _pil_resize_max(im: Image.Image, max_w: int) -> Image.Image:
    if max_w and im.width > max_w:
        ratio = max_w / im.width
        new_h = int(im.height * ratio)
        return im.resize((max_w, new_h), Image.LANCZOS)
    return im

def _pil_to_base64(im: Image.Image, fmt: str = "png", quality: int = 85) -> str:
    buf = io.BytesIO()
    if fmt.lower() == "jpeg" or fmt.lower() == "jpg":
        im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def extract_pdf_images_as_md(doc: fitz.Document, page_index: int) -> str:
    """Extrait les images natives d'une page PDF et les retourne en Markdown base64."""
    md_parts = []
    page = doc.load_page(page_index)
    imgs = page.get_images(full=True)
    if not imgs:
        return ""
    for idx, img in enumerate(imgs, start=1):
        xref = img[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n >= 4:  # CMYK/with alpha -> convert to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            im = _pil_resize_max(im, IMG_MAX_WIDTH)
            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
            md_parts.append(f'![{IMG_ALT_PREFIX} p{page_index+1}-{idx}]({data_uri})')
        except Exception:
            continue
    return "\n\n".join(md_parts)

def should_embed_images_for_page(ocr_txt: str, ocr_score: float) -> bool:
    if EMBED_IMAGES == "all":
        return True
    if EMBED_IMAGES == "none":
        return False
    # ocr_only
    if not ocr_txt or len(ocr_txt.strip()) < OCR_MIN_CHARS or ocr_score < OCR_SCORE_GOOD_ENOUGH:
        return True
    return False

# ---------------------------
# Mini interface web (NOUVELLE UI + chrono)
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
    file: UploadFile = File(...),
    use_plugins: bool = Form(False),
    docintel_endpoint: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),   # ignoré pour Azure; gardé pour compat
    use_llm: bool = Form(False),
    force_ocr: bool = Form(False),
):
    """
    Convertit le fichier avec MarkItDown.
    Fallback OCR (Tesseract) si le texte est pauvre, ou si force_ocr=true.
    Peut embarquer des images PDF en base64 selon EMBED_IMAGES.
    Optionnel: résumé Azure OpenAI (use_llm=true).
    """
    t0 = time.time()
    try:
        if not docintel_endpoint:
            docintel_endpoint = DEFAULT_DOCINTEL_ENDPOINT

        md = MarkItDown(
            enable_plugins=use_plugins,
            docintel_endpoint=docintel_endpoint
        )

        content = await file.read()

        # Save input
        in_path = None
        if SAVE_UPLOADS:
            in_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(in_path, "wb") as f:
                f.write(content)

        # MarkItDown
        stream = io.BytesIO(content)
        result = md.convert_stream(stream, file_name=file.filename)

        markdown = getattr(result, "text_content", "") or ""
        metadata = getattr(result, "metadata", None) or {}
        warnings = getattr(result, "warnings", None)
        if warnings:
            metadata["warnings"] = warnings

        # OCR fallback & images
        is_pdf = guess_is_pdf(file.filename, file.content_type)
        is_img = guess_is_image(file.filename, file.content_type)

        ocr_scores_per_page: List[float] = []
        pdf_images_md_per_page: List[str] = []

        if OCR_ENABLED and (force_ocr or (len(markdown.strip()) < OCR_MIN_CHARS and (is_pdf or is_img))):
            if is_pdf:
                ocr_text, pages_done, ocr_scores_per_page = ocr_pdf_bytes(content, OCR_LANGS, OCR_DPI, OCR_MAX_PAGES)
                metadata["ocr_pages"] = pages_done
                metadata["ocr_langs"] = OCR_LANGS
                metadata["ocr_dpi"] = OCR_DPI

                # Embedding d'images PDF par page si demandé
                if EMBED_IMAGES != "none":
                    try:
                        doc = fitz.open(stream=content, filetype="pdf")
                        for i in range(min(doc.page_count, OCR_MAX_PAGES)):
                            md_imgs = extract_pdf_images_as_md(doc, i)
                            pdf_images_md_per_page.append(md_imgs)
                        doc.close()
                    except Exception:
                        pass

                # Assemblage Markdown + images (page par page)
                if ocr_text.strip():
                    if OCR_MODE == "append" and markdown.strip():
                        markdown += "\n\n# OCR (extrait)\n" + ocr_text
                    else:
                        if len(markdown.strip()) < OCR_MIN_CHARS:
                            markdown = f"# OCR\n{ocr_text}"
                        else:
                            markdown += "\n\n# OCR (extrait)\n" + ocr_text

                    # Si images à intégrer
                    if EMBED_IMAGES != "none" and pdf_images_md_per_page:
                        md_lines = []
                        pages = ocr_text.split("\n\n## Page ")
                        # reconstruire en insérant images après chaque page OCR si condition
                        # pages[0] peut contenir "# OCR" + début, gérer proprement
                        for idx, chunk in enumerate(pages):
                            if not chunk.strip():
                                continue
                            # rétablir l’entête si nécessaire
                            if idx == 0 and chunk.startswith("# OCR"):
                                md_lines.append(chunk.strip())
                                continue
                            prefix = "## Page " if idx > 0 else ""
                            if prefix:
                                md_lines.append(prefix + chunk.strip())
                            # insérer images pour cette page si utile
                            page_num = None
                            try:
                                # chunk commence souvent par "N\n\n...", on récupère N
                                first_line = chunk.splitlines()[0].strip()
                                page_num = int(first_line.split()[0]) if first_line and first_line[0].isdigit() else None
                            except Exception:
                                page_num = None
                            # fallback : use idx
                            if page_num is None:
                                page_num = idx
                            # decide embed
                            if 1 <= page_num <= len(ocr_scores_per_page):
                                score = ocr_scores_per_page[page_num - 1]
                                imgs_md = pdf_images_md_per_page[page_num - 1] if page_num - 1 < len(pdf_images_md_per_page) else ""
                                if imgs_md and should_embed_images_for_page(chunk, score):
                                    md_lines.append("\n\n### Captures\n" + imgs_md)
                        markdown = "\n\n".join(md_lines) if md_lines else markdown

                else:
                    metadata["ocr_note"] = "OCR tenté mais aucun texte détecté."
                    # si on n’a pas de texte mais EMBED_IMAGES=ocr_only|all, on peut tout de même pousser les images
                    if EMBED_IMAGES != "none" and pdf_images_md_per_page:
                        blocks = []
                        for i, md_imgs in enumerate(pdf_images_md_per_page, start=1):
                            if md_imgs:
                                blocks.append(f"## Page {i}\n\n### Captures\n{md_imgs}")
                        if blocks:
                            markdown += ("\n\n# Images extraites\n" + "\n\n".join(blocks))

            elif is_img:
                ocr_text, score = ocr_image_bytes(content, OCR_LANGS)
                if ocr_text.strip():
                    if OCR_MODE == "append" and markdown.strip():
                        markdown += "\n\n# OCR (extrait)\n" + ocr_text
                    else:
                        if len(markdown.strip()) < OCR_MIN_CHARS:
                            markdown = ocr_text
                        else:
                            markdown += "\n\n# OCR (extrait)\n" + ocr_text
                else:
                    metadata["ocr_note"] = "OCR tenté mais aucun texte détecté."

                # image seule : option d’embed si ocr_only ou all
                if EMBED_IMAGES in ("ocr_only", "all"):
                    try:
                        with Image.open(io.BytesIO(content)) as im:
                            im = _pil_resize_max(im, IMG_MAX_WIDTH)
                            data_uri = _pil_to_base64(im, IMG_FORMAT, IMG_JPEG_QUALITY)
                            if EMBED_IMAGES == "all" or (EMBED_IMAGES == "ocr_only" and (not ocr_text or score < OCR_SCORE_GOOD_ENOUGH)):
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

        # mesure durée
        duration_ms = int((time.time() - t0) * 1000)
        metadata["duration_ms"] = duration_ms

        return JSONResponse({
            "filename": file.filename,
            "output_filename": out_name if SAVE_OUTPUTS else None,
            "markdown": markdown,
            "metadata": metadata,
            "duration_ms": duration_ms,
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
