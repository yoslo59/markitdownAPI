import os
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from markitdown import MarkItDown
from openai import AzureOpenAI


# ---------------------------
# Config via variables d'env
# ---------------------------
SAVE_UPLOADS  = os.getenv("SAVE_UPLOADS", "false").lower() == "true"
SAVE_OUTPUTS  = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"
UPLOAD_DIR    = os.getenv("UPLOAD_DIR", "/data/uploads")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "/data/outputs")

AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_KEY        = os.getenv("AZURE_OPENAI_KEY", "").strip()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

# Dossiers persistants (si activés)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# App FastAPI
# ---------------------------
app = FastAPI(title="MarkItDown API", version="1.2")

# CORS (utile si front séparé)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def get_azure_client() -> Optional[AzureOpenAI]:
    """Retourne un client AzureOpenAI si correctement configuré, sinon None."""
    if AZURE_ENDPOINT and AZURE_KEY:
        return AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version=AZURE_API_VER
        )
    return None


# ---------------------------
# Mini interface web
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    # UI simple pour uploader un fichier et lancer la conversion
    return """
<!doctype html>
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
  </style>
</head>
<body>
  <h1>MarkItDown — Conversion</h1>
  <p class="muted">Upload un document (PDF, DOCX, PPTX, XLSX, HTML, etc.) → Markdown. Optionnel : résumé via Azure OpenAI.</p>

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

  try{
    const res = await fetch(endpoint, { method:"POST", body: fd });
    if(!res.ok){ throw new Error("HTTP "+res.status); }
    const json = await res.json();
    $("md").value = json.markdown || "";
    $("meta").value = JSON.stringify(json.metadata || {}, null, 2);

    // Download du MD
    const blob = new Blob([$("md").value], {type:"text/markdown;charset=utf-8"});
    const url  = URL.createObjectURL(blob);
    const a = $("download");
    a.href = url;
    a.download = (json.output_filename || "sortie.md");
    a.style.display = "inline-block";
    $("status").textContent = "OK";
  }catch(e){
    $("status").textContent = "Erreur : " + e.message;
  }finally{
    $("convert").disabled = false;
  }
};
</script>
</body>
</html>
    """


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
):
    """
    Reçoit un fichier, exécute MarkItDown et renvoie le Markdown + métadonnées.
    Optionnel: résumé Azure OpenAI si `use_llm=true` et que l'Azure est configuré.
    """

    # Instanciation MarkItDown (plugins/Document Intelligence en option)
    md = MarkItDown(
        enable_plugins=use_plugins,
        docintel_endpoint=docintel_endpoint
    )

    # Lecture du contenu
    content = await file.read()

    # Sauvegarde d'entrée si demandé
    in_path = None
    if SAVE_UPLOADS:
        in_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(in_path, "wb") as f:
            f.write(content)

    stream = io.BytesIO(content)
    result = md.convert_stream(stream, file_name=file.filename)
    markdown = result.text_content or ""

    # Sauvegarde de sortie si demandé
    out_name = f"{os.path.splitext(file.filename)[0]}.md"
    out_path = None
    if SAVE_OUTPUTS:
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown)

    # Résumé via Azure (optionnel, robuste : n'échoue pas la conversion si erreur)
    azure_summary = None
    if use_llm:
        client = get_azure_client()
        if client:
            try:
                # on tronque le contenu pour éviter des requêtes trop volumineuses
                snippet = markdown[:12000]
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "Tu es un assistant qui résume des documents techniques en français, de manière concise et structurée."},
                        {"role": "user", "content": f"Résume le document suivant en 10 points maximum, avec un titre en H1 et des sous-titres:\n\n{snippet}"}
                    ],
                    temperature=0.2,
                    max_tokens=800
                )
                azure_summary = resp.choices[0].message.content
            except Exception as e:
                azure_summary = f"[Erreur Azure OpenAI: {type(e).__name__}: {e}]"
        else:
            azure_summary = "[Azure OpenAI non configuré]"

    # Métadonnées enrichies
    metadata = result.metadata or {}
    if in_path:  metadata["saved_input_path"]  = in_path
    if out_path: metadata["saved_output_path"] = out_path
    if use_llm:  metadata["azure_summary"]     = azure_summary

    return JSONResponse({
        "filename": file.filename,
        "output_filename": out_name if SAVE_OUTPUTS else None,
        "markdown": markdown,
        "metadata": metadata,
    })


# ---------------------------
# Healthcheck
# ---------------------------
@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"
