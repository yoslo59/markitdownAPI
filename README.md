# MarkItDown API — Conversion de documents vers Markdown (avec OCR & images base64)

**MarkItDown API** expose un service HTTP pour convertir des documents (PDF, DOCX, PPTX, XLSX, HTML, etc.) en **Markdown**.  
Il inclut :
- Fallback **OCR** (Tesseract) pour les PDF/images scannés (multi-DPI, multi-PSM, prétraitement, détection de tableaux ASCII).
- **Intégration d’images en base64** (configurable) pour conserver les captures si l’OCR est faible.
- **Résumé optionnel** via **Azure OpenAI** (si configuré).
- Une **mini-UI web** pour tester l’upload et la conversion.

> Objectif : obtenir un Markdown exploitable rapidement, même lorsque les sources sont des captures d’écran ou des scans “durs”.

---

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Architecture & Composants](#architecture--composants)
- [Déploiement rapide (Docker Compose)](#déploiement-rapide-docker-compose)
- [Build & run locaux](#build--run-locaux)
- [Variables d’environnement](#variables-denvironnement)
- [Volumes & persistance](#volumes--persistance)
- [API](#api)
- [Mini-UI intégrée](#mini-ui-intégrée)
- [OCR — Détails & réglages](#ocr--détails--réglages)
- [Images base64 — Stratégies](#images-base64--stratégies)
- [Azure OpenAI — Résumé](#azure-openai--résumé)
- [Performance : conseils](#performance--conseils)
- [Sécurité](#sécurité)
- [Dépannage](#dépannage)
- [Feuille de route](#feuille-de-route)
- [Licence](#licence)

---

## Fonctionnalités

- Conversion **vers Markdown** à partir de formats bureautiques et web.
- **Fallback OCR** pour PDF/images :
  - Rendu multi-DPI (300/350/400), multi-PSM (6/4/11), **prétraitement** (niveaux de gris, contraste, sharpening, binarisation douce).
  - Détection des **tableaux ASCII** & encapsulation en blocs ```text``` (préserve l’alignement).
  - **Short-circuit** : stoppe les essais dès que la qualité est “suffisante”.
- **Images en base64** (PNG/JPEG) :
  - `none` / `ocr_only` / `all` (au besoin ou systématique).
  - Redimensionnement max configurable (`IMG_MAX_WIDTH`).
- **Résumé Azure OpenAI** (optionnel).
- **Mini-UI** intégrée pour tester l’API (upload, options, download du `.md`).

---

## Prérequis

- Docker / Docker Compose.
- Accès réseau sortant si usage d’Azure OpenAI (facultatif).
- Pour l’OCR :
  - Paquets Tesseract & langues : `tesseract-ocr`, `tesseract-ocr-fra`, `tesseract-ocr-eng`.

> Le **Dockerfile** installe tout ce qu’il faut (Tesseract + packages Python : `markitdown[all]`, `pymupdf`, `pytesseract`, `pillow`, `fastapi`, …).

---

## Architecture & Composants

- **FastAPI** : endpoints `/convert`, `/health`, `/config` + mini-UI.
- **MarkItDown** : conversion “structurée” → Markdown.
- **Tesseract OCR** : fallback sur scans/screenshots.
- **PyMuPDF** : rasterisation des pages PDF & extraction des images natives.
- **Azure OpenAI** (optionnel) : résumé court.

---

## Déploiement rapide (Docker Compose)

`docker-compose.yml` :

```yaml
services:
  markitdown-api:
    image: markitdown-api:local
    container_name: markitdown-api
    ports:
      - "5704:5704"
    environment:
      ENABLE_PLUGINS: "false"
      SAVE_UPLOADS: "true"
      SAVE_OUTPUTS: "true"
      UPLOAD_DIR: "/data/uploads"
      OUTPUT_DIR: "/data/outputs"

      # Azure OpenAI (optionnel)
      AZURE_OPENAI_ENDPOINT: ""   # ex: https://<resource>.openai.azure.com/
      AZURE_OPENAI_KEY: ""
      AZURE_OPENAI_DEPLOYMENT: ""
      AZURE_OPENAI_API_VERSION: ""

      # OCR
      OCR_ENABLED: "true"
      OCR_DPI: "350"
      OCR_MAX_PAGES: "25"
      OCR_MIN_CHARS: "500"
      OCR_PSMS: "6,4,11"
      OCR_DPI_CANDIDATES: "300,350,400"
      OCR_SCORE_GOOD_ENOUGH: "0.6"

      # Images base64
      EMBED_IMAGES: "ocr_only"    # none | ocr_only | all
      IMG_FORMAT: "png"
      IMG_JPEG_QUALITY: "85"
      IMG_MAX_WIDTH: "1400"
      IMG_ALT_PREFIX: "Capture"

    volumes:
      - md_uploads:/data/uploads
      - md_output:/data/outputs

    restart: unless-stopped

volumes:
  md_uploads:
  md_output:
```

## Build & run locaux

### 1) Build de l’image

```bash
docker build -t markitdown-api:local .
```

### 2) Lancer via Compose

```bash
docker compose up -d
```

OU sans Compose :

```bash
docker run -d --name markitdown-api   -p 5704:5704   -e SAVE_UPLOADS=true -e SAVE_OUTPUTS=true   -v md_uploads:/data/uploads -v md_output:/data/outputs   markitdown-api:local
```

---

## Variables d’environnement

| Variable | Default | Description |
|---|---:|---|
| `ENABLE_PLUGINS` | `false` | Active les plugins MarkItDown si besoin. |
| `SAVE_UPLOADS` | `false` | Sauvegarde le fichier source sur `/data/uploads`. |
| `SAVE_OUTPUTS` | `false` | Sauvegarde le `.md` final sur `/data/outputs`. |
| `UPLOAD_DIR` / `OUTPUT_DIR` | `/data/...` | Dossiers persistants. |
| `AZURE_OPENAI_ENDPOINT` | `""` | URL Azure OpenAI (ex. `https://<res>.openai.azure.com/`). |
| `AZURE_OPENAI_KEY` | `""` | Clé API Azure OpenAI. |
| `AZURE_OPENAI_DEPLOYMENT` | `""` | **Nom du déploiement** (ex. `gpt-4o-mini`). |
| `AZURE_OPENAI_API_VERSION` | `""` | Ex. `2025-01-01-preview`. |
| `OCR_ENABLED` | `true` | Active le fallback OCR. |
| `OCR_LANGS` | `fra+eng` | Langues Tesseract. |
| `OCR_DPI` | `350` | DPI par défaut. |
| `OCR_MAX_PAGES` | `25` | Limite page OCR. |
| `OCR_MIN_CHARS` | `500` | Seuil “texte pauvre” pour déclencher OCR. |
| `OCR_MODE` | `append` | `append` ou `replace_when_empty`. |
| `OCR_KEEP_SPACES` | `true` | Préserve les espaces (tableaux ASCII). |
| `OCR_TWO_PASS` | `true` | Passe brute + passe prétraitée. |
| `OCR_TABLE_MODE` | `true` | Favorise la structure ASCII. |
| `OCR_PSMS` | `6,4,11` | PSM Tesseract testés (6=block, 4=columns, 11=sparse). |
| `OCR_DPI_CANDIDATES` | `300,350,400` | DPI testés par page. |
| `OCR_SCORE_GOOD_ENOUGH` | `0.6` | Seuil “assez bon” → stoppe les essais. |
| `EMBED_IMAGES` | `ocr_only` | `none` / `ocr_only` / `all`. |
| `IMG_FORMAT` | `png` | `png` ou `jpeg`. |
| `IMG_JPEG_QUALITY` | `85` | Qualité JPEG si `IMG_FORMAT=jpeg`. |
| `IMG_MAX_WIDTH` | `1400` | Redimensionnement max (px), 0 = off. |
| `IMG_ALT_PREFIX` | `Capture` | Préfixe alt Markdown. |
| `DEFAULT_DOCINTEL_ENDPOINT` | `""` | (Optionnel) Azure Document Intelligence. |

---

## Volumes & persistance

- `md_uploads` → `/data/uploads` : copies des fichiers envoyés si `SAVE_UPLOADS=true`.
- `md_output`  → `/data/outputs` : fichiers `.md` exportés si `SAVE_OUTPUTS=true`.

---

## API

### 1) Convertir un document

`POST /convert` (multipart/form-data)

**Champs :**
- `file` *(obligatoire)* : fichier à convertir.
- `use_plugins` *(bool, défaut `false`)* : active les plugins MarkItDown.
- `docintel_endpoint` *(string, optionnel)* : Azure Document Intelligence (meilleur OCR/structure).
- `use_llm` *(bool, défaut `false`)* : résume via Azure OpenAI si configuré.
- `force_ocr` *(bool, défaut `false`)* : force l’OCR même si du texte est détecté.

**Réponse :**
```json
{
  "filename": "doc.pdf",
  "output_filename": "doc.md",
  "markdown": "...",
  "metadata": {
    "warnings": null,
    "ocr_pages": 4,
    "ocr_langs": "fra+eng",
    "ocr_dpi": 350,
    "azure_summary": "..."
  }
}
```

**Exemples cURL :**

```bash
# Conversion simple
curl -F "file=@/path/to/file.pdf" http://localhost:5704/convert > out.json

# Forcer OCR + résumer
curl -F "file=@/path/to/file.pdf"      -F "force_ocr=true"      -F "use_llm=true"      http://localhost:5704/convert
```

### 2) Healthcheck

`GET /health` → `"ok"`

### 3) Config (UI)

`GET /config` → `{ "docintel_default": "<endpoint ou vide>" }`

---

## Mini-UI intégrée

- Accessible sur `http://<host>:5704/`
- Upload de fichier, options (`plugins`, `force_ocr`, `use_llm`), champ **Azure Document Intelligence**, bouton **Convertir**, zone Markdown + métadonnées, bouton **Télécharger Markdown**.

---

## OCR — Détails & réglages

- **Multi-DPI** & **Multi-PSM** : améliore la reconnaissance sur **screenshots compressés** et **tableaux ASCII**.
- **Prétraitement** : niveaux de gris, auto-contraste, sharpening, binarisation douce → meilleure lisibilité pour Tesseract.
- **Détection de tableaux** : blocs ASCII détectés et encapsulés dans ```text``` pour éviter la casse d’alignement par Markdown.
- **Short-circuit** : si un rendu atteint `OCR_SCORE_GOOD_ENOUGH`, on arrête les essais (gain de temps).

**Réglages conseillés (compromis qualité/vitesse) :**
- Si c’est **lent** : `OCR_PSMS=6` et `OCR_DPI_CANDIDATES=300`.  
- Si c’est **bruité** : garde `OCR_PSMS=6,4`, `OCR_DPI_CANDIDATES=300,350`.

---

## Images base64 — Stratégies

- `EMBED_IMAGES=none` : pas d’images → Markdown minimal.
- `EMBED_IMAGES=ocr_only` : **idéal** — on insère les images **seulement** quand l’OCR est peu exploitable (court/bruyant).
- `EMBED_IMAGES=all` : toutes les images → fichiers plus lourds, mais fidélité visuelle maximale.

Format : `PNG` recommandé (garde mieux les traits).  
Largeur : `IMG_MAX_WIDTH=1400` (confortable).  
Pour économiser : `IMG_FORMAT=jpeg` + `IMG_JPEG_QUALITY=80`.

---

## Azure OpenAI — Résumé

- Renseigne `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`.
- Coche “**Résumé Azure LLM**” dans la mini-UI, ou envoie `use_llm=true` en POST.
- Le service **n’échoue pas** si Azure est mal configuré : un message explicite apparaît dans `metadata.azure_summary`.

---

## Performance : conseils

- **I/O** : monte des volumes sur SSD si possible (`/data/uploads`, `/data/outputs`).
- **OCR** : réduire les combinaisons testées :  
  `OCR_PSMS=6` et `OCR_DPI_CANDIDATES=300` → **le plus rapide**.  
- **Timeout côté client** : pour des PDF volumineux, ajuste les délais.
- **Mise à l’échelle** : plusieurs réplicas derrière un reverse proxy si nécessaire.

---

## Dépannage

- **Mini-UI : bouton Convertir inactif**  
  → Vide le cache navigateur (Ctrl/Cmd+Shift+R).  
- **Azure : erreurs `unsupported_parameter` (max_tokens/temperature)**  
  → L’API Azure **chat completions** attend `max_completion_tokens` et ne permet pas certains params selon modèles. Ici, c’est déjà géré (pas de `temperature` non supportée).
- **OCR trop bruyant**  
  → Passe `OCR_PSMS=6,4` et `OCR_DPI_CANDIDATES=300,350`.  
  → Garde `IMG_FORMAT=png`.
- **OCR trop lent**  
  → `OCR_PSMS=6` + `OCR_DPI_CANDIDATES=300` + `OCR_SCORE_GOOD_ENOUGH=0.7`.

---

## Feuille de route

- Option **Azure Document Intelligence** plus “intégrée” (si endpoint fourni).
- Post-traitement **ASCII → tables Markdown** (conversion automatique des blocs ```text``` en tables `| col |`).
- Export **.md + assets** en archive ZIP (images en fichiers au lieu de base64).

---

## Licence

MIT

---

## Auteur

- Yoslo59. Contributions bienvenues : PRs, issues, suggestions.

---
