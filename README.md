# 📝 MarkItDown API (Dockerized)

> **Convertisseur universel de documents vers Markdown.**
> Rapide, léger, conteneurisé et doté d'une interface web moderne avec prévisualisation en direct.

## 🚀 Présentation

**MarkItDown API** est une solution autonome permettant de convertir divers formats de fichiers (PDF, DOCX, HTML, Images) en un fichier **Markdown** unique et portable.

Contrairement aux solutions classiques, cette application privilégie la conservation de la mise en page et l'intégration des images directement dans le Markdown (encodage Base64). Cela rend les fichiers de sortie totalement indépendants (pas de dossiers d'images externes).

### ✨ Fonctionnalités clés

* **📄 Support PDF Avancé :** Analyse de la structure du document (titres, paragraphes) via PyMuPDF. Extraction des images et réintégration en Base64 à leur emplacement d'origine.
* **📝 Support DOCX :** Conversion des documents Word via `Mammoth`, avec préservation des images.
* **🌐 Support HTML & Web :** Nettoyage du HTML et conversion en Markdown propre.
* **🖼️ Gestion des Images :** Les images seules sont encapsulées en balises Markdown.
* **🖥️ Interface UI Moderne :**
    * Drag & Drop.
    * Mode Sombre (Dark Mode).
    * **Split View :** Éditeur de code à gauche / Rendu visuel en direct à droite.
* **🐳 Docker Ready :** Déploiement instantané via Docker Compose.

## 🛠️ Installation & Démarrage

### Prérequis

* Docker
* Docker Compose

### Démarrage rapide

1.  **Clonez le dépôt :**
    ```bash
    git clone https://github.com/yoslo59/markitdownAPI.git
    cd markitdown-api
    ```

2.  **Lancez le conteneur :**
    ```bash
    docker compose up -d --build
    ```

3.  **Accédez à l'application :**
    * Ouvrez votre navigateur sur : `http://localhost:5704`

## ⚙️ Configuration

L'application est configurable via les variables d'environnement définies dans le fichier `docker-compose.yml`.

### Variables principales

| Variable | Valeur par défaut | Description |
| :--- | :--- | :--- |
| `SAVE_UPLOADS` | `true` | Sauvegarde les fichiers envoyés dans `/data/uploads`. |
| `SAVE_OUTPUTS` | `true` | Sauvegarde les fichiers Markdown générés dans `/data/outputs`. |
| `UPLOAD_DIR` | `/data/uploads` | Chemin interne du dossier d'upload. |
| `OUTPUT_DIR` | `/data/outputs` | Chemin interne du dossier de sortie. |

### Configuration des Images (Base64)

| Variable | Valeur par défaut | Description |
| :--- | :--- | :--- |
| `IMG_FORMAT` | `png` | Format de conversion des images (`png` ou `jpeg`). |
| `IMG_JPEG_QUALITY` | `85` | Qualité de compression (si format jpeg). |
| `IMG_MAX_WIDTH` | `1400` | Redimensionnement max des images (en px) pour limiter la taille du fichier final. |
| `IMG_ALT_PREFIX` | `Capture` | Préfixe utilisé dans le texte alternatif des images (`![Capture - page 1]...`). |

## 🔌 API Documentation

L'application expose une API REST documentée automatiquement via Swagger UI.

Une fois le conteneur lancé, accédez à la documentation interactive :
👉 **`http://localhost:5704/docs`**

### Endpoint principal

* **POST** `/convert`
    * Convertit un fichier uploadé en Markdown.
    * **Paramètre :** `file` (Multipart/Form-data).
    * **Réponse :** JSON contenant le code Markdown, le nom du fichier et les métadonnées.

## 🏗️ Architecture Technique

L'application repose sur un pipeline de traitement intelligent selon le type de fichier :

1.  **Détection du type MIME :** Le fichier est analysé pour déterminer s'il s'agit d'un PDF, DOCX, HTML ou d'une image.
2.  **Traitement PDF (PyMuPDF) :**
    * Le texte est extrait vectoriellement pour garantir une précision parfaite (pas d'erreurs OCR).
    * Les blocs d'images sont découpés, redimensionnés et convertis en Base64.
    * Les en-têtes et pieds de page répétitifs sont détectés et supprimés automatiquement.
3.  **Traitement DOCX (Mammoth) :**
    * Conversion interne en HTML brut, extraction des images, puis transformation en Markdown via *MarkItDown*.
4.  **Nettoyage :** Le Markdown final subit une passe de nettoyage pour retirer les espaces superflus et normaliser la syntaxe.

## 💻 Développement Local

Si vous souhaitez contribuer ou modifier le code sans Docker :

1.  **Créer un environnement virtuel :**
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
    ```

2.  **Installer les dépendances :**
    ```bash
    pip install markitdown mammoth fastapi uvicorn python-multipart pymupdf pillow
    ```

3.  **Lancer le serveur :**
    ```bash
    uvicorn main:app --reload --port 5704
    ```

## 📜 Licence

Ce projet est sous licence MIT. Vous êtes libre de l'utiliser, le modifier et le distribuer.
