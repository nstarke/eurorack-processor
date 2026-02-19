# Eurorack Processor

**Documentation Generator for Eurorack Modules** :contentReference[oaicite:0]{index=0}

This repo is a small Python-based toolkit for generating consistent, repeatable documentation for Eurorack modules from structured inputs (CSV) plus reusable templates/prompts, with optional styling assets.

> Repo structure (top-level): `css/`, `csv/`, `prompts/`, `scripts/` :contentReference[oaicite:1]{index=1}  
> Primary language: Python (with some CSS) :contentReference[oaicite:2]{index=2}

---

## What it does

- Takes module data (typically from `csv/`) :contentReference[oaicite:3]{index=3}
- Uses prompts/templates (in `prompts/`) to format/assemble doc content :contentReference[oaicite:4]{index=4}
- Uses scripts (in `scripts/`) to generate output artifacts :contentReference[oaicite:5]{index=5}
- Applies styling from `css/` when producing web-friendly docs :contentReference[oaicite:6]{index=6}

Typical outputs you might generate:
- A “module manual” page (HTML/Markdown)
- Spec sheets
- Tables for I/O, ranges, calibration notes, etc.
- Assets ready to publish on GitHub Pages or bundle with releases

---

## Quick start

### 1) Clone
```bash
git clone https://github.com/nstarke/eurorack-processor.git
cd eurorack-processor
# eurorack-processor

**Documentation Generator for Eurorack Modules** :contentReference[oaicite:0]{index=0}

This repo is a small Python-based toolkit for generating consistent, repeatable documentation for Eurorack modules from structured inputs (CSV) plus reusable templates/prompts, with optional styling assets.

> Repo structure (top-level): `css/`, `csv/`, `prompts/`, `scripts/` :contentReference[oaicite:1]{index=1}  
> Primary language: Python (with some CSS) :contentReference[oaicite:2]{index=2}

---

## What it does

- Takes module data (typically from `csv/`) :contentReference[oaicite:3]{index=3}
- Uses prompts/templates (in `prompts/`) to format/assemble doc content :contentReference[oaicite:4]{index=4}
- Uses scripts (in `scripts/`) to generate output artifacts :contentReference[oaicite:5]{index=5}
- Applies styling from `css/` when producing web-friendly docs :contentReference[oaicite:6]{index=6}

Typical outputs you might generate:
- A “module manual” page (HTML/Markdown)
- Spec sheets
- Tables for I/O, ranges, calibration notes, etc.
- Assets ready to publish on GitHub Pages or bundle with releases

---

## Quick start

### 1) Clone
```bash
git clone https://github.com/nstarke/eurorack-processor.git
cd eurorack-processor
```

### 2) Create a virtualenv (recommended)
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```bash
pip install -r scripts/requirements.txt
```

## Usage

```bash
python3 scripts/process_manuals.py -h
usage: process_manuals.py [-h] --prompt PROMPT --csv CSV
                          [--manuals-dir MANUALS_DIR]
                          [--output-directory OUTPUT_DIRECTORY]
                          [--workers WORKERS] [--model MODEL]
                          [--key-file KEY_FILE] [--css CSS]
                          [--generate-pdf | --no-generate-pdf]
                          [--generate-html | --no-generate-html]
                          [--pdf-engine PDF_ENGINE]

options:
  -h, --help            show this help message and exit
  --prompt PROMPT       Path to a file containing a prompt to run against all
                        modules/manuals.
  --csv CSV             Path to csv file containing modules and manual file
                        paths
  --manuals-dir MANUALS_DIR
                        manuals directory [default='manuals']
  --output-directory OUTPUT_DIRECTORY
                        directory to write output files to [default='output']
  --workers WORKERS
  --model MODEL
  --key-file KEY_FILE   Path to a file containing an OpenAI API Key [default
                        'openai.key']
  --css CSS             Optional CSS file for HTML/PDF styling
  --generate-pdf, --no-generate-pdf
  --generate-html, --no-generate-html
  --pdf-engine PDF_ENGINE

```

