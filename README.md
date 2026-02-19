# Eurorack Processor

**Documentation Generator for Eurorack Modules**

This repo is a small Python-based toolkit for generating consistent, repeatable documentation for Eurorack modules from structured inputs (CSV) plus reusable templates/prompts, with optional styling assets.

Repo structure (top-level): `css/`, `csv/`, `prompts/`, `scripts/`

---

## What it does

- Takes module data (typically from `csv/`) 
- Uses prompts/templates (in `prompts/`) to format/assemble doc content
- Uses scripts (in `scripts/`) to generate output artifacts
- Applies styling from `css/` when producing web-friendly docs

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

### 4) Install OpenAI API Key
By default the script looks for an OpenAI API key in a file `openai.key`.


## Usage

```bash
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
                        Directory to where manuals are initially stored.
                        [default='manuals']
  --output-directory OUTPUT_DIRECTORY
                        Directory to write output files to [default='output']
  --workers WORKERS
  --model MODEL
  --key-file KEY_FILE   Path to a file containing an OpenAI API Key [default
                        'openai.key']
  --css CSS             Optional CSS file for HTML/PDF styling
  --generate-pdf, --no-generate-pdf
  --generate-html, --no-generate-html
  --pdf-engine PDF_ENGINE

```

### Run
```bash
python3 scripts/process_manuals.py --prompt prompts/cheatsheet.txt --csv csv/MODULES.csv
```
