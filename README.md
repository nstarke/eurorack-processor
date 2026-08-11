# Eurorack Processor

[Here is an example of what the output looks like from my rack](https://starkerack.com/)

**Documentation Generator for Eurorack Modules**

This repo is a small Python-based toolkit for generating consistent, repeatable documentation for Eurorack modules from structured inputs (CSV) plus reusable templates/prompts, with optional styling assets.

Repo structure (top-level): `css/`, `csv/`, `prompts/`, `scripts/`

---

## What it does

- Takes module data (typically from `csv/`) 
- Uses prompts/templates (in `prompts/`) to format/assemble doc content
- Uses scripts (in `scripts/`) to generate output artifacts
- Applies styling from `css/` when producing web-friendly docs
- Answers ad-hoc questions about your rack using the relevant module manuals (`ask.py`)

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


## Usage: `process_manuals.py`

Generates a documentation page (markdown/HTML/PDF) for every module in the CSV by
running a prompt file against each module's manual.

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

## Usage: `ask.py`

Answers a one-off question about your system. Unlike `process_manuals.py` (which runs
one prompt against *every* module), `ask.py` takes the question directly on the command
line and only consults the manuals that are relevant:

1. Asks the LLM which modules from the CSV are "in scope" for the question
2. Finds the corresponding manual PDFs via the CSV's `manual file name` column,
   plus any previous answers in the `answers` directory whose "Modules In Scope"
   list involves an in-scope module (override the search location with
   `--markdown-dir`)
3. Submits the question plus those manuals and previous answers to the LLM
   provider
4. Writes the answer as a markdown file — including the original question and the
   in-scope module list — to the `answers` output directory

Supported LLM providers:
- `openai` (default) — uses the OpenAI API; reads the API key from `openai.key`
- `claude` — uses the [Claude Code](https://claude.com/claude-code) CLI (`claude` must
  be on your `PATH`); no API key file needed
- `codex` — uses the [OpenAI Codex](https://developers.openai.com/codex/cli) CLI
  (`codex` must be on your `PATH`); no API key file needed

### Claude Authentication

The `claude` provider only supports subscription login (Claude Pro/Max). To sign in,
run `claude` and use the `/login` command — this opens a browser OAuth flow and stores
your credentials locally in `~/.claude/.credentials.json`. Usage is billed against your
subscription.

`ask.py` checks for that credentials file before running; if it doesn't exist, the
script exits with instructions to log in first.

### Codex Authentication

The `codex` provider uses ChatGPT subscription login (Plus/Pro/Team/Enterprise). To
sign in, run `codex login` and choose "Sign in with ChatGPT" — this opens a browser
OAuth flow and stores your credentials locally in `~/.codex/auth.json`. Usage is billed
against your subscription.

`ask.py` checks for that auth file before running; if it doesn't exist, the script
exits with instructions to log in first.

```bash
usage: ask.py [-h] --prompt PROMPT --csv CSV [--manuals-dir MANUALS_DIR]
              [--markdown-dir MARKDOWN_DIR]
              [--output-directory OUTPUT_DIRECTORY]
              [--llm-provider {openai,claude,codex}] [--model MODEL]
              [--key-file KEY_FILE] [--max-manuals MAX_MANUALS]

options:
  -h, --help            show this help message and exit
  --prompt PROMPT       The question to answer.
  --csv CSV             Path to csv file containing modules and manual file
                        paths (e.g. README.csv)
  --manuals-dir MANUALS_DIR
                        Directory where manual PDFs are stored [default: the
                        CSV's directory]
  --markdown-dir MARKDOWN_DIR
                        Directory searched recursively for previous answers
                        involving the in-scope modules [default: the answers
                        output directory]
  --output-directory OUTPUT_DIRECTORY
                        Directory to write the answer markdown to
                        [default='answers']
  --llm-provider {openai,claude,codex}
                        LLM provider: OpenAI API, Claude Code CLI, or Codex
                        CLI [default='openai']
  --model MODEL         Model override (backend-specific; default gpt-4.1 for
                        openai)
  --key-file KEY_FILE   Path to a file containing an OpenAI API Key [default
                        'openai.key']
  --max-manuals MAX_MANUALS
                        Maximum number of manual PDFs to attach [default=10]
```

### Run
```bash
python3 scripts/ask.py --prompt "How do I use the clock input on the 2hp Arp module?" \
  --csv ../eurorack-manuals-repo/README.csv --llm-provider claude
```

Notes:
- If `--manuals-dir` is not given, PDFs are looked up in the same directory as the CSV.
- If more than `--max-manuals` manuals are in scope, only the first N are attached
  (the dropped ones are logged).
- Answer files are named after the question plus a timestamp, e.g.
  `answers/how-do-i-use-the-clock-input-on-the-2hp-arp-module-20260811-143221.md`

## Usage: `find_manuals.py`

Builds the manuals collection itself: takes either a newline-delimited list of
modules (`--modules`) or an existing README.csv-style CSV (`--input-csv`), finds
each module's manual PDF on the internet, downloads it, and writes a CSV in the
same format as `eurorack-manuals-repo/README.csv`
(`"manufacturer","module",quantity,"manual file name"`).

With `--input-csv`, the fourth "manual file name" column (and the header row) are
optional — the script iterates over every row and only (re)processes the ones
whose manual value is missing, empty, or not a valid PDF on disk. By default the
CSV is updated in place and manuals are looked up/downloaded in the CSV's own
directory, so filling in the gaps of an existing collection is just:

```bash
python3 scripts/find_manuals.py --input-csv ../eurorack-manuals-repo/README.csv
```

For each module it tries, in order:

1. **Manual PDF** — an LLM CLI (`claude -p` with WebSearch, or `codex exec --search`)
   researches the module and returns candidate manual PDF URLs plus the product page
   URL; each candidate is downloaded and validated as a real PDF.
2. **Product page as PDF** — if no manual PDF is found, the product web page is saved
   as a PDF (headless Chrome/Chromium, falling back to weasyprint).
3. **archive.org** — as a last resort, the archive.org item library is searched for a
   matching PDF, and the Wayback Machine is checked for a snapshot of the product page.

Input lines may be free text (`Make Noise Maths`) or pre-parsed
(`Make Noise,Maths`); duplicate lines increment the module's `quantity`. Modules
already in the output CSV with a valid PDF on disk are skipped, and the CSV is
rewritten after each module, so interrupted runs can simply be re-run.

Authentication works the same as `ask.py`: `claude` uses your Claude Pro/Max
subscription login, `codex` uses your ChatGPT subscription login (see above).

```bash
usage: find_manuals.py [-h] (--modules MODULES | --input-csv INPUT_CSV)
                       [--output-csv OUTPUT_CSV] [--manuals-dir MANUALS_DIR]
                       [--llm-provider {claude,codex}] [--model MODEL]

options:
  -h, --help            show this help message and exit
  --modules MODULES     Newline-delimited list of modules ('-' reads stdin).
                        Lines may be free text ('Make Noise Maths') or
                        'manufacturer,module'.
  --input-csv INPUT_CSV
                        Existing README.csv-style CSV; rows whose 'manual file
                        name' column is missing, empty, or not a valid PDF on
                        disk are (re)processed. Header row and the fourth
                        column are optional.
  --output-csv OUTPUT_CSV
                        CSV to create/update, in eurorack-manuals-repo
                        README.csv format [default: --input-csv if given, else
                        'README.csv']
  --manuals-dir MANUALS_DIR
                        Directory to download manual PDFs into [default: the
                        --input-csv's directory if given, else 'manuals']
  --llm-provider {claude,codex}
                        LLM CLI used to research manual URLs [default='claude']
  --model MODEL         Model override (backend-specific)
```

### Run
```bash
python3 scripts/find_manuals.py --modules modules.txt \
  --output-csv ../eurorack-manuals-repo/README.csv \
  --manuals-dir ../eurorack-manuals-repo \
  --llm-provider claude
```

Modules where nothing could be found are written to the CSV with an empty
`manual file name` and listed at the end of the run.
