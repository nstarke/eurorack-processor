# Eurorack Processor

[Here is an example of what the output looks like from my rack](https://starkerack.com/)

**Documentation Generator for Eurorack Modules**

This repo is a small Python-based toolkit for generating consistent, repeatable documentation for Eurorack modules from structured inputs (CSV) plus reusable templates/prompts, with optional styling assets.

Repo structure (top-level): `css/`, `csv/`, `prompts/`, `scripts/`, `.claude/skills/`

---

## What it does

- Takes module data from a README.csv-style CSV (built by `find_manuals.py`, or
  hand-written — see `csv/`)
- Uses prompts/templates (in `prompts/`) to format/assemble doc content
- Uses scripts (in `scripts/`) to generate output artifacts
- Applies styling from `css/` when producing web-friendly docs
- Answers ad-hoc questions about your rack using the relevant module manuals
  (`ask.py`), saving each answer as markdown/HTML/PDF with a browsable index
- Finds and downloads the module manuals themselves from a plain list of modules
  or a ModularGrid rack, building the CSV as it goes (`find_manuals.py`)
- Ships Claude Code skills (`/eurorack-import`, `/eurorack-question`) that chain
  the scripts into one-command workflows

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

## Claude Code skills

The repo ships two [Claude Code](https://claude.com/claude-code) skills (in
`.claude/skills/`) that wrap the scripts below as slash commands. Run `claude`
from the repo root and use:

### `/eurorack-import [MODULARGRID_URL]`

Imports a ModularGrid rack: runs `find_manuals.py --rack-url` to build
`README.csv` from the rack's module list and find/download every module's
manual PDF into `manuals/`, then runs `process_manuals.py` (with
`prompts/cheatsheet.txt`) to generate per-module documentation pages into
`output/`. Accepts any public rack URL containing the rack id (data sheet or
rack view link). Finishes with a summary of real manuals downloaded,
product-page fallbacks, modules where no manual was found, and the generated
docs. Safe to re-run — only modules without a valid PDF are (re)processed.

```
/eurorack-import https://modulargrid.net/e/modules_racks/data_sheet/2250471
```

### `/eurorack-question [QUESTION]`

Asks a one-off question about your system: runs `ask.py` against `README.csv`
and the `manuals/` directory, relays the answer, and documents it in `answers/`
as markdown, HTML, and PDF, refreshing the `answers/index.html` listing (later
questions build on previous answers). Requires a prior `/eurorack-import` (or an
existing `README.csv` + `manuals/`).

```
/eurorack-question How do I use the clock input on the 2hp Arp module?
```

Both skills use the `claude` CLI as the LLM backend by default, so they need a
logged-in `claude` (see [Claude Authentication](#claude-authentication) below).

## Usage: `process_manuals.py`

Generates a documentation page (markdown/HTML/PDF) for every module in the CSV by
running a prompt file against each module's manual, plus navigable `index.html`
pages for browsing the output tree.

Supports the same LLM providers as `ask.py` (see below for authentication):
`claude` (the default; Claude Code CLI, default model `claude-fable-5`),
`openai` (reads the API key from `openai.key`), and `codex` (Codex CLI).

```bash
usage: process_manuals.py [-h] --prompt PROMPT --input-csv INPUT_CSV
                          [--manuals-dir MANUALS_DIR]
                          [--output-directory OUTPUT_DIRECTORY]
                          [--workers WORKERS]
                          [--llm-provider {openai,claude,codex}]
                          [--model MODEL] [--key-file KEY_FILE] [--css CSS]
                          [--generate-pdf | --no-generate-pdf]
                          [--generate-html | --no-generate-html]
                          [--pdf-engine PDF_ENGINE]

options:
  -h, --help            show this help message and exit
  --prompt PROMPT       Path to a file containing a prompt to run against all
                        modules/manuals.
  --input-csv INPUT_CSV
                        Path to csv file containing modules and manual file
                        paths
  --manuals-dir MANUALS_DIR
                        Directory to where manuals are initially stored.
                        [default='manuals']
  --output-directory OUTPUT_DIRECTORY
                        Directory to write output files to [default='output']
  --workers WORKERS
  --llm-provider {openai,claude,codex}
                        LLM provider: OpenAI API, Claude Code CLI, or Codex
                        CLI [default='claude']
  --model MODEL         Model override (backend-specific; default claude-
                        fable-5 for claude, gpt-4.1 for openai)
  --key-file KEY_FILE   Path to a file containing an OpenAI API Key; only used
                        with --llm-provider openai [default 'openai.key']
  --css CSS             Optional CSS file for HTML/PDF styling
  --generate-pdf, --no-generate-pdf
  --generate-html, --no-generate-html
  --pdf-engine PDF_ENGINE

```

### Run
```bash
python3 scripts/process_manuals.py --prompt prompts/cheatsheet.txt --input-csv README.csv --llm-provider claude
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
   in-scope module list — to the `answers` output directory, plus HTML and PDF
   versions using the same conversion pipeline (and `--css`/`--generate-html`/
   `--generate-pdf`/`--pdf-engine` options) as `process_manuals.py`
5. Regenerates `answers/index.html`: a table of every answer (newest first) with
   its question, links to each available format (md/html/pdf), in-scope modules,
   and date — like the navigable indexes `process_manuals.py` writes for its
   output
6. Makes sure the top-level `index.html` (in the answers directory's parent)
   links to `answers/index.html` — an existing index gets the link appended to
   its list only if missing; if there is no index yet, a minimal one is created

The output directory must be a dedicated subdirectory: pointing
`--output-directory` at the current directory or a repository root is refused,
since the answer files and `index.html` would be written straight into it
(overwriting any existing `index.html`).

Supported LLM providers:
- `claude` (default) — uses the [Claude Code](https://claude.com/claude-code) CLI
  (`claude` must be on your `PATH`); no API key file needed. Defaults to the
  `claude-fable-5` model.
- `openai` — uses the OpenAI API; reads the API key from `openai.key`. Defaults
  to the `gpt-4.1` model.
- `codex` — uses the [OpenAI Codex](https://developers.openai.com/codex/cli) CLI
  (`codex` must be on your `PATH`); no API key file needed. Uses the CLI's
  configured default model unless `--model` is given.

Passing `--model` with no value lists the known models for the selected provider
and exits, e.g. `ask.py --llm-provider codex --model` — any other model ID the
backend supports may also be passed.

### Claude Authentication

The `claude` provider only supports subscription login (Claude Pro/Max). To sign in,
run `claude` and use the `/login` command — this opens a browser OAuth flow and stores
your credentials locally in `~/.claude/.credentials.json`. Usage is billed against your
subscription.

The scripts check for that credentials file before running; if it doesn't exist, they
exit with instructions to log in first.

### Codex Authentication

The `codex` provider uses ChatGPT subscription login (Plus/Pro/Team/Enterprise). To
sign in, run `codex login` and choose "Sign in with ChatGPT" — this opens a browser
OAuth flow and stores your credentials locally in `~/.codex/auth.json`. Usage is billed
against your subscription.

The scripts check for that auth file before running; if it doesn't exist, they exit
with instructions to log in first.

```bash
usage: ask.py [-h] [--prompt PROMPT] [--input-csv INPUT_CSV]
              [--manuals-dir MANUALS_DIR] [--markdown-dir MARKDOWN_DIR]
              [--output-directory OUTPUT_DIRECTORY]
              [--llm-provider {openai,claude,codex}] [--model [MODEL]]
              [--key-file KEY_FILE] [--max-manuals MAX_MANUALS] [--css CSS]
              [--generate-pdf | --no-generate-pdf]
              [--generate-html | --no-generate-html] [--pdf-engine PDF_ENGINE]

options:
  -h, --help            show this help message and exit
  --prompt PROMPT       The question to answer.
  --input-csv INPUT_CSV
                        Path to csv file containing modules and manual file
                        paths [default='README.csv']
  --manuals-dir MANUALS_DIR
                        Directory where manual PDFs are stored
                        [default='manuals']
  --markdown-dir MARKDOWN_DIR
                        Directory searched recursively for previous answers
                        involving the in-scope modules [default: the answers
                        output directory]
  --output-directory OUTPUT_DIRECTORY
                        Directory to write the answer markdown to
                        [default='answers']
  --llm-provider {openai,claude,codex}
                        LLM provider: OpenAI API, Claude Code CLI, or Codex
                        CLI [default='claude']
  --model [MODEL]       Model override (backend-specific; default claude-
                        fable-5 for claude, gpt-4.1 for openai). Pass --model
                        with no value to list known models for the selected
                        provider.
  --key-file KEY_FILE   Path to a file containing an OpenAI API Key [default
                        'openai.key']
  --max-manuals MAX_MANUALS
                        Maximum number of manual PDFs to attach [default=10]
  --css CSS             Optional CSS file for HTML/PDF styling
                        [default='css/basic.css']
  --generate-pdf, --no-generate-pdf
  --generate-html, --no-generate-html
  --pdf-engine PDF_ENGINE
```

### Run
```bash
python3 scripts/ask.py --prompt "How do I use the clock input on the 2hp Arp module?" \
  --input-csv ../eurorack-manuals-repo/README.csv
```

Notes:
- If `--input-csv` / `--manuals-dir` are not given, `README.csv` and the `manuals`
  directory in the current working directory are used.
- If more than `--max-manuals` manuals are in scope, only the first N are attached
  (the dropped ones are logged).
- Answer files are named after the question plus a timestamp, e.g.
  `answers/how-do-i-use-the-clock-input-on-the-2hp-arp-module-20260811-143221.md`,
  with the `.html` and `.pdf` versions written alongside under the same name

## Usage: `find_manuals.py`

Builds the manuals collection itself: given a list of modules, it finds each
module's manual PDF on the internet, downloads it, and writes a CSV in the same
format as `eurorack-manuals-repo/README.csv`:

```csv
"manufacturer","module","quantity","manual file name"
"2hp","Pluck",1,"2hp_Pluck_Manual.pdf"
"Make Noise","MATHS",1,"Make_Noise_MATHS_Manual.pdf"
```

The other scripts (`process_manuals.py`, `ask.py`) consume this CSV and the
downloaded PDFs.

### Prerequisites

- The `claude` or `codex` CLI on your `PATH`, logged in — authentication works
  the same as for `ask.py` (see above). The LLM is used to research each
  module's manual URL and product page via web search.
- Chrome/Chromium (optional, recommended) — used to save product pages as PDFs
  when no manual exists; falls back to weasyprint if not installed.

### 1. Start from a list of modules

Write one module per line — free text or `manufacturer,module`. Blank lines and
`#` comments are ignored, and listing a module twice sets its `quantity` to 2:

```
# modules.txt
Make Noise Maths
2hp,Pluck
Mutable Instruments Plaits
Mutable Instruments Plaits
```

Then run:

```bash
python3 scripts/find_manuals.py --modules modules.txt \
  --output-csv README.csv --manuals-dir manuals
```

`--modules -` reads the list from stdin instead of a file.

### 2. Or start from an existing CSV

`--input-csv` accepts a README.csv-style CSV and fills in its gaps: every row
whose `manual file name` is missing, empty, or not a valid PDF on disk gets
(re)processed, while rows with valid manuals are left untouched. The header row
and the fourth column are both optional, so a bare
`"manufacturer","module",quantity` listing works too.

By default the CSV is updated in place and manuals are looked up/downloaded in
the `manuals` directory of the current working directory. When no input flag is
given at all, `README.csv` in the current directory is used, so completing an
existing collection is just:

```bash
python3 scripts/find_manuals.py
# or, for a CSV elsewhere:
python3 scripts/find_manuals.py --input-csv ../eurorack-manuals-repo/README.csv \
  --manuals-dir ../eurorack-manuals-repo
```

### 3. Or start from a ModularGrid rack

`--rack-url` accepts a link to a ModularGrid rack and processes every module in
it. Any rack URL containing the rack id works — e.g. a data sheet link
(`https://modulargrid.net/e/modules_racks/data_sheet/2250471`) or a rack view
link (`https://modulargrid.net/e/racks/view/2250471`). The module list
(manufacturer, name, and quantity for duplicated modules) is read from the
rack's public view page, so no ModularGrid login is needed — but the rack must
be public, not private.

```bash
python3 scripts/find_manuals.py \
  --rack-url "https://modulargrid.net/e/modules_racks/data_sheet/2250471" \
  --output-csv README.csv --manuals-dir manuals
```

### How it finds a manual

For each module it tries, in order:

1. **Manual PDF** — the LLM CLI (`claude -p` with WebSearch, or
   `codex exec --search`) researches the module and returns candidate manual
   PDF URLs plus the product page URL; each candidate is downloaded and
   validated as a real PDF (HTML error pages and dead links are rejected).
2. **Product page as PDF** — if no manual PDF is found, the product web page is
   saved as a PDF (headless Chrome/Chromium, falling back to weasyprint). These
   are named `..._Product_Page.pdf` so they're distinguishable from real manuals.
3. **archive.org** — as a last resort, the archive.org item library is searched
   for a matching PDF, and the Wayback Machine is checked for a snapshot of the
   product page.

Modules where all three approaches fail still get a CSV row with an empty
`manual file name`, and are listed in a summary at the end of the run.

### Resuming and re-running

The CSV is rewritten after every module, and modules that already have a valid
PDF on disk are skipped (without spending an LLM call), so an interrupted or
partially failed run can simply be re-run — only the missing modules are
processed. Delete a module's PDF (or blank its CSV entry) to force a re-fetch.

### Options

```bash
usage: find_manuals.py [-h]
                       [--modules MODULES | --input-csv INPUT_CSV | --rack-url RACK_URL]
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
                        column are optional. [default: 'README.csv' when
                        neither --modules nor --rack-url is given]
  --rack-url RACK_URL   ModularGrid rack URL (e.g. https://modulargrid.net/e/m
                        odules_racks/data_sheet/2250471); the rack's module
                        list is fetched from its public view page.
  --output-csv OUTPUT_CSV
                        CSV to create/update, in eurorack-manuals-repo
                        README.csv format [default: --input-csv if given, else
                        'README.csv']
  --manuals-dir MANUALS_DIR
                        Directory to download manual PDFs into
                        [default='manuals']
  --llm-provider {claude,codex}
                        LLM CLI used to research manual URLs [default='claude']
  --model MODEL         Model override (backend-specific)
```
