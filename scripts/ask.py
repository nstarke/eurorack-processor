#!/usr/bin/env python3
"""
Ask a question about your eurorack system.

Flow:
  1. Take a question ("prompt") as a CLI argument.
  2. Ask the LLM backend which modules from the CSV are in scope for the question.
  3. Find the corresponding manual PDFs via the CSV ("manual file name" column),
     plus any previous answers in the answers directory whose "Modules In Scope"
     list involves an in-scope module.
  4. Submit the question plus those manuals and previous answers to the backend.
  5. Write the answer as a markdown file into the answers output directory, plus
     HTML and PDF versions (same conversion pipeline as process_manuals.py).
  6. Regenerate the answers directory's index.html listing every answer.
  7. Link the answers index from the top-level index.html next to the answers
     directory — only if answers/index.html exists and isn't already linked
     (creating a minimal top-level index if missing).
"""

import argparse
import csv
import datetime
import html as htmllib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from process_manuals import convert_md_to_html, convert_md_to_pdf


# ---------- helpers ----------

def load_openai_key(key_path: Path) -> str:
    if not key_path.exists():
        raise FileNotFoundError(f"OpenAI key file not found: {key_path}")

    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"OpenAI key file is empty: {key_path}")

    return key


def read_csv_rows(csv_path: Path, required_fields: list[str]) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)

        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            first_line = sample.splitlines()[0] if sample else ""
            lowered = first_line.lower()
            has_header = any(field.lower() in lowered for field in required_fields)

        reader = csv.DictReader(f) if has_header else csv.DictReader(f, fieldnames=required_fields)
        return list(reader)


def is_probably_pdf(path: Path) -> tuple[bool, str]:
    try:
        if not path.exists():
            return False, "file does not exist"
        if not path.is_file():
            return False, "not a file"
        if path.stat().st_size < 8:
            return False, "file too small"
        with path.open("rb") as f:
            head = f.read(5)
        if head != b"%PDF-":
            return False, f"missing PDF header (%PDF-), got {head!r}"
        return True, "ok"
    except Exception as e:
        return False, f"exception while checking PDF: {e}"


def answer_scoped_modules(text: str) -> set[str]:
    """Parse the '## Modules In Scope' list out of a previously written answer."""
    modules: set[str] = set()
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower() == "## modules in scope":
            in_section = True
        elif in_section:
            if stripped.startswith("- "):
                modules.add(stripped[2:].strip().lower())
            elif stripped.startswith("#") or stripped.startswith("---"):
                break
    return modules


def find_markdown_docs(scoped: list[dict], answers_dir: Path) -> list[Path]:
    """Find previous answers in the answers directory involving any in-scope module."""
    if not answers_dir.exists():
        return []
    targets = {
        f"{row['manufacturer'].strip()} {row['module'].strip()}".lower() for row in scoped
    }
    docs: list[Path] = []
    for md in sorted(answers_dir.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if targets & answer_scoped_modules(text):
            docs.append(md)
    return docs


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:max_len].rstrip("-") or "answer"


def extract_json_array(text: str) -> list:
    """Pull the first JSON array out of an LLM response (tolerates code fences/prose)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"No JSON array found in response:\n{text}")
        text = text[start : end + 1]
    return json.loads(text)


# ---------- index generation ----------

def parse_answer_question(text: str) -> str | None:
    """Pull the question text out of an answer file's '# Question' section."""
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower() == "# question":
            in_section = True
        elif in_section:
            if stripped.startswith("#") or stripped.startswith("---"):
                break
            if stripped:
                return stripped
    return None


def parse_answer_modules(text: str) -> list[str]:
    """Like answer_scoped_modules, but preserves the original casing for display."""
    modules: list[str] = []
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower() == "## modules in scope":
            in_section = True
        elif in_section:
            if stripped.startswith("- "):
                modules.append(stripped[2:].strip())
            elif stripped.startswith("#") or stripped.startswith("---"):
                break
    return modules


def write_answers_index(answers_dir: Path):
    """
    Regenerate <answers_dir>/index.html listing every answer, newest first, with
    its question, links to each available format (md/html/pdf), in-scope modules,
    and date.
    """
    entries = []
    for md in sorted(answers_dir.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        question = parse_answer_question(text) or md.stem
        modules = sorted(parse_answer_modules(text), key=str.lower)
        match = re.search(r"(\d{8})-(\d{6})$", md.stem)
        if match:
            date, time = match.group(1), match.group(2)
            timestamp = f"{date[:4]}-{date[4:6]}-{date[6:]} {time[:2]}:{time[2:4]}"
        else:
            timestamp = ""
        formats = {
            kind: md.with_suffix(f".{kind}").relative_to(answers_dir).as_posix()
            for kind in ("md", "html", "pdf")
            if md.with_suffix(f".{kind}").exists()
        }
        entries.append((timestamp, question, modules, formats))

    entries.sort(key=lambda e: e[0], reverse=True)

    def link(kind: str, href: str | None) -> str:
        if href is None:
            return '<span style="color:#999">—</span>'
        return f'<a href="{htmllib.escape(href)}">{htmllib.escape(kind)}</a>'

    if entries:
        rows = []
        for timestamp, question, modules, formats in entries:
            # Link the question to the friendliest available format.
            main_href = formats.get("html") or formats.get("md")
            rows.append(
                "<tr>"
                f'<td><a href="{htmllib.escape(main_href)}">{htmllib.escape(question)}</a></td>'
                f"<td>{link('md', formats.get('md'))}</td>"
                f"<td>{link('html', formats.get('html'))}</td>"
                f"<td>{link('pdf', formats.get('pdf'))}</td>"
                f"<td>{htmllib.escape(', '.join(modules))}</td>"
                f"<td>{htmllib.escape(timestamp)}</td>"
                "</tr>"
            )
        answers_table = f"""
<table>
  <thead>
    <tr>
      <th>Question</th>
      <th>md</th>
      <th>html</th>
      <th>pdf</th>
      <th>Modules In Scope</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
"""
    else:
        answers_table = "<p><em>No answers yet.</em></p>"

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Answers</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.6rem; }}
    th {{ background: #f6f6f6; text-align: left; }}
    a {{ text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Answers</h1>
  {answers_table}
</body>
</html>
"""
    (answers_dir / "index.html").write_text(html_doc, encoding="utf-8")


def index_links_to(text: str, href: str) -> bool:
    """
    True if the HTML in `text` already links to `href` (e.g. 'answers/index.html'),
    accepting equivalent forms such as './answers/index.html', '/answers/index.html',
    'answers/', or 'answers'.
    """
    dir_href = href.removesuffix("/index.html")
    accepted = {href, dir_href, dir_href + "/"}
    for match in re.finditer(r"""href\s*=\s*["']([^"']+)["']""", text, re.IGNORECASE):
        target = htmllib.unescape(match.group(1)).strip()
        target = target.split("#", 1)[0].split("?", 1)[0]
        while target.startswith("./"):
            target = target[2:]
        target = target.lstrip("/")
        if target in accepted:
            return True
    return False


def update_top_level_index(answers_dir: Path) -> Path | None:
    """
    Ensure the index.html in the answers directory's parent (the site root) links
    to <answers_dir>/index.html. Only acts when <answers_dir>/index.html actually
    exists. Creates a minimal index if none exists; an existing index is only
    modified if it doesn't already contain a link to the answers index (in any
    equivalent form, e.g. './answers/index.html' or 'answers/').
    Skipped entirely when the answers directory is itself the top level (the
    current working directory or a repository root) — its own index.html already
    is the top-level index, and the parent would be outside the project.
    Returns the path of the created/changed index, or None.
    """
    answers_dir = answers_dir.resolve()
    if answers_dir == Path.cwd().resolve() or (answers_dir / ".git").exists():
        return None
    if not (answers_dir / "index.html").exists():
        return None
    index_path = answers_dir.parent / "index.html"
    href = f"{answers_dir.name}/index.html"
    link = f'<li><a href="{htmllib.escape(href)}">{htmllib.escape(answers_dir.name)}</a></li>'

    if not index_path.exists():
        html_doc = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Index</title></head>
<body>
<h1>Index</h1>
<ul>{link}</ul>
</body>
</html>
"""
        index_path.write_text(html_doc, encoding="utf-8")
        return index_path

    text = index_path.read_text(encoding="utf-8")
    if index_links_to(text, href):
        return None
    if "</ul>" in text:
        text = text.replace("</ul>", f"{link}</ul>", 1)
    elif "</body>" in text:
        text = text.replace("</body>", f"<ul>{link}</ul>\n</body>", 1)
    else:
        text += f"\n<ul>{link}</ul>\n"
    index_path.write_text(text, encoding="utf-8")
    return index_path


# ---------- backends ----------

# Sentinel for a bare `--model` with no value: list models instead of running.
LIST_MODELS = "<list>"

# Common model choices per provider, shown by a bare `--model` (no value).
# Any other model ID the backend supports may also be passed.
KNOWN_MODELS = {
    "claude": [
        ("claude-fable-5", "Claude Fable 5 (default)"),
        ("claude-opus-5", "Claude Opus 5"),
        ("claude-sonnet-5", "Claude Sonnet 5"),
        ("claude-haiku-4-5", "Claude Haiku 4.5"),
        ("fable / opus / sonnet / haiku", "aliases for the latest model of each tier"),
    ],
    "codex": [
        ("gpt-5.1-codex", "GPT-5.1 Codex"),
        ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini"),
        ("gpt-5.1", "GPT-5.1"),
        ("gpt-5-codex", "GPT-5 Codex"),
        ("gpt-5", "GPT-5"),
    ],
    "openai": [
        ("gpt-4.1", "GPT-4.1 (default)"),
        ("gpt-4.1-mini", "GPT-4.1 mini"),
        ("gpt-4o", "GPT-4o"),
        ("gpt-4o-mini", "GPT-4o mini"),
    ],
}


def print_known_models(provider: str) -> None:
    print(f"Known models for --llm-provider {provider}:")
    for model_id, description in KNOWN_MODELS[provider]:
        print(f"  {model_id:<30} {description}")
    print("Any other model ID the backend supports may also be passed to --model.")

class OpenAIBackend:
    def __init__(self, key_file: Path, model: str | None):
        from openai import OpenAI

        self.client = OpenAI(api_key=load_openai_key(key_file))
        self.model = model or "gpt-4.1"

    def complete_text(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
        return response.output_text

    def answer_with_documents(self, prompt: str, pdf_paths: list[Path], md_paths: list[Path]) -> str:
        content = []
        for pdf in pdf_paths:
            with pdf.open("rb") as f:
                uploaded = self.client.files.create(file=f, purpose="user_data")
            content.append({"type": "input_file", "file_id": uploaded.id})
        # Markdown is plain text, so it goes inline rather than as a file upload.
        for md in md_paths:
            text = md.read_text(encoding="utf-8", errors="replace")
            content.append({
                "type": "input_text",
                "text": f"--- Previous answer document: {md.name} ---\n\n{text}",
            })
        content.append({"type": "input_text", "text": prompt})

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
        )
        return response.output_text


class ClaudeBackend:
    """Uses the Claude Code CLI (`claude -p`) with a subscription login; no API key file needed."""

    CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"

    def __init__(self, model: str | None):
        if shutil.which("claude") is None:
            raise FileNotFoundError(
                "`claude` CLI not found on PATH. "
                "Install Claude Code: https://claude.com/claude-code"
            )
        if not self.CREDENTIALS_FILE.exists():
            raise RuntimeError(
                f"Claude Code subscription login not found ({self.CREDENTIALS_FILE} does not exist).\n"
                "Run `claude` and use the /login command to sign in with your "
                "Claude Pro/Max subscription, then re-run this script."
            )
        self.model = model or "claude-fable-5"

    def _run(self, prompt: str, extra_args: list[str] | None = None) -> str:
        cmd = ["claude", "-p"]
        if self.model:
            cmd.extend(["--model", self.model])
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI failed:\n{result.stderr}")
        return result.stdout.strip()

    def complete_text(self, prompt: str) -> str:
        return self._run(prompt)

    def answer_with_documents(self, prompt: str, pdf_paths: list[Path], md_paths: list[Path]) -> str:
        all_paths = pdf_paths + md_paths
        pdf_list = "\n".join(f"- {p.resolve()}" for p in pdf_paths)
        full_prompt = (
            f"{prompt}\n\n"
            f"Read the following manual PDFs before answering:\n{pdf_list}\n"
        )
        if md_paths:
            md_list = "\n".join(f"- {p.resolve()}" for p in md_paths)
            full_prompt += (
                f"\nAlso read these previous question-and-answer documents "
                f"about the modules:\n{md_list}\n"
            )
        add_dirs: list[str] = []
        for parent in {str(p.resolve().parent) for p in all_paths}:
            add_dirs.extend(["--add-dir", parent])
        return self._run(full_prompt, extra_args=["--allowedTools", "Read"] + add_dirs)


class CodexBackend:
    """Uses the OpenAI Codex CLI (`codex exec`) with a ChatGPT subscription login."""

    AUTH_FILE = Path.home() / ".codex" / "auth.json"

    def __init__(self, model: str | None):
        if shutil.which("codex") is None:
            raise FileNotFoundError(
                "`codex` CLI not found on PATH. "
                "Install Codex: https://developers.openai.com/codex/cli"
            )
        if not self.AUTH_FILE.exists():
            raise RuntimeError(
                f"Codex ChatGPT login not found ({self.AUTH_FILE} does not exist).\n"
                "Run `codex login` and choose 'Sign in with ChatGPT' to sign in with "
                "your ChatGPT subscription, then re-run this script."
            )
        self.model = model

    def _run(self, prompt: str) -> str:
        # codex exec prints progress logs to stdout; --output-last-message captures
        # just the agent's final answer.
        with tempfile.NamedTemporaryFile(mode="r", suffix=".md", delete=False) as out:
            out_path = Path(out.name)
        try:
            cmd = [
                "codex", "exec",
                "--sandbox", "read-only",
                "--skip-git-repo-check",
                "--output-last-message", str(out_path),
            ]
            if self.model:
                cmd.extend(["-m", self.model])
            result = subprocess.run(cmd, input=prompt, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"codex CLI failed:\n{result.stderr}")
            answer = out_path.read_text(encoding="utf-8").strip()
            if not answer:
                raise RuntimeError(f"codex CLI returned an empty answer:\n{result.stderr}")
            return answer
        finally:
            out_path.unlink(missing_ok=True)

    def complete_text(self, prompt: str) -> str:
        return self._run(prompt)

    def answer_with_documents(self, prompt: str, pdf_paths: list[Path], md_paths: list[Path]) -> str:
        pdf_list = "\n".join(f"- {p.resolve()}" for p in pdf_paths)
        full_prompt = (
            f"{prompt}\n\n"
            f"Read the following manual PDFs before answering (extract their text with "
            f"a tool such as pdftotext if needed):\n{pdf_list}\n"
        )
        if md_paths:
            md_list = "\n".join(f"- {p.resolve()}" for p in md_paths)
            full_prompt += (
                f"\nAlso read these previous question-and-answer documents "
                f"about the modules:\n{md_list}\n"
            )
        return self._run(full_prompt)


# ---------- output naming ----------

FILENAME_TEMPLATE = """Come up with a short, descriptive filename for a document answering
the following question about a eurorack modular synthesizer system:

{question}

Respond with ONLY the filename: 3-6 lowercase words separated by hyphens, no file
extension, no other text. Example: patching-krell-with-maths
"""


def generate_filename(backend, question: str) -> str:
    """Ask the LLM for a descriptive filename slug; fall back to slugifying the prompt."""
    try:
        response = backend.complete_text(FILENAME_TEMPLATE.format(question=question))
        # Take the last non-empty line in case the model adds prose, then sanitize.
        lines = [line.strip().strip("`") for line in response.splitlines() if line.strip()]
        slug = slugify(lines[-1]) if lines else ""
        if slug and slug != "answer":
            return slug
        print(f"[WARN] LLM returned an unusable filename ({response!r}); using the prompt instead")
    except Exception as e:
        print(f"[WARN] Filename generation failed ({e}); using the prompt instead")
    return slugify(question)


# ---------- scoping ----------

SCOPING_TEMPLATE = """You are helping answer a question about a eurorack modular synthesizer system.

Here is the full list of modules in the system, as "manufacturer,module" lines:

{module_list}

Question:
{question}

Which modules are in scope for this question? Select every module that is directly
mentioned, or that belongs to a category the question is about (e.g. "filters",
"sequencers", "delays"). If the question is about the whole system, select all modules.

Respond with ONLY a JSON array of objects, each with "manufacturer" and "module" keys,
copied exactly from the list above. Respond with [] if no modules are relevant.
"""


def determine_scope(backend, question: str, rows: list[dict]) -> list[dict]:
    module_list = "\n".join(
        f"{row['manufacturer'].strip()},{row['module'].strip()}" for row in rows
    )
    response = backend.complete_text(
        SCOPING_TEMPLATE.format(module_list=module_list, question=question)
    )
    selected = extract_json_array(response)

    by_key = {
        (row["manufacturer"].strip().lower(), row["module"].strip().lower()): row
        for row in rows
    }
    matched = []
    for item in selected:
        key = (str(item.get("manufacturer", "")).strip().lower(),
               str(item.get("module", "")).strip().lower())
        row = by_key.get(key)
        if row is None:
            print(f"[WARN] Scoped module not found in CSV: {item}")
        else:
            matched.append(row)
    return matched


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Answer a question using in-scope module manuals.")

    parser.add_argument("--prompt",
                        help="The question to answer.")
    parser.add_argument("--input-csv", type=Path, default=Path("README.csv"),
                        help="Path to csv file containing modules and manual file paths "
                             "[default='README.csv']")
    parser.add_argument("--manuals-dir", type=Path, default=Path("manuals"),
                        help="Directory where manual PDFs are stored [default='manuals']")
    parser.add_argument("--markdown-dir", type=Path, default=None,
                        help="Directory searched recursively for previous answers involving "
                             "the in-scope modules [default: the answers output directory]")
    parser.add_argument("--output-directory", type=Path, default=Path("answers"),
                        help="Directory to write the answer markdown to [default='answers']")
    parser.add_argument("--llm-provider", choices=["openai", "claude", "codex"], default="claude",
                        help="LLM provider: OpenAI API, Claude Code CLI, or Codex CLI [default='claude']")
    parser.add_argument("--model", nargs="?", const=LIST_MODELS, default=None,
                        help="Model override (backend-specific; default claude-fable-5 for claude, "
                             "gpt-4.1 for openai). Pass --model with no value to list known "
                             "models for the selected provider.")
    parser.add_argument("--key-file", type=Path, default=Path("openai.key"),
                        help="Path to a file containing an OpenAI API Key [default 'openai.key']")
    parser.add_argument("--max-manuals", type=int, default=10,
                        help="Maximum number of manual PDFs to attach [default=10]")
    parser.add_argument("--css", type=Path, default=Path("css/basic.css"),
                        help="Optional CSS file for HTML/PDF styling [default='css/basic.css']")
    parser.add_argument("--generate-pdf", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--generate-html", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pdf-engine")

    args = parser.parse_args()

    if args.model == LIST_MODELS:
        print_known_models(args.llm_provider)
        sys.exit(0)

    # Required unless we exited above to list models.
    if args.prompt is None:
        parser.error("the following arguments are required: --prompt")
    if not args.input_csv.exists():
        sys.exit(f"[ERROR] CSV not found: {args.input_csv}")

    out_dir = args.output_directory.resolve()
    if out_dir == Path.cwd().resolve() or (out_dir / ".git").exists():
        sys.exit(
            f"[ERROR] --output-directory resolves to {out_dir}, which is the current "
            "directory or a repository root. Answers and index.html would be written "
            "straight into it (overwriting any existing index.html); use a dedicated "
            "subdirectory such as 'answers' instead."
        )

    manuals_dir = args.manuals_dir

    try:
        if args.llm_provider == "openai":
            backend = OpenAIBackend(args.key_file, args.model)
        elif args.llm_provider == "claude":
            backend = ClaudeBackend(args.model)
        else:
            backend = CodexBackend(args.model)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as e:
        sys.exit(f"[ERROR] {e}")

    rows = read_csv_rows(args.input_csv, ["manufacturer", "module", "quantity", "manual file name"])
    if not rows:
        sys.exit(f"No module rows found in {args.input_csv}")

    print(f"[INFO] Determining which of {len(rows)} modules are in scope...")
    scoped = determine_scope(backend, args.prompt, rows)
    if not scoped:
        sys.exit("[ERROR] No modules were determined to be in scope for this question.")

    print(f"[INFO] In scope: " + ", ".join(
        f"{r['manufacturer'].strip()} {r['module'].strip()}" for r in scoped))

    # Collect manual PDFs, deduped (some modules share a manual file)
    pdf_paths: list[Path] = []
    seen: set[str] = set()
    for row in scoped:
        manual_name = row["manual file name"].strip()
        if not manual_name:
            print(f"[WARN] {row['manufacturer']} – {row['module']}: manual missing, skipping")
            continue
        if manual_name in seen:
            continue
        seen.add(manual_name)

        pdf = manuals_dir / manual_name
        ok, reason = is_probably_pdf(pdf)
        if not ok:
            print(f"[WARN] Skipping {pdf}: {reason}")
            continue
        pdf_paths.append(pdf)

    md_paths = find_markdown_docs(scoped, args.markdown_dir or args.output_directory)
    if md_paths:
        print(f"[INFO] Found {len(md_paths)} previous answer(s): "
              + ", ".join(p.name for p in md_paths))

    if not pdf_paths and not md_paths:
        sys.exit("[ERROR] No valid manual PDFs or markdown documents found for the in-scope modules.")

    if len(pdf_paths) > args.max_manuals:
        dropped = pdf_paths[args.max_manuals:]
        pdf_paths = pdf_paths[: args.max_manuals]
        print(f"[WARN] Too many manuals; using first {args.max_manuals}, dropping: "
              + ", ".join(p.name for p in dropped))

    module_names = ", ".join(f"{r['manufacturer'].strip()} {r['module'].strip()}" for r in scoped)
    attachments = "module manuals" if not md_paths else \
        "module manuals and previous question-and-answer documents"
    answer_prompt = (
        f"You are a eurorack modular synthesizer expert. Using the attached {attachments} "
        f"(for: {module_names}), answer the following question. "
        "Format your answer as markdown.\n\n"
        f"Question: {args.prompt}"
    )

    print(f"[INFO] Asking {args.llm_provider} with {len(pdf_paths)} manual(s) "
          f"and {len(md_paths)} markdown document(s)...")
    answer = backend.answer_with_documents(answer_prompt, pdf_paths, md_paths)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    print("[INFO] Generating output filename...")
    name = generate_filename(backend, args.prompt)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.output_directory / f"{name}-{timestamp}.md"

    scoped_lines = "\n".join(
        f"- {r['manufacturer'].strip()} {r['module'].strip()}" for r in scoped
    )
    header = (
        f"# Question\n\n"
        f"{args.prompt.strip()}\n\n"
        f"## Modules In Scope\n\n"
        f"{scoped_lines}\n\n"
        f"---\n\n"
        f"# Answer\n\n"
    )
    out_path.write_text(header + answer + "\n", encoding="utf-8")
    print(f"[OK] Answer written to {out_path}")

    css = args.css if args.css and args.css.exists() else None
    if args.css and css is None:
        print(f"[WARN] CSS file not found, styling skipped: {args.css}")

    if args.generate_html:
        html_path = out_path.with_suffix(".html")
        try:
            convert_md_to_html(out_path, html_path, css)
            print(f"[OK] HTML written to {html_path}")
        except Exception as e:
            print(f"[WARN] HTML generation failed: {e}")

    if args.generate_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        try:
            convert_md_to_pdf(out_path, pdf_path, css, args.pdf_engine)
            print(f"[OK] PDF written to {pdf_path}")
        except Exception as e:
            print(f"[WARN] PDF generation failed: {e}")

    write_answers_index(args.output_directory)
    print(f"[OK] Index updated at {args.output_directory / 'index.html'}")

    top_index = update_top_level_index(args.output_directory)
    if top_index:
        print(f"[OK] Top-level index updated at {top_index}")


if __name__ == "__main__":
    main()
