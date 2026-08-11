#!/usr/bin/env python3
"""
Ask a question about your eurorack system.

Flow:
  1. Take a question ("prompt") as a CLI argument.
  2. Ask the LLM backend which modules from the CSV are in scope for the question.
  3. Find the corresponding manual PDFs via the CSV ("manual file name" column).
  4. Submit the question plus those manuals to the backend (openai or claude).
  5. Write the answer as a markdown file into the answers output directory.
"""

import argparse
import csv
import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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


# ---------- backends ----------

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

    def answer_with_pdfs(self, prompt: str, pdf_paths: list[Path]) -> str:
        content = []
        for pdf in pdf_paths:
            with pdf.open("rb") as f:
                uploaded = self.client.files.create(file=f, purpose="user_data")
            content.append({"type": "input_file", "file_id": uploaded.id})
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
        self.model = model

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

    def answer_with_pdfs(self, prompt: str, pdf_paths: list[Path]) -> str:
        file_list = "\n".join(f"- {p.resolve()}" for p in pdf_paths)
        full_prompt = (
            f"{prompt}\n\n"
            f"Read the following manual PDFs before answering:\n{file_list}\n"
        )
        add_dirs: list[str] = []
        for parent in {str(p.resolve().parent) for p in pdf_paths}:
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

    def answer_with_pdfs(self, prompt: str, pdf_paths: list[Path]) -> str:
        file_list = "\n".join(f"- {p.resolve()}" for p in pdf_paths)
        full_prompt = (
            f"{prompt}\n\n"
            f"Read the following manual PDFs before answering (extract their text with "
            f"a tool such as pdftotext if needed):\n{file_list}\n"
        )
        return self._run(full_prompt)


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

    parser.add_argument("--prompt", required=True,
                        help="The question to answer.")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Path to csv file containing modules and manual file paths (e.g. README.csv)")
    parser.add_argument("--manuals-dir", type=Path, default=None,
                        help="Directory where manual PDFs are stored [default: the CSV's directory]")
    parser.add_argument("--output-directory", type=Path, default=Path("answers"),
                        help="Directory to write the answer markdown to [default='answers']")
    parser.add_argument("--llm-provider", choices=["openai", "claude", "codex"], default="openai",
                        help="LLM provider: OpenAI API, Claude Code CLI, or Codex CLI [default='openai']")
    parser.add_argument("--model", default=None,
                        help="Model override (backend-specific; default gpt-4.1 for openai)")
    parser.add_argument("--key-file", type=Path, default=Path("openai.key"),
                        help="Path to a file containing an OpenAI API Key [default 'openai.key']")
    parser.add_argument("--max-manuals", type=int, default=10,
                        help="Maximum number of manual PDFs to attach [default=10]")

    args = parser.parse_args()

    manuals_dir = args.manuals_dir or args.csv.parent

    try:
        if args.llm_provider == "openai":
            backend = OpenAIBackend(args.key_file, args.model)
        elif args.llm_provider == "claude":
            backend = ClaudeBackend(args.model)
        else:
            backend = CodexBackend(args.model)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        sys.exit(f"[ERROR] {e}")

    rows = read_csv_rows(args.csv, ["manufacturer", "module", "quantity", "manual file name"])
    if not rows:
        sys.exit(f"No module rows found in {args.csv}")

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

    if not pdf_paths:
        sys.exit("[ERROR] No valid manual PDFs found for the in-scope modules.")

    if len(pdf_paths) > args.max_manuals:
        dropped = pdf_paths[args.max_manuals:]
        pdf_paths = pdf_paths[: args.max_manuals]
        print(f"[WARN] Too many manuals; using first {args.max_manuals}, dropping: "
              + ", ".join(p.name for p in dropped))

    module_names = ", ".join(f"{r['manufacturer'].strip()} {r['module'].strip()}" for r in scoped)
    answer_prompt = (
        "You are a eurorack modular synthesizer expert. Using the attached module manuals "
        f"(for: {module_names}), answer the following question. "
        "Format your answer as markdown.\n\n"
        f"Question: {args.prompt}"
    )

    print(f"[INFO] Asking {args.llm_provider} with {len(pdf_paths)} manual(s)...")
    answer = backend.answer_with_pdfs(answer_prompt, pdf_paths)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.output_directory / f"{slugify(args.prompt)}-{timestamp}.md"

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


if __name__ == "__main__":
    main()
