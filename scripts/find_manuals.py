#!/usr/bin/env python3
"""
Find and download eurorack module manuals.

Input: either a newline-delimited list of modules (free text like
"Make Noise Maths", or pre-parsed "manufacturer,module" lines), or an existing
README.csv-style CSV (--input-csv) whose "manual file name" column may be
missing, empty, or point at files that are not valid PDFs — only those rows
are (re)processed. For each module the script:

  1. Uses an LLM CLI (`claude -p` or `codex exec`) with web search to locate
     the official manual PDF URL and the product web page.
  2. Downloads and validates the manual PDF.
  3. If no manual PDF can be found, saves the product web page as a PDF
     instead (headless Chrome, falling back to weasyprint).
  4. As a last resort, searches archive.org — both the item library and the
     Wayback Machine — for the manual or a snapshot of the product page.

Output: PDFs in --manuals-dir plus a CSV matching the format of
eurorack-manuals-repo/README.csv:

    "manufacturer","module",quantity,"manual file name"

Duplicate input lines increment the module's quantity. Modules already present
in the output CSV (with a valid PDF on disk) are skipped, so re-running the
script only processes what's missing.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import requests

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

CSV_FIELDS = ["manufacturer", "module", "quantity", "manual file name"]


# ---------- helpers ----------

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


def safe_manual_name(manufacturer: str, module: str, suffix: str = "Manual") -> str:
    base = f"{manufacturer}_{module}_{suffix}"
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("_")
    return f"{base}.pdf"


def extract_json_object(text: str) -> dict:
    """Pull the first JSON object out of an LLM response (tolerates code fences/prose)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"No JSON object found in response:\n{text}")
        text = text[start : end + 1]
    return json.loads(text)


# ---------- LLM backends ----------

class ClaudeBackend:
    """Uses the Claude Code CLI (`claude -p`) with WebSearch/WebFetch enabled."""

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

    def complete_text(self, prompt: str) -> str:
        cmd = ["claude", "-p", "--allowedTools", "WebSearch,WebFetch"]
        if self.model:
            cmd.extend(["--model", self.model])
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI failed:\n{result.stderr}")
        return result.stdout.strip()


class CodexBackend:
    """Uses the OpenAI Codex CLI (`codex exec --search`) with a ChatGPT login."""

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

    def complete_text(self, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(mode="r", suffix=".md", delete=False) as out:
            out_path = Path(out.name)
        try:
            cmd = [
                "codex", "exec",
                "--sandbox", "read-only",
                "--skip-git-repo-check",
                "--search",
                "--output-last-message", str(out_path),
            ]
            if self.model:
                cmd.extend(["-m", self.model])
            result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"codex CLI failed:\n{result.stderr}")
            answer = out_path.read_text(encoding="utf-8").strip()
            if not answer:
                raise RuntimeError(f"codex CLI returned an empty answer:\n{result.stderr}")
            return answer
        finally:
            out_path.unlink(missing_ok=True)


# ---------- research ----------

RESEARCH_TEMPLATE = """You are researching the eurorack modular synthesizer module: "{line}"

Task: find the OFFICIAL user manual PDF for this module on the internet.

1. Determine the manufacturer name and the module name.
2. Search the web for the manufacturer's official manual PDF for this module.
   Direct links to .pdf files on the manufacturer's own site are strongly
   preferred. PDFs hosted by reputable sources (ModularGrid, retailers,
   ModWiggler attachments) are acceptable if no official link exists.
3. Also find the module's product web page (manufacturer page preferred,
   otherwise its ModularGrid page).

Respond with ONLY a JSON object, no prose and no code fences, shaped exactly like:

{{"manufacturer": "...", "module": "...", "pdf_urls": ["https://..."], "product_page_url": "https://..."}}

Rules:
- "pdf_urls": up to 3 candidate direct-download PDF URLs, best first.
  Use [] if you cannot find any PDF manual.
- "product_page_url": use null if you cannot find a product page.
- Use the manufacturer's and module's official spelling/capitalization.
"""


def research_module(backend, line: str) -> dict:
    response = backend.complete_text(RESEARCH_TEMPLATE.format(line=line))
    info = extract_json_object(response)
    info.setdefault("pdf_urls", [])
    if not isinstance(info["pdf_urls"], list):
        info["pdf_urls"] = [info["pdf_urls"]]
    info["manufacturer"] = str(info.get("manufacturer") or "").strip()
    info["module"] = str(info.get("module") or "").strip()
    url = info.get("product_page_url")
    info["product_page_url"] = str(url).strip() if url else None
    if not info["manufacturer"] or not info["module"]:
        raise ValueError(f"LLM response missing manufacturer/module: {info}")
    return info


# ---------- download / render ----------

def download_pdf(url: str, dest: Path) -> bool:
    """Download url to dest and verify it is a real PDF. Returns True on success."""
    try:
        with requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=90, stream=True
        ) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    except Exception as e:
        print(f"    [WARN] download failed: {url}: {e}")
        dest.unlink(missing_ok=True)
        return False

    ok, reason = is_probably_pdf(dest)
    if not ok:
        print(f"    [WARN] {url} is not a valid PDF ({reason})")
        dest.unlink(missing_ok=True)
        return False
    return True


def find_chrome() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        path = shutil.which(name)
        if path:
            return path
    return None


def render_page_to_pdf(url: str, dest: Path) -> bool:
    """Save a web page as a PDF using headless Chrome, falling back to weasyprint."""
    chrome = find_chrome()
    if chrome:
        cmd = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={dest}",
            url,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            print(f"    [WARN] headless chrome timed out rendering {url}")
        ok, _ = is_probably_pdf(dest)
        if ok:
            return True
        dest.unlink(missing_ok=True)

    try:
        from weasyprint import HTML

        HTML(url=url).write_pdf(str(dest))
        ok, _ = is_probably_pdf(dest)
        if ok:
            return True
        dest.unlink(missing_ok=True)
    except Exception as e:
        print(f"    [WARN] weasyprint failed rendering {url}: {e}")
        dest.unlink(missing_ok=True)
    return False


# ---------- archive.org fallbacks ----------

def archive_org_item_pdf(query: str, dest: Path) -> bool:
    """Search the archive.org item library for a PDF matching the query."""
    try:
        r = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f'({query}) AND mediatype:(texts)',
                "fl[]": "identifier",
                "rows": "5",
                "page": "1",
                "output": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
    except Exception as e:
        print(f"    [WARN] archive.org search failed: {e}")
        return False

    for doc in docs:
        identifier = doc.get("identifier")
        if not identifier:
            continue
        try:
            meta = requests.get(
                f"https://archive.org/metadata/{identifier}",
                headers={"User-Agent": USER_AGENT},
                timeout=60,
            ).json()
        except Exception:
            continue
        for f in meta.get("files", []):
            name = f.get("name", "")
            if name.lower().endswith(".pdf"):
                url = f"https://archive.org/download/{identifier}/{quote(name)}"
                print(f"    [INFO] trying archive.org item: {url}")
                if download_pdf(url, dest):
                    return True
    return False


def wayback_snapshot_url(url: str) -> str | None:
    """Return the closest Wayback Machine snapshot URL for a page, if any."""
    try:
        r = requests.get(
            "https://archive.org/wayback/available",
            params={"url": url},
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        r.raise_for_status()
        closest = r.json().get("archived_snapshots", {}).get("closest", {})
        if closest.get("available") and closest.get("url"):
            return closest["url"]
    except Exception as e:
        print(f"    [WARN] wayback lookup failed for {url}: {e}")
    return None


# ---------- per-module pipeline ----------

def acquire_manual(info: dict, manuals_dir: Path) -> str:
    """
    Try, in order: direct PDF download, product page render, archive.org.
    Returns the manual file name (in manuals_dir), or "" if everything failed.
    """
    manufacturer, module = info["manufacturer"], info["module"]

    # 1. Direct PDF manual
    manual_name = safe_manual_name(manufacturer, module)
    dest = manuals_dir / manual_name
    if is_probably_pdf(dest)[0]:
        print(f"    [INFO] already downloaded: {manual_name}")
        return manual_name
    for url in info["pdf_urls"]:
        print(f"    [INFO] trying manual PDF: {url}")
        if download_pdf(url, dest):
            return manual_name

    # 2. Product page saved as PDF
    page_name = safe_manual_name(manufacturer, module, suffix="Product_Page")
    page_dest = manuals_dir / page_name
    if is_probably_pdf(page_dest)[0]:
        print(f"    [INFO] already rendered: {page_name}")
        return page_name
    if info["product_page_url"]:
        print(f"    [INFO] no PDF manual; rendering product page: {info['product_page_url']}")
        if render_page_to_pdf(info["product_page_url"], page_dest):
            return page_name

    # 3a. archive.org item library
    print(f"    [INFO] searching archive.org for: {manufacturer} {module}")
    if archive_org_item_pdf(f'"{manufacturer}" "{module}" manual', dest):
        return manual_name

    # 3b. Wayback Machine snapshot of the product page
    if info["product_page_url"]:
        snapshot = wayback_snapshot_url(info["product_page_url"])
        if snapshot:
            print(f"    [INFO] rendering wayback snapshot: {snapshot}")
            if render_page_to_pdf(snapshot, page_dest):
                return page_name

    return ""


# ---------- CSV handling ----------

def read_csv_rows(csv_path: Path, required_fields: list[str]) -> list[dict]:
    """Read a CSV that may or may not have a header row."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)

        # csv.Sniffer misdetects headerless single-row files; the literal
        # column name in line 1 is a decisive signal for this format.
        first_line = sample.splitlines()[0] if sample else ""
        has_header = "manufacturer" in first_line.lower()

        reader = csv.DictReader(f) if has_header else csv.DictReader(f, fieldnames=required_fields)
        return list(reader)


def load_existing_rows(csv_path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not csv_path.exists():
        return rows
    for row in read_csv_rows(csv_path, CSV_FIELDS):
        if not row.get("manufacturer") or not row.get("module"):
            continue
        key = (row["manufacturer"].strip().lower(), row["module"].strip().lower())
        rows[key] = row
    return rows


def write_rows(csv_path: Path, rows: dict[tuple[str, str], dict]):
    ordered = sorted(rows.values(), key=lambda r: (r["manufacturer"].lower(), r["module"].lower()))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(CSV_FIELDS)
        for row in ordered:
            try:
                quantity = int(str(row.get("quantity", 1)).strip() or 1)
            except ValueError:
                quantity = 1
            writer.writerow([
                row["manufacturer"].strip(),
                row["module"].strip(),
                quantity,
                (row.get("manual file name") or "").strip(),
            ])


# ---------- main ----------

def read_module_lines(path: Path) -> list[tuple[str, int]]:
    """Read the input list; duplicate lines collapse into a quantity count."""
    text = sys.stdin.read() if str(path) == "-" else path.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    order: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.lower()
        if key not in counts:
            counts[key] = 0
            order.append(line)
        counts[key] += 1
    return [(line, counts[line.lower()]) for line in order]


def build_work_from_modules(path: Path) -> list[dict]:
    work = []
    for line, quantity in read_module_lines(path):
        item = {"line": line, "manufacturer": None, "module": None, "quantity": quantity}
        if "," in line:
            manufacturer, module = (part.strip() for part in line.split(",", 1))
            if manufacturer and module:
                item["manufacturer"], item["module"] = manufacturer, module
        work.append(item)
    return work


def build_work_from_csv(path: Path, rows: dict[tuple[str, str], dict]) -> list[dict]:
    """
    Seed `rows` with every CSV row (so valid rows carry through to the output)
    and return work items for all of them; rows that already have a valid
    manual are skipped later by the main loop's on-disk check.
    """
    work = []
    for row in read_csv_rows(path, CSV_FIELDS):
        manufacturer = (row.get("manufacturer") or "").strip()
        module = (row.get("module") or "").strip()
        if not manufacturer or not module:
            continue
        try:
            quantity = int(str(row.get("quantity") or "1").strip() or 1)
        except ValueError:
            quantity = 1
        manual = (row.get("manual file name") or "").strip()
        key = (manufacturer.lower(), module.lower())
        rows.setdefault(key, {
            "manufacturer": manufacturer,
            "module": module,
            "quantity": quantity,
            "manual file name": manual,
        })
        work.append({
            "line": f"{manufacturer},{module}",
            "manufacturer": manufacturer,
            "module": module,
            "quantity": quantity,
        })
    return work


def main():
    parser = argparse.ArgumentParser(
        description="Find and download manuals for a list of eurorack modules."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--modules", type=Path,
                        help="Newline-delimited list of modules ('-' reads stdin). "
                             "Lines may be free text ('Make Noise Maths') or 'manufacturer,module'.")
    source.add_argument("--input-csv", type=Path,
                        help="Existing README.csv-style CSV; rows whose 'manual file name' "
                             "column is missing, empty, or not a valid PDF on disk are "
                             "(re)processed. Header row and the fourth column are optional.")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="CSV to create/update, in eurorack-manuals-repo README.csv format "
                             "[default: --input-csv if given, else 'README.csv']")
    parser.add_argument("--manuals-dir", type=Path, default=None,
                        help="Directory to download manual PDFs into "
                             "[default: the --input-csv's directory if given, else 'manuals']")
    parser.add_argument("--llm-provider", choices=["claude", "codex"], default="claude",
                        help="LLM CLI used to research manual URLs [default='claude']")
    parser.add_argument("--model", default=None,
                        help="Model override (backend-specific)")
    args = parser.parse_args()

    output_csv = args.output_csv or args.input_csv or Path("README.csv")
    manuals_dir = args.manuals_dir or (args.input_csv.parent if args.input_csv else Path("manuals"))

    try:
        if args.llm_provider == "claude":
            backend = ClaudeBackend(args.model)
        else:
            backend = CodexBackend(args.model)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"[ERROR] {e}")

    rows = load_existing_rows(output_csv)

    if args.input_csv:
        if not args.input_csv.exists():
            sys.exit(f"[ERROR] Input CSV not found: {args.input_csv}")
        work = build_work_from_csv(args.input_csv, rows)
        if not work:
            sys.exit(f"[ERROR] No module rows found in {args.input_csv}")
    else:
        work = build_work_from_modules(args.modules)
        if not work:
            sys.exit(f"[ERROR] No module lines found in {args.modules}")

    manuals_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    def has_valid_manual(key: tuple[str, str]) -> bool:
        existing = rows.get(key)
        return bool(existing) and is_probably_pdf(
            manuals_dir / (existing.get("manual file name") or "")
        )[0]

    for i, item in enumerate(work, 1):
        line = item["line"]
        print(f"[{i}/{len(work)}] {line}")

        # Items with a known manufacturer/module (CSV rows or pre-parsed lines)
        # can skip the LLM entirely when a valid manual already exists on disk.
        if item["manufacturer"] and has_valid_manual((item["manufacturer"].lower(), item["module"].lower())):
            print(f"    [SKIP] already has a valid manual")
            continue

        try:
            info = research_module(backend, line)
        except Exception as e:
            print(f"    [FAIL] LLM research failed: {e}")
            failures.append(line)
            continue

        # Keep the caller's naming when it was given explicitly, so CSV rows
        # are updated in place rather than duplicated under new spelling.
        if item["manufacturer"]:
            info["manufacturer"], info["module"] = item["manufacturer"], item["module"]

        key = (info["manufacturer"].lower(), info["module"].lower())
        if has_valid_manual(key):
            print(f"    [SKIP] already has a valid manual")
            continue

        manual_name = acquire_manual(info, manuals_dir)
        if manual_name:
            print(f"    [OK] {info['manufacturer']} {info['module']} -> {manual_name}")
        else:
            print(f"    [FAIL] no manual, product page, or archive.org result for {line}")
            failures.append(line)

        rows[key] = {
            "manufacturer": info["manufacturer"],
            "module": info["module"],
            "quantity": item["quantity"],
            "manual file name": manual_name,
        }
        # Write incrementally so an interrupted run keeps its progress.
        write_rows(output_csv, rows)

    write_rows(output_csv, rows)
    print(f"\n[DONE] {len(rows)} modules in {output_csv}, manuals in {manuals_dir}/")
    if failures:
        print(f"[WARN] {len(failures)} module(s) with no manual found:")
        for line in failures:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
