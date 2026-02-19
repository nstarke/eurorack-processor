#!/usr/bin/env python3

import csv, os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import shutil
import subprocess
import html as htmllib

from openai import OpenAI


# ---------- helpers ----------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def safe_basename(manufacturer: str, module: str) -> str:
    return f"{manufacturer}_{module}".replace(" ", "_").replace("/", "_")


def md_exists_for_module(name: str, md_dir: Path) -> tuple[bool, Path]:
    md_path = md_dir / f"{name}.md"
    return md_path.exists(), md_path


def ensure_relative_path(from_dir: Path, target: Path) -> str:
    """Return a posix relative path from from_dir to target (works for siblings)."""
    rel = os.path.relpath(str(target), start=str(from_dir))
    return Path(rel).as_posix()


# ---------- conversion backends ----------

def pandoc_available() -> bool:
    return shutil.which("pandoc") is not None


def pandoc_convert(md_path: Path, out_path: Path, to_format: str, css: Path | None, pdf_engine: str | None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["pandoc", str(md_path), "-o", str(out_path)]

    if to_format == "html":
        cmd.extend(["-t", "html"])
        if css:
            cmd.extend(["--css", str(css)])

    if to_format == "pdf":
        if css:
            cmd.extend(["--css", str(css)])
        if pdf_engine:
            cmd.extend(["--pdf-engine", pdf_engine])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"pandoc failed converting {md_path.name} -> {out_path.name}\n{result.stderr}"
        )


def convert_md_to_html(md_path: Path, html_path: Path, css: Path | None):
    html_path.parent.mkdir(parents=True, exist_ok=True)

    if pandoc_available():
        pandoc_convert(md_path, html_path, "html", css, None)
        return

    import markdown as mdlib

    md_text = md_path.read_text(encoding="utf-8")
    html_body = mdlib.markdown(md_text, extensions=["extra", "tables", "fenced_code"])

    css_link = f'<link rel="stylesheet" href="{css.name}">' if css else ""

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{htmllib.escape(md_path.stem)}</title>
  {css_link}
</head>
<body>
{html_body}
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")

    if css:
        shutil.copy(css, html_path.parent / css.name)


def convert_md_to_pdf(md_path: Path, pdf_path: Path, css: Path | None, pdf_engine: str | None):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if pandoc_available():
        pandoc_convert(md_path, pdf_path, "pdf", css, pdf_engine)
        return

    import markdown as mdlib
    from weasyprint import HTML, CSS

    md_text = md_path.read_text(encoding="utf-8")
    html_body = mdlib.markdown(md_text, extensions=["extra", "tables", "fenced_code"])

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{htmllib.escape(md_path.stem)}</title>
</head>
<body>
{html_body}
</body>
</html>
"""

    stylesheets = [CSS(filename=str(css))] if css else None
    HTML(string=html_doc, base_url=str(md_path.parent)).write_pdf(
        str(pdf_path),
        stylesheets=stylesheets,
    )


# ---------- index generation ----------

def write_prompt_index(prompt_dir: Path):
    md_files = sorted((prompt_dir / "md").glob("*.md"))
    html_files = sorted((prompt_dir / "html").glob("*.html"))
    pdf_files = sorted((prompt_dir / "pdf").glob("*.pdf"))
    manual_files = sorted((prompt_dir / "manuals").glob("*.pdf"))

    def section(title, files, subdir):
        if not files:
            return f"<h2>{title}</h2><p><em>None</em></p>"
        items = "\n".join(
            f'<li><a href="{subdir}/{f.name}">{f.name}</a></li>' for f in files
        )
        return f"<h2>{title}</h2><ul>{items}</ul>"

    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{prompt_dir.name}</title></head>
<body>
<h1>{prompt_dir.name}</h1>
<p><a href="../index.html">← All prompts</a></p>
{section("Manual PDFs", manual_files, "manuals")}
{section("Markdown", md_files, "md")}
{section("HTML", html_files, "html")}
{section("PDF", pdf_files, "pdf")}
</body>
</html>
"""
    (prompt_dir / "index.html").write_text(html, encoding="utf-8")


def write_top_level_index(base_output_dir: Path):
    items = []
    for d in sorted(p for p in base_output_dir.iterdir() if p.is_dir()):
        items.append(f'<li><a href="{d.name}/index.html">{d.name}</a></li>')

    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Outputs</title></head>
<body>
<h1>Prompt Outputs</h1>
<ul>{"".join(items)}</ul>
</body>
</html>
"""
    (base_output_dir / "index.html").write_text(html, encoding="utf-8")


# ---------- worker ----------

def process_row(
    row,
    base_prompt,
    output_dir: Path,
    md_dir,
    html_dir,
    pdf_dir,
    manuals_dir,
    client,
    upload_cache,
    cache_lock,
    model,
    generate_pdf,
    generate_html,
    css,
    pdf_engine,
):
    manufacturer = row["manufacturer"].strip()
    module = row["module"].strip()
    manual_name = row["manual file name"].strip()

    name = safe_basename(manufacturer, module)

    # ----- skip EARLY if md already exists -----
    already_done, md_existing = md_exists_for_module(name, md_dir)
    if already_done:
        return f"[SKIP] {manufacturer} – {module}: markdown exists ({md_existing})"

    manual_pdf_src = (manuals_dir / manual_name)
    if not manual_pdf_src.exists():
        return f"[WARN] Missing manual {manual_pdf_src}"

    # Copy manual into output_dir/<prompt>/manuals/
    manuals_out_dir = output_dir / "manuals"
    manuals_out_dir.mkdir(parents=True, exist_ok=True)

    # Keep original manual filename as-is in manuals/ to make it easy to map back to CSV
    manual_pdf_dst = manuals_out_dir / manual_pdf_src.name
    if not manual_pdf_dst.exists():
        shutil.copy2(manual_pdf_src, manual_pdf_dst)

    # Build a relative link to the manual from md/html/pdf directories
    manual_rel_from_md = Path(ensure_relative_path(md_dir, manual_pdf_dst))
    manual_rel_from_html = Path(ensure_relative_path(html_dir, manual_pdf_dst)) if (generate_html and html_dir) else None
    manual_rel_from_pdf = Path(ensure_relative_path(pdf_dir, manual_pdf_dst)) if (generate_pdf and pdf_dir) else None

    # Upload cache uses the source manual path (stable) to avoid reuploading within a run
    with cache_lock:
        file_id = upload_cache.get(manual_pdf_src)

    if not file_id:
        with manual_pdf_src.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="user_data")
        file_id = uploaded.id
        with cache_lock:
            upload_cache[manual_pdf_src] = file_id

    # Add a manual link header to the prompt so the generated markdown includes it
    # Also prepend it to output explicitly so it always exists even if the model ignores instructions.
    manual_link_md = f"[Manual PDF]({manual_rel_from_md.as_posix()})"
    preamble_md = (
        f"# {manufacturer} — {module}\n\n"
        f"- {manual_link_md}\n\n"
        f"---\n\n"
    )

    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id},
                {"type": "input_text", "text": base_prompt + "\n\nInclude a link to the manual PDF at the top of the markdown output."},
            ],
        }],
    )

    md_path = md_dir / f"{name}.md"
    md_path.write_text(preamble_md + response.output_text, encoding="utf-8")

    outputs = [md_path]

    if generate_html:
        if html_dir is None:
            raise RuntimeError("Internal error: html_dir is None while generate_html=True")
        html_path = html_dir / f"{name}.html"
        convert_md_to_html(md_path, html_path, css)
        outputs.append(html_path)

        # Ensure the HTML has a visible manual link even if markdown->html didn't render it as desired
        # (Our preamble does, but this is a safe post-pass.)
        if manual_rel_from_html is not None:
            html_text = html_path.read_text(encoding="utf-8")
            if "Manual PDF" not in html_text:
                inject = f'<p><a href="{htmllib.escape(manual_rel_from_html.as_posix())}">Manual PDF</a></p>\n'
                html_text = html_text.replace("<body>", "<body>\n" + inject, 1)
                html_path.write_text(html_text, encoding="utf-8")

    if generate_pdf:
        if pdf_dir is None:
            raise RuntimeError("Internal error: pdf_dir is None while generate_pdf=True")
        pdf_path = pdf_dir / f"{name}.pdf"
        convert_md_to_pdf(md_path, pdf_path, css, pdf_engine)
        outputs.append(pdf_path)

        # You can't "link-inject" into a finished PDF reliably without re-rendering,
        # but since the manual link is in the markdown header, it will appear in the rendered PDF.

    return f"[OK] {manufacturer} {module} → " + ", ".join(p.name for p in outputs)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--manuals-dir", type=Path, default=Path("manuals"))
    parser.add_argument("--output-directory", type=Path, default=Path("output"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--key-file", type=Path, default=Path("openai.key"))
    parser.add_argument("--css", type=Path, help="Optional CSS file for HTML/PDF styling", default=Path("css/basic.css"))

    parser.add_argument("--generate-pdf", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--generate-html", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pdf-engine")

    args = parser.parse_args()

    api_key = load_openai_key(args.key_file)
    client = OpenAI(api_key=api_key)

    base_output = args.output_directory
    base_output.mkdir(exist_ok=True)

    output_dir = base_output / args.prompt.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    md_dir = output_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    html_dir = None
    pdf_dir = None

    if args.generate_html:
        html_dir = output_dir / "html"
        html_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_pdf:
        pdf_dir = output_dir / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)

    # New: manuals output directory
    (output_dir / "manuals").mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(args.csv, ["manufacturer", "module", "quantity", "manual file name"])
    base_prompt = read_text(args.prompt)

    upload_cache = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(
            lambda row: process_row(
                row,
                base_prompt,
                output_dir,
                md_dir,
                html_dir,
                pdf_dir,
                args.manuals_dir,
                client,
                upload_cache,
                lock,
                args.model,
                args.generate_pdf,
                args.generate_html,
                args.css,
                args.pdf_engine,
            ),
            rows,
        ):
            print(r)

    write_prompt_index(output_dir)
    write_top_level_index(base_output)

    print("Done")


if __name__ == "__main__":
    main()
