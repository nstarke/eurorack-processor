---
name: eurorack-import
description: Import a ModularGrid rack — build README.csv from the rack URL and find/download all module manuals via scripts/find_manuals.py. Use when the user provides a modulargrid.net rack or data_sheet URL, or asks to import/refresh their rack.
argument-hint: "[MODULARGRID_URL]"
---

Import a ModularGrid rack into this repo: build the module CSV and download every module's manual PDF.

The rack URL is: $ARGUMENTS

## Steps

1. **Validate the URL.** It must be a modulargrid.net URL containing a rack id — e.g.
   `https://modulargrid.net/e/modules_racks/data_sheet/2250471` or
   `https://modulargrid.net/e/racks/view/2250471`. If no URL was given, ask the user for one.
   The rack must be public (no ModularGrid login is used).

2. **Run the importer** from the repo root, using the repo's virtualenv:

   ```bash
   .venv/bin/python3 scripts/find_manuals.py \
     --rack-url "<URL>" \
     --output-csv README.csv \
     --manuals-dir manuals
   ```

   Notes:
   - This is slow: it makes an LLM web-search call per module and downloads PDFs.
     Run it in the background (Bash `run_in_background`) and monitor its output rather
     than blocking with a short timeout.
   - It is safe to re-run: modules that already have a valid PDF in `manuals/` are
     skipped, and `README.csv` is rewritten after every module, so an interrupted
     run just resumes where it left off.
   - Do not edit `README.csv` while the script is running.

3. **Verify and report.** When the run finishes:
   - Read `README.csv` and count the rows.
   - List any rows whose `manual file name` column is empty — these are modules where
     no manual, product page, or archive.org fallback was found.
   - Note which downloaded files end in `_Product_Page.pdf` (product page saved as PDF
     because no real manual exists).
   - Summarize for the user: total modules, real manuals downloaded, product-page
     fallbacks, and missing modules. For missing modules, mention that re-running the
     skill retries only those rows, and that a manual can be dropped into `manuals/`
     by hand and its filename added to the CSV.

## Troubleshooting

- `claude` CLI not logged in → the script exits with login instructions; relay them
  (`claude` then `/login`). `--llm-provider codex` is an alternative backend.
- Rack page fetch fails → the rack is probably private on ModularGrid; ask the user to
  make it public.
- To force a re-fetch of one module, delete its PDF from `manuals/` (or blank its CSV
  entry) and re-run.
