---
name: eurorack-question
description: Ask a question about the user's Eurorack system via scripts/ask.py — it picks the relevant module manuals, answers using them, and documents the answer in answers/. Use when the user asks how to use/patch/configure modules in their rack.
argument-hint: "[QUESTION]"
---

Answer a question about the user's Eurorack system using the module manuals, and document the answer.

The question is: $ARGUMENTS

## Steps

1. **Check prerequisites.** `README.csv` must exist and have module rows, and the
   `manuals/` directory should contain the manual PDFs. If either is missing or empty,
   tell the user to run `/eurorack-import [MODULARGRID_URL]` first and stop.
   If no question was given, ask the user for one.

2. **Run ask.py** from the repo root, using the repo's virtualenv. Pass the question
   verbatim as a single quoted argument:

   ```bash
   .venv/bin/python3 scripts/ask.py \
     --prompt "<QUESTION>" \
     --input-csv README.csv \
     --manuals-dir manuals \
     --output-directory answers
   ```

   Notes:
   - The script makes two LLM calls (module scoping, then the answer) via the `claude`
     CLI by default, so allow a few minutes — use a generous Bash timeout or run in the
     background and monitor.
   - It also pulls in previous answers from `answers/` that involve the in-scope
     modules, so answers build on each other; nothing extra is needed for that.

3. **Relay and document.** The script writes the answer as markdown to `answers/`,
   named after the question plus a timestamp (e.g.
   `answers/how-do-i-use-the-clock-input-...-20260811-143221.md`). Find the file it
   just created (newest file in `answers/`, or take the path from the script's output):
   - Read it and give the user the answer content (not just the file path), plus the
     "Modules In Scope" list and where the file was saved.

## Troubleshooting

- `claude` CLI not logged in → the script exits with login instructions; relay them
  (`claude` then `/login`). `--llm-provider codex` or `--llm-provider openai`
  (needs `openai.key`) are alternative backends.
- If more than 10 manuals are in scope, only the first 10 are attached
  (`--max-manuals` raises the limit; the dropped ones are logged).
