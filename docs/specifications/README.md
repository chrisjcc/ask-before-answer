# AskBeforeAnswer TRD/TDD — Editable Source Templates

This is the raw Python source that generated:
- `TRD_AskBeforeAnswer_ClarifyOrAnswer.pdf`
- `TDD_AskBeforeAnswer_SystemArchitecture.pdf`

Built with [ReportLab](https://www.reportlab.com/) (`pip install reportlab`).

## Files

| File | Purpose |
|---|---|
| `doc_style.py` | Shared style/template module — colors, fonts, the title/metadata table, section headers, data tables, NOTE/IMPORTANT callout boxes, ASCII diagram blocks, code blocks, and the page header/footer. Both documents import this. Edit here to change the shared look (e.g. palette, fonts, table banding) for **both** documents at once. |
| `gen_trd.py` | Content + section flow for the **TRD**. Edit here to change TRD wording, tables, requirements, or add/remove sections. |
| `gen_tdd.py` | Content + section flow for the **TDD**. Edit here to change TDD wording, diagrams, or add/remove sections. |

## How it works

Each `gen_*.py` script builds a list of ReportLab "flowables" (`story = [...]`) using helper
functions from `doc_style.py`:

- `h1(number, "Title")` — numbered navy section heading
- `h2("Title")` — blue underlined subsection heading
- `P("text", "Body")` — a paragraph (accepts basic HTML-like tags: `<b>`, `<i>`, `&mdash;`, etc.)
- `bullets([...])` / `numbered([...])` — bullet or numbered lists
- `data_table(header_row, rows, col_widths=[...])` — navy-header, gray-banded table
- `note_box("NOTE", "text")` — tan callout box (pass `bg=...`, `border=...` for the red/orange "IMPORTANT" variant)
- `code_block([...lines...])` — monospace command/snippet box
- `diagram_block([...lines...], align=TA_LEFT)` — monospace ASCII flow-diagram box
- `meta_table([...])` — the navy Document ID/Version/Status/... table on page 1

At the bottom of each script, `build_doc(output_path, running_title, footer_left, story)` renders
the PDF.

## Editing content

Open `gen_trd.py` or `gen_tdd.py` and edit the `story.append(...)` calls directly — they read
top-to-bottom in document order. To add a new section, copy an existing `h1(...)` block (heading +
paragraphs/tables/etc.) and adjust the section number and text. Renumber subsequent `h1()` calls
if you insert a section in the middle.

Table rows are plain Python lists of strings, so most edits are just changing the text inside
`data_table([...], [[...], [...]])` calls. Basic HTML entities/tags work inside any text string
(`&mdash;`, `&amp;`, `<b>bold</b>`, `<i>italic</i>`).

## Regenerating the PDFs

```bash
pip install reportlab --break-system-packages   # if not already installed

python3 gen_trd.py   # writes TRD_AskBeforeAnswer_ClarifyOrAnswer.pdf
python3 gen_tdd.py   # writes TDD_AskBeforeAnswer_SystemArchitecture.pdf
```

By default both scripts write to `/mnt/user-data/outputs/`. Change the path in the
`build_doc(...)` call at the bottom of each script to write elsewhere.

## Changing the shared visual style

Everything visual (palette, fonts, spacing, table/callout styling, header/footer) lives in
`doc_style.py` under the `# ---- palette --` and `# ---- styles --` sections near the top:

- `NAVY`, `BLUE`, `TAN_BG`, etc. — hex colors used throughout
- `styles["H1"]`, `styles["H2"]`, `styles["Body"]`, etc. — `ParagraphStyle` definitions (font, size, leading, color)
- `_header_footer()` — the running header/footer drawn on every page

Because both `gen_trd.py` and `gen_tdd.py` import from `doc_style.py`, a single edit there
(e.g. changing `NAVY` to a different hex color) updates both documents consistently the next
time you regenerate them.
