"""Render report/study_summary.md to an A4 PDF (Markdown -> HTML -> headless Chrome/Edge).

Run:  python scripts/build_summary_pdf.py [--md report/study_summary.md]
Writes the .html and .pdf next to the Markdown file and prints the page count.
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser",
]
CSS = """
@page { size: A4; margin: 14mm 16mm 13mm 16mm; }
body { font-family: "Georgia", "Times New Roman", serif; font-size: 9.6pt; line-height: 1.32; color: #111; }
h1 { font-size: 13.5pt; margin: 0 0 4pt 0; line-height: 1.2; }
h2 { font-size: 11pt; margin: 9pt 0 3pt 0; border-bottom: 0.6pt solid #bbb; padding-bottom: 1pt; }
p { margin: 3pt 0; }
ul, ol { margin: 2pt 0 3pt 0; padding-left: 16pt; }
li { margin: 1pt 0; }
table { border-collapse: collapse; margin: 4pt 0 5pt 0; font-size: 8.8pt; }
th, td { border: 0.5pt solid #999; padding: 1.5pt 5pt; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f0efea; }
img { display: block; width: 56%; margin: 2pt auto; }
code { font-family: Consolas, monospace; font-size: 8.6pt; }
em { color: #333; }
"""


def find_browser() -> str:
    """First Chrome/Edge executable found on this machine."""
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("no Chrome or Edge found for printing to PDF")


def main() -> int:
    """Convert, print, and report the page count."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path, default=ROOT / "report" / "study_summary.md")
    args = ap.parse_args()
    md_path = args.md.resolve()
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables"])
    # PDF title metadata comes from <title>; use the document's first heading, not the file name
    title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), md_path.stem)
    html_path = md_path.with_suffix(".html")
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style>"
                         f"</head><body>{body}</body></html>", encoding="utf-8")
    pdf_path = md_path.with_suffix(".pdf")
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=120)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    print(f"wrote {pdf_path} ({pages} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
