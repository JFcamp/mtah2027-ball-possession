"""docs/branca analytics.md -> docs/branca analytics.pdf (needs the `markdown` package and Microsoft Edge).

    python scripts/make_pdf.py

Not part of the pipeline. Edge headless prints the HTML to PDF.
"""

import io
import os
import subprocess
import sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CSS = """body{font-family:Segoe UI,Arial,sans-serif;font-size:9.5pt;line-height:1.35;margin:0}
h1{font-size:15pt;margin:0 0 6pt}h2{font-size:11.5pt;margin:10pt 0 3pt;border-bottom:1px solid #999}
table{border-collapse:collapse;width:100%;margin:4pt 0;font-size:8.3pt}
th,td{border:1px solid #aaa;padding:1.5pt 4pt;text-align:left}th{background:#eee}
code{font-family:Consolas,monospace;font-size:8.3pt;word-break:break-all}
pre{background:#f4f4f4;padding:4pt;font-size:8.3pt;white-space:pre-wrap}
p,li{margin:3pt 0}@page{size:A4;margin:14mm 14mm}"""


def main():
    src = io.open(os.path.join(ROOT, "docs", "branca analytics.md"), encoding="utf-8").read()
    body = markdown.markdown(src, extensions=["tables", "fenced_code"])
    html = os.path.join(ROOT, "docs", "_tmp.html")
    io.open(html, "w", encoding="utf-8").write(
        "<html><head><meta charset=utf-8><style>" + CSS + "</style></head><body>" + body + "</body></html>")
    pdf = os.path.join(ROOT, "docs", "branca analytics.pdf")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf=" + pdf, "file:///" + html.replace("\\", "/")],
                   check=False, capture_output=True, timeout=120)
    os.remove(html)
    print("wrote", pdf, os.path.getsize(pdf))


if __name__ == "__main__":
    sys.exit(main())
