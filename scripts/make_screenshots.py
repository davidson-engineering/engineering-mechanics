"""Render the screenshots in the README and docs/report.md (docs/images/*.png).

    uv run --with pillow --with pymupdf python scripts/make_screenshots.py

Each shot renders an example's HTML report, keeps only the sections it is
about, captures it with headless Chrome at 2x and trims the empty page
background; the print shot prints the report to PDF and lays out its first
pages side by side. Needs Google Chrome or Chromium (set CHROME to its path
if it is not found); pngquant, if installed, shrinks the files.

The README of every release on PyPI links report.png and report-3d.png on
main, so keep those two names.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from PIL import Image, ImageChops

from engmech.io.loader import load_model
from engmech.report.html import render_report

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
SCALE = 2
WIDTH = 1100  # CSS px: every shot is shown at the same scale
MARGIN = 24  # CSS px of page background kept around the content (the page's side margin)
PAGE = (245, 246, 248)  # the report's page background

# the parts of a load case's card
TABLES = "#case-1 > h3:not(:last-of-type), #case-1 > .table-wrap, #case-1 > .caption"
DIAGRAM = "#case-1 > h3:last-of-type, #case-1 > .figure"
DETAILS = "#case-1 > details"
OPEN_DETAILS = "document.querySelectorAll('details').forEach(function (d) { d.open = true; });"

# A model that equilibrium alone cannot solve, to show how a report says so
# (the bundled examples all solve cleanly).
GATE = """\
name: Gate on two hinges
description: |
  A 1.2 m wide, 60 kg gate hangs on two hinges 0.9 m apart. Equilibrium
  gives the horizontal hinge forces, but not how the weight is shared
  between the hinges: that depends on how stiff each hinge is.
analysis: planar
units: {length: m, force: N}
gravity: -y
points:
  A: [0, 0]
  B: [0, 0.9]
bodies:
  gate:
    shapes: [{type: point, mass: 60 kg, at: [0.6, 0.45]}]
    outline: [[0, -0.1], [1.2, -0.1], [1.2, 1.0], [0, 1.0], [0, -0.1]]
supports:
  A: {type: pin, at: A}
  B: {type: pin, at: B}
"""


def pick_view(label: str) -> str:
    """Script that switches the first diagram to the view with this button label."""
    return (
        "(function pick() { var b = Array.prototype.find.call("
        "document.querySelectorAll('#figure-1 .views button'), "
        f"function (b) {{ return b.textContent === {label!r}; }}); "
        "if (b) b.click(); else setTimeout(pick, 50); })();"
    )


@dataclass
class Shot:
    output: str
    example: str  # a bundled example, or a model file's text
    keep: str  # CSS selectors of the parts of the report to show
    hide: str = ""  # and of anything inside them to leave out
    css: str = ""
    script: str = ""
    pages: tuple[int, ...] = ()  # print these pages (1-based) instead of the screen


SHOTS = [
    # the README's hero: a 3D free-body diagram
    Shot(
        "report-3d.png",
        "excavator",
        keep="#case-1",
        hide=f"#case-1 > .section-head, {TABLES}, {DETAILS}",
        css="#case-1 > h3 { margin-top: 0; }",
    ),
    Shot("report.png", "load-combinations", keep="header.page, #summary"),
    Shot("report-warnings.png", GATE, keep="#summary, #case-1", hide=DETAILS),
    Shot("report-cases.png", "load-combinations", keep="#cases"),
    Shot("report-results.png", "excavator", keep="#case-1", hide=f"{DIAGRAM}, {DETAILS}"),
    Shot(
        "report-diagram.png",
        "truss",
        keep="#case-1",
        hide=f"#case-1 > .section-head, {TABLES}, {DETAILS}",
        css="#case-1 > h3 { margin-top: 0; }",
    ),
    Shot(
        "report-free-body.png",
        "frame",
        keep="#case-1",
        hide=f"#case-1 > .section-head, {TABLES}, {DETAILS}",
        css="#case-1 > h3 { margin-top: 0; }",
        script=pick_view("left"),
    ),
    Shot(
        "report-verification.png",
        "frame",
        keep="#case-1",
        hide=f"#case-1 > .section-head, {TABLES}, {DIAGRAM}",
        css="#case-1 > details:first-of-type { margin-top: 0; border-top: 0; }",
        script=OPEN_DETAILS,
    ),
    Shot("report-checks.png", "load-combinations", keep="#checks"),
    Shot("report-notes.png", "excavator", keep="#notes"),
    Shot("report-model.png", "motor-arm", keep="#model", script=OPEN_DETAILS),
    Shot("report-provenance.png", "frame", keep="#provenance"),
    Shot("report-print.png", "frame", keep="*", pages=(1, 2, 3)),
]


def chrome() -> str:
    candidates = [
        os.environ.get("CHROME"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        *(shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "chrome")),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    sys.exit("Chrome or Chromium not found; set CHROME to its path")


def render(shot: Shot, folder: Path) -> str:
    """The shot's report, loaded from a copy of its model file in ``folder`` so
    that the report names the file without the path it happens to be at."""
    if "\n" in shot.example:
        name, text = "gate", shot.example
    else:
        name = shot.example
        text = (resources.files("engmech") / "examples" / f"{name}.yaml").read_text("utf-8")
    (folder / f"{name}.yaml").write_text(text, encoding="utf-8")
    cwd = Path.cwd()
    os.chdir(folder)
    try:
        results = load_model(f"{name}.yaml").solve()
    finally:
        os.chdir(cwd)
    # logo=None: the maintainer's own logo settings must not end up in the docs
    html = render_report(results, source_path=f"{name}.yaml", logo=None)
    style = f"main > :not(:is({shot.keep})) {{ display: none !important; }}"
    if shot.hide:
        style += f" {shot.hide} {{ display: none !important; }}"
    html = html.replace("</head>", f"<style>{style} {shot.css}</style>\n</head>", 1)
    if shot.script:
        html = html.replace("</body>", f"<script>{shot.script}</script>\n</body>", 1)
    return html


def run_chrome(args: list[str], output: Path, profile: Path) -> None:
    output.unlink(missing_ok=True)
    command = [
        chrome(),
        "--headless",
        f"--user-data-dir={profile}",  # never touch the user's own Chrome profile
        "--no-first-run",
        "--enable-unsafe-swiftshader",  # WebGL for the 3D diagrams
        "--virtual-time-budget=20000",  # let plotly finish drawing
        *args,
    ]
    # Some Chrome builds (seen with 154 on macOS) write the file and then
    # never exit, so stop Chrome once it reports the file written.
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    ) as chrome_process:
        watchdog = threading.Timer(120, chrome_process.kill)
        watchdog.start()
        try:
            for line in chrome_process.stdout:
                if "bytes written to file" in line:
                    break
        finally:
            watchdog.cancel()
            chrome_process.terminate()
    if not output.is_file():
        sys.exit(f"Chrome did not write {output.name}")


def capture(html_path: Path, png_path: Path, profile: Path) -> None:
    args = [
        "--hide-scrollbars",
        f"--force-device-scale-factor={SCALE}",
        f"--window-size={WIDTH},6000",
        f"--screenshot={png_path}",
        html_path.as_uri(),
    ]
    run_chrome(args, png_path, profile)


def trim(png_path: Path) -> None:
    """Crop the empty background below (and above) the content, keeping a margin."""
    image = Image.open(png_path).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((0, 0)))
    box = ImageChops.difference(image, background).getbbox()
    if box is None:
        sys.exit(f"{png_path.name}: the screenshot is empty")
    pad = MARGIN * SCALE
    top, bottom = max(box[1] - pad, 0), min(box[3] + pad, image.height)
    image.crop((0, top, image.width, bottom)).save(png_path, optimize=True)


def print_pages(html_path: Path, png_path: Path, pages: tuple[int, ...], profile: Path) -> None:
    """Print to PDF and lay the pages out side by side, as sheets on the page
    background, as wide as the screenshots."""
    try:
        import pymupdf
    except ImportError:
        sys.exit("the print shot needs PyMuPDF: uv run --with pymupdf ...")
    pdf_path = png_path.with_suffix(".pdf")
    args = ["--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    run_chrome(args, pdf_path, profile)
    document = pymupdf.open(pdf_path)
    gap = MARGIN * SCALE
    sheet_width = (WIDTH * SCALE - gap * (len(pages) + 1)) // len(pages)
    sheets = []
    for number in pages:
        page = document[number - 1]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(*[sheet_width / page.rect.width] * 2))
        sheets.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
    height = max(s.height for s in sheets) + 2 * gap
    image = Image.new("RGB", (WIDTH * SCALE, height), PAGE)
    for k, sheet in enumerate(sheets):
        x = gap + k * (sheet_width + gap)
        image.paste((226, 232, 240), (x - 2, gap - 2, x + sheet.width + 2, gap + sheet.height + 2))
        image.paste(sheet, (x, gap))
    image.save(png_path, optimize=True)
    pdf_path.unlink()


def compress(png_path: Path) -> None:
    if shutil.which("pngquant"):
        subprocess.run(
            [
                "pngquant",
                "--quality=80-98",
                "--speed=1",
                "--strip",
                "--skip-if-larger",
                "--force",
                "--output",
                str(png_path),
                str(png_path),
            ],
            check=False,  # exits non-zero when it keeps the original, which is fine
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])  # optionally, just these outputs
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for shot in SHOTS:
            if only and shot.output not in only:
                continue
            html_path = tmp / f"{Path(shot.output).stem}.html"
            html_path.write_text(render(shot, tmp), encoding="utf-8")
            png_path = OUT / shot.output
            if shot.pages:
                print_pages(html_path, png_path, shot.pages, tmp / "chrome-profile")
            else:
                capture(html_path, png_path, tmp / "chrome-profile")
                trim(png_path)
            compress(png_path)
            with Image.open(png_path) as im:
                size = f"{im.width}x{im.height}"
            kb = png_path.stat().st_size / 1024
            print(f"wrote {png_path.relative_to(OUT.parent.parent)} ({size}, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
