"""Render the report screenshots used in the README (docs/images/*.png).

    uv run --with pillow python scripts/make_screenshots.py

Each shot renders a bundled example's HTML report, hides the sections the
shot leaves out, captures it with headless Chrome at 2x and trims the empty
page background. Needs Google Chrome or Chromium (set CHROME to its path if
it is not found); pngquant, if installed, shrinks the files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from importlib import resources
from pathlib import Path

from PIL import Image, ImageChops

from engmech.io.loader import load_model
from engmech.report.html import render_report

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
SCALE = 2
MARGIN = 24  # CSS px of page background kept around the content (the page's side margin)
BELOW_RESULTS = "details, #checks, #notes, #model, #provenance, #method, footer"

SHOTS = [
    {
        # the top of a report: what it leads with
        "example": "truss",
        "output": "report.png",
        "width": 1100,
        "hide": BELOW_RESULTS,
    },
    {
        # a 3D free-body diagram on its own
        "example": "boom",
        "output": "report-3d.png",
        "width": 1100,
        "hide": BELOW_RESULTS
        + ", header.page, #summary, #case-1 > .section-head, #case-1 > .table-wrap,"
        " #case-1 > .caption, #case-1 > h3:not(:last-of-type)",
        # the diagram's heading is now the first thing in its card
        "css": "#case-1 > h3 { margin-top: 0; }",
    },
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


def render(example: str, hide: str, css: str = "") -> str:
    model = load_model(resources.files("engmech") / "examples" / f"{example}.yaml")
    # logo=None: the maintainer's own logo settings must not end up in the README
    html = render_report(model.solve(), source_path=f"{example}.yaml", logo=None)
    style = f"<style>{hide} {{ display: none !important; }} {css}</style>"
    return html.replace("</head>", style + "\n</head>", 1)


def capture(html_path: Path, png_path: Path, width: int, profile: Path) -> None:
    png_path.unlink(missing_ok=True)
    command = [
        chrome(),
        "--headless",
        f"--user-data-dir={profile}",  # never touch the user's own Chrome profile
        "--no-first-run",
        "--hide-scrollbars",
        "--enable-unsafe-swiftshader",  # WebGL for the 3D diagrams
        f"--force-device-scale-factor={SCALE}",
        f"--window-size={width},5000",
        "--virtual-time-budget=15000",  # let plotly finish drawing
        f"--screenshot={png_path}",
        html_path.as_uri(),
    ]
    # Some Chrome builds (seen with 154 on macOS) write the screenshot and then
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
    if not png_path.is_file():
        sys.exit(f"Chrome did not write {png_path.name}")


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
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for shot in SHOTS:
            html_path = tmp / f"{shot['example']}.html"
            html = render(shot["example"], shot["hide"], shot.get("css", ""))
            html_path.write_text(html, encoding="utf-8")
            png_path = OUT / shot["output"]
            capture(html_path, png_path, shot["width"], tmp / "chrome-profile")
            trim(png_path)
            compress(png_path)
            with Image.open(png_path) as im:
                size = f"{im.width}x{im.height}"
            kb = png_path.stat().st_size / 1024
            print(f"wrote {png_path.relative_to(OUT.parent.parent)} ({size}, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
