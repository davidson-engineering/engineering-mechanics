"""Report branding: the logo shown in the report header.

The logo is chosen from the most specific setting available:

1. the ``logo`` argument (the CLI's ``--logo`` / ``--no-logo``);
2. the model file's ``report: {logo: ...}`` (paths relative to the file);
3. the ``ENGMECH_LOGO`` environment variable;
4. ``[report] logo`` in the user config file (``engmech config`` shows where).

With none of these set, reports have no logo.

A logo is a local image file or an http(s) URL. It is embedded in the
report as a data URI, so reports stay self-contained and open offline.
Use ``none`` at any level to show no logo.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import click

from engmech.errors import InputError

MAX_BYTES = 2_000_000
TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}
NONE = {"none", "off", "false", "no", ""}
UNSET = object()  # "no logo argument given" (None means "no logo")


@dataclass(frozen=True)
class Logo:
    data_uri: str
    source: str  # where it came from, for messages and tests


def config_path() -> Path:
    """The user config file (override with the ENGMECH_CONFIG variable)."""
    if os.environ.get("ENGMECH_CONFIG"):
        return Path(os.environ["ENGMECH_CONFIG"])
    return Path(click.get_app_dir("engmech")) / "config.toml"


def user_config() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        return tomllib.loads(text)
    except UnicodeDecodeError as exc:
        raise InputError(f"{path}: invalid config file: {exc}") from None
    except tomllib.TOMLDecodeError as exc:
        hint = ""
        if re.search(r'"[^"\n]*\\', text):  # the usual cause: a Windows path in "..."
            hint = (
                '; a backslash inside "..." starts an escape sequence in TOML, so write '
                "Windows paths in single quotes, as in logo = 'C:\\branding\\logo.png', "
                "or with forward slashes"
            )
        raise InputError(f"{path}: invalid config file: {exc}{hint}") from None


def resolve_logo(argument=UNSET, model_logo=None, model_dir: Path | None = None) -> Logo | None:
    """The logo to show, from the most specific setting that is present."""
    if argument is not UNSET:
        return load_logo(argument, Path.cwd(), "--logo")
    if model_logo is not None:
        return load_logo(model_logo, model_dir or Path.cwd(), "report.logo in the model file")
    if os.environ.get("ENGMECH_LOGO") is not None:
        return load_logo(os.environ["ENGMECH_LOGO"], Path.cwd(), "ENGMECH_LOGO")
    config = user_config().get("report", {})
    if "logo" in config:
        path = config_path()
        return load_logo(config["logo"], path.parent, str(path))
    return None


def load_logo(spec, base_dir: Path, where: str) -> Logo | None:
    if spec is None or spec is False or str(spec).strip().lower() in NONE:
        return None
    text = str(spec).strip()
    if text.lower().startswith(("http://", "https://")):
        data, mime = _fetch(text, where)
    else:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        if not path.is_file():
            raise InputError(f"{where}: logo file not found: {path}")
        mime = TYPES.get(path.suffix.lower())
        if mime is None:
            raise InputError(
                f"{where}: unsupported logo type {path.suffix!r} (use {', '.join(sorted(TYPES))})"
            )
        data = path.read_bytes()
    if len(data) > MAX_BYTES:
        raise InputError(
            f"{where}: logo is {len(data) / 1e6:.1f} MB; use an image under "
            f"{MAX_BYTES / 1e6:.0f} MB (it is embedded in every report)"
        )
    return Logo(_data_uri(data, mime), text)


def _fetch(url: str, where: str) -> tuple[bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = response.read(MAX_BYTES + 1)
            mime = response.headers.get_content_type()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise InputError(f"{where}: could not download the logo from {url} ({exc})") from None
    if not mime.startswith("image/"):
        guessed = mimetypes.guess_type(url)[0]
        if not (guessed and guessed.startswith("image/")):
            raise InputError(f"{where}: {url} is not an image (content type {mime})")
        mime = guessed
    return data, mime


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
