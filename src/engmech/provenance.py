"""Where a result came from: software versions, platform and the exact input.

Engineering records need to show which program, which version and which
input produced a number. Every report and JSON export carries this block.
"""

from __future__ import annotations

import datetime as dt
import platform
import sys
from importlib import metadata

from engmech import __version__
from engmech.model import BuiltModel

LIBRARIES = ("numpy", "scipy", "pint", "pydantic", "ruamel.yaml", "plotly")


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment() -> dict:
    """The software and platform in use."""
    return {
        "engmech": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": {name: _version(name) for name in LIBRARIES},
    }


def provenance(model: BuiltModel) -> dict:
    """A JSON-serialisable record of how a result was produced."""
    record = {
        **environment(),
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "input": None,
        "parameter_overrides": dict(model.overrides),
    }
    if model.source_sha256 is not None:
        record["input"] = {
            "path": model.source_path,
            "sha256": model.source_sha256,
            "bytes": model.source_size,
        }
    return record
