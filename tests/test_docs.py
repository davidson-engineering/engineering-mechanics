"""Documentation snippets must stay valid."""

import re
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError

from engmech.io.loader import loads_model

ROOT = Path(__file__).parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def yaml_blocks(path: Path) -> list[str]:
    return re.findall(r"```yaml\n(.*?)```", path.read_text(), re.S)


def parse(block: str) -> None:
    """Parse a snippet. Blocks listing alternatives repeat a key on purpose;
    those are checked one top-level line at a time."""
    yaml = YAML(typ="safe")
    try:
        yaml.load(block)
    except DuplicateKeyError:
        for line in block.splitlines():
            if line.strip() and not line.startswith((" ", "#")):
                yaml.load(line)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_yaml_snippets_parse(path):
    blocks = yaml_blocks(path)
    assert blocks
    for block in blocks:
        parse(block)


def test_readme_quick_start_solves():
    block = yaml_blocks(ROOT / "README.md")[0]
    results = loads_model(block).solve()
    assert results.primary["A"].force[1] == pytest.approx(11_000)
    assert results.primary["B"].scalars["N"] == pytest.approx(13_000)


def test_reference_minimal_file_solves():
    block = yaml_blocks(ROOT / "docs" / "input-format.md")[0]
    assert loads_model(block).solve().status == "ok"
