import csv
import json

import pytest
from click.testing import CliRunner

from engmech.cli import main

BEAM = """\
name: Test beam
analysis: planar
units: {length: m, force: kN}
parameters:
  P: 12 kN
supports:
  A: {type: pin, at: [0, 0]}
  B: {type: roller, at: [6, 0], normal: +y}
loads:
  - {force: [0, -P], at: [2, 0]}
checks:
  - {target: B.N, max: 5 kN}
"""


@pytest.fixture
def beam(tmp_path):
    path = tmp_path / "beam.yaml"
    path.write_text(BEAM, encoding="utf-8")
    return path


def run(*args):
    return CliRunner().invoke(main, [str(a) for a in args], catch_exceptions=False)


def test_solve_prints_reactions(beam):
    result = run("solve", beam)
    assert result.exit_code == 0, result.output
    assert "Support reactions" in result.output
    assert "Equilibrium verified" in result.output


def test_failed_check_sets_exit_code(beam):
    result = run("solve", beam, "--set", "P=24 kN")
    assert result.exit_code == 2
    assert "FAIL" in result.output


def test_json_output(beam):
    result = run("solve", beam, "--json", "-")
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["cases"]["default"]["joints"]["A"]["components"]["Fy"] == pytest.approx(8)
    assert data["units"]["force"] == "kN"


def test_units_override(beam):
    data = json.loads(run("solve", beam, "--json", "-", "--units", "US-in").output)
    assert data["cases"]["default"]["joints"]["A"]["components"]["Fy"] == pytest.approx(
        8000 / 4.4482216152605
    )


def test_report(beam, tmp_path):
    out = tmp_path / "r.html"
    result = run("report", beam, "-o", out, "--cdn")
    assert result.exit_code == 0
    html = out.read_text(encoding="utf-8")
    assert "<h1>Test beam</h1>" in html
    assert "Support reactions" in html
    assert "cdn.plot.ly" in html


def test_check_and_mass(beam, tmp_path):
    result = run("check", beam)
    assert "statically determinate and stable" in result.output
    massy = tmp_path / "m.yaml"
    massy.write_text(
        "bodies:\n  b:\n    shapes: [{type: box, mass: 2, size: [1, 1, 1], center: [0, 0, 0]}]\n",
        encoding="utf-8",
    )
    result = run("mass", massy, "--about", "[0, 0, 1]")
    assert result.exit_code == 0, result.output
    assert "0.3333" in result.output


def test_sweep_to_csv(beam, tmp_path):
    out = tmp_path / "s.csv"
    result = run("sweep", beam, "--param", "P=0 kN:12 kN:4", "--output", "A.Fy,B.N", "--csv", out)
    assert result.exit_code == 0, result.output
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["P (kN)", "A.Fy (kN)", "B.N (kN)"]
    assert [float(r[2]) for r in rows[1:]] == pytest.approx([0, 1.333333, 2.666667, 4])


def test_input_errors_exit_1(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("supports:\n  A: {type: pin, att: [0, 0]}\n", encoding="utf-8")
    result = CliRunner().invoke(main, ["solve", str(bad)])
    assert result.exit_code == 1
    assert "bad.yaml:2:" in result.output
    assert "did you mean 'at'" in result.output


def test_examples_commands(tmp_path):
    listing = run("examples", "list")
    assert "beam" in listing.output
    result = run("examples", "copy", "beam", tmp_path)
    assert (tmp_path / "beam.yaml").exists()
    assert run("solve", tmp_path / "beam.yaml").exit_code == 0
    assert result.exit_code == 0


def test_schema_command(tmp_path):
    out = tmp_path / "schema.json"
    run("schema", "-o", out)
    assert json.loads(out.read_text(encoding="utf-8"))["title"] == "engmech model"
