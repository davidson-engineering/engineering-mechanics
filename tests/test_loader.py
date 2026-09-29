"""Model files: parsing, validation messages with line numbers, schema."""

import json

import pytest

from engmech.io.loader import ModelFileError, loads_model
from engmech.io.schema import json_schema

BEAM = """\
analysis: planar
units: {length: m, force: kN}
parameters:
  P: 12 kN
points:
  A: [0, 0]
  B: [6, 0]
supports:
  A: {type: pin, at: A}
  B: {type: roller, at: B, normal: +y}
loads:
  - {force: [0, -P], at: [2, 0]}
"""


def problems(text):
    """Problems reported for a file, whether found while loading or solving."""
    try:
        model = loads_model(text, "m.yaml")
    except ModelFileError as exc:
        return exc.problems
    with pytest.raises(ModelFileError) as info:
        model.solve()
    return info.value.problems


def test_minimal_file_solves():
    r = loads_model(BEAM).solve()
    assert r.primary["A"].force[1] == pytest.approx(8000)
    assert r.primary["B"].scalars["N"] == pytest.approx(4000)


def test_parameter_override():
    r = loads_model(BEAM).solve({"P": "24 kN"})
    assert r.primary["B"].scalars["N"] == pytest.approx(8000)


def test_unknown_field_suggests_the_right_one():
    text = BEAM.replace("normal: +y", "nromal: +y")
    msgs = problems(text)
    assert any(
        "m.yaml:10:" in p and "unknown field 'nromal'" in p and "'normal'" in p for p in msgs
    )


def test_missing_field_and_unknown_type():
    msgs = problems(BEAM.replace("{type: roller, at: B, normal: +y}", "{type: roller, at: B}"))
    assert any("missing required field 'normal'" in p for p in msgs)
    msgs = problems(BEAM.replace("type: pin", "type: pinn"))
    assert any("m.yaml:9:" in p and "'pinn'" in p for p in msgs)


def test_value_errors_are_located():
    text = BEAM.replace("B: [6, 0]", "B: [6 N, 0]")
    msgs = problems(text)
    assert any(p.startswith("m.yaml:7:") and "is not a length" in p for p in msgs), msgs


def test_build_errors_point_at_the_line():
    msgs = problems(BEAM.replace("normal: +y", "normal: +z"))
    assert msgs[0].startswith("m.yaml:10:"), msgs
    assert "xy-plane" in msgs[0]


def test_invalid_yaml_hint():
    msgs = problems("loads:\n  - {force: [0, -10] kN, at: [0, 0]}\n")
    assert "must be quoted" in msgs[0]


def test_loads_under_bodies_and_distributed():
    text = """\
analysis: planar
bodies:
  beam:
    loads:
      - {distributed: {start: [0, 0], end: [4, 0], intensity: 3 kN/m, direction: -y}}
supports:
  A: {type: pin, body: beam, at: [0, 0]}
  B: {type: roller, body: beam, at: [4, 0], normal: +y}
"""
    r = loads_model(text).solve()
    assert r.primary["B"].scalars["N"] == pytest.approx(6000)


def test_joint_needs_two_bodies():
    text = """\
bodies: {a: {}, b: {}}
joints:
  J: {type: ball, body: a, at: [0, 0, 0]}
"""
    assert any("bodies: [a, b]" in p for p in problems(text))


def test_empty_and_non_mapping_files():
    assert "empty" in problems("")[0]
    assert "mapping" in problems("- 1\n- 2\n")[0]


def test_schema_is_valid_json_schema():
    schema = json_schema()
    json.dumps(schema)
    assert schema["title"] == "engmech model"
    assert "supports" in schema["properties"]


def test_committed_schema_is_current():
    """schema/engmech.schema.json must match the code (run `engmech schema -o ...`)."""
    from pathlib import Path

    committed = Path(__file__).parents[1] / "schema" / "engmech.schema.json"
    assert json.loads(committed.read_text()) == json_schema(), (
        "regenerate with: engmech schema -o schema/engmech.schema.json"
    )
