"""The shipped validation suite passes, and fails when the solver is wrong."""

import numpy as np
import pytest

from engmech import solver, validation


def test_validation_suite_passes():
    run = validation.run(n=10)
    failed = [(o.name, o.detail) for o in run.outcomes if not o.passed]
    assert not failed
    summary = run.summary()
    assert summary["benchmark"]["cases"] >= 22
    assert summary["property"]["cases"] == len(validation.PROPERTIES)


def _original_couple_bug(system, factors):
    """The pre-0.2 bug: an extra r x M for every couple."""
    b = np.zeros(len(system.rows))
    index = {key: i for i, key in enumerate(system.rows)}
    for w in system.model.loads:
        f = factors.get(w.case, 0.0)
        r = w.point - system.ref
        full = np.concatenate([w.force, w.moment + np.cross(r, w.force) + np.cross(r, w.moment)])
        for c in range(6):
            i = index.get((w.body, c))
            if i is not None:
                b[i] -= f * full[c]
    return b


def _sign_error_on_distributed(system, factors):
    """A plausible slip: the first moment of a distributed load with the wrong sign."""
    b = solver.__dict__["_true_load_vector"](system, factors)
    for w in system.model.loads:
        if w.kind == "distributed":
            return -b
    return b


@pytest.mark.parametrize("fault", [_original_couple_bug, _sign_error_on_distributed])
def test_validation_catches_injected_faults(monkeypatch, fault):
    monkeypatch.setitem(solver.__dict__, "_true_load_vector", solver.load_vector)
    monkeypatch.setattr(solver, "load_vector", fault)
    run = validation.run(n=5)
    assert not run.passed
    assert any(o.suite == "benchmark" and not o.passed for o in run.outcomes)


def test_validation_report_and_json(tmp_path):
    from click.testing import CliRunner

    from engmech.cli import main

    report = tmp_path / "validation.html"
    result = CliRunner().invoke(main, ["validate", "--report", str(report), "--cases", "5"])
    assert result.exit_code == 0, result.output
    html = report.read_text(encoding="utf-8")
    assert "Software validation report" in html
    assert "PASS" in html
    js = CliRunner().invoke(main, ["validate", "--json", "-", "--cases", "5"])
    import json

    data = json.loads(js.output)
    assert data["passed"] is True
    assert data["environment"]["engmech"]
