"""Every bundled example solves, verifies equilibrium and passes its hand checks.

The examples double as the verification suite: each one's ``checks`` hold
answers derived by hand in its description.
"""

from importlib import resources

import pytest

from engmech.io.loader import load_model

EXAMPLES = sorted(
    p for p in (resources.files("engmech") / "examples").iterdir() if p.name.endswith(".yaml")
)
VERIFICATION = sorted(
    p for p in (resources.files("engmech") / "benchmarks").iterdir() if p.name.endswith(".yaml")
)


@pytest.mark.parametrize("path", EXAMPLES + VERIFICATION, ids=lambda p: p.name)
def test_example(path):
    results = load_model(str(path)).solve()
    assert results.status == "ok", [c.warnings for c in results.cases.values()]
    for case in results.cases.values():
        assert case.verified, (case.name, case.max_residual)
    assert results.checks, "every example should carry hand-derived checks"
    failed = [(c.name, c.case, c.value, c.detail) for c in results.checks if not c.passed]
    assert not failed
