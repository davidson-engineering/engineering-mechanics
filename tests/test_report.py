from importlib import resources

import pytest

from engmech.io.loader import load_model
from engmech.report.html import render_report

EXAMPLES = resources.files("engmech") / "examples"


@pytest.mark.parametrize(
    "name", ["beam", "frame", "boom", "load-combinations", "gyroscope", "truss"]
)
def test_report_renders_every_section(name):
    results = load_model(str(EXAMPLES / f"{name}.yaml")).solve()
    html = render_report(results, plotly_cdn=True)
    for heading in (
        "Summary",
        "Support reactions",
        "Equilibrium verification",
        "Model",
        "Method and conventions",
    ):
        assert heading in html
    assert html.count("Plotly.newPlot") == len(results.cases)


def test_figure_views_for_multibody():
    results = load_model(str(EXAMPLES / "frame.yaml")).solve()
    fig = results.figure()
    buttons = fig.layout.updatemenus[0].buttons
    assert [b.label for b in buttons] == ["Whole model", "Free body: left", "Free body: right"]


def test_indeterminate_report_explains_itself():
    from engmech import Force, Model, Pin

    m = Model("two pins", planar=True)
    m.support("A", Pin(at=[0, 0]))
    m.support("B", Pin(at=[4, 0]))
    m.load(Force([3, -10], at=[1, 0]))
    html = render_report(m.solve(), plotly_cdn=True)
    assert "Statically indeterminate" in html
    assert "indet." in html
