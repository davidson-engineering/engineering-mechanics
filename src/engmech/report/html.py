"""Self-contained HTML report."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup
from plotly.offline import get_plotlyjs

from engmech import __version__
from engmech.report import tables as t
from engmech.report.figure import model_figure
from engmech.results import Results
from engmech.units import UnitSystem

_env = Environment(
    loader=PackageLoader("engmech", "report/templates"),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_report(
    results: Results, units=None, source_path: str | None = None, plotly_cdn: bool = False
) -> str:
    units = results.model.output_units if units is None else UnitSystem.from_spec(units)
    model = results.model
    cases = []
    for name, case in results.cases.items():
        fig = model_figure(results, case=name, units=units)
        style, headline = t.status_summary(results, case)
        cases.append(
            {
                "name": name,
                "case": case,
                "style": style,
                "headline": headline,
                "figure": Markup(fig.to_json()),
                "tables": [
                    x
                    for x in (
                        t.joint_table(results, case, units, "support"),
                        t.joint_table(results, case, units, "joint"),
                        t.unknown_table(results, case, units),
                    )
                    if x
                ],
                "verification": [
                    t.balance_table(results, case, units),
                    t.resultant_table(results, case, units),
                ],
                "loads": t.loads_table(results, case, units),
                "factors": " + ".join(f"{f:g} × {c}" for c, f in case.factors.items()),
                "excited": [
                    [m.describe(units, model.planar) for m in mode] for mode in case.excited_modes
                ],
                "redundancy": [
                    ", ".join(f"{j}.{c} ({w:+.3g})" for j, c, w in mode) for mode in case.redundancy
                ],
            }
        )
    mechanism = []
    if results.analysis.degrees_of_freedom and results.status != "unbalanced":
        mechanism = [
            [m.describe(units, model.planar) for m in mode] for mode in results.mechanism[:6]
        ]
    bodies_with_mass = [n for n, b in model.bodies.items() if b.mass is not None]
    dynamics = any(b.motion is not None for b in model.bodies.values())
    n_sup = sum(1 for j in model.joints.values() if j.kind == "support")
    n_joint = sum(1 for j in model.joints.values() if j.kind == "joint")
    facts = [
        "Planar (xy)" if model.planar else "Spatial (3D)",
        "Dynamics (Newton-Euler)" if dynamics else "Statics",
        f"{len(model.bodies)} bod{'y' if len(model.bodies) == 1 else 'ies'}",
        f"{n_sup} support{'s' * (n_sup != 1)}",
    ]
    if n_joint:
        facts.append(f"{n_joint} joint{'s' * (n_joint != 1)}")
    facts.append(f"{units.label('force')}, {units.label('length')}, {units.label('moment')}")

    template = _env.get_template("report.html.j2")
    return template.render(
        title=model.name,
        description=_paragraphs(model.description),
        facts=facts,
        source=source_path,
        generated=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        version=__version__,
        results=results,
        overall_ok=results.ok,
        status=results.status,
        cases=cases,
        multi=len(cases) > 1,
        checks=t.check_table(results, units),
        notes=results.notes + t.dropped_notes(results),
        mechanism=mechanism,
        determinacy=t.determinacy_rows(results),
        mass=t.mass_table(model, units),
        inertia={n: t.inertia_detail(model, n, units) for n in bodies_with_mass},
        inertia_unit=units.label("inertia"),
        definitions=t.definition_table(model, units),
        points=t.points_table(model, units),
        parameters=t.parameters_table(model),
        units=units,
        planar=model.planar,
        gravity=_gravity_text(model, units),
        provenance=results.provenance(),
        source_text=model.source_text,
        plotly_js=None if plotly_cdn else Markup(get_plotlyjs()),
    )


def write_report(results: Results, path, units=None, source_path=None, plotly_cdn=False) -> Path:
    path = Path(path)
    path.write_text(render_report(results, units, source_path, plotly_cdn), encoding="utf-8")
    return path


def _paragraphs(text: str) -> list[tuple[str, str]]:
    """Split a description into ('p', text) paragraphs, reflowing hard-wrapped
    lines, and ('pre', text) blocks for indented lines such as equations."""
    out: list[tuple[str, str]] = []
    for block in (text or "").strip().split("\n\n"):
        lines = block.split("\n")
        prose, pre = [], []
        for line in lines:
            if line.startswith((" ", "\t")):
                if prose:
                    out.append(("p", " ".join(prose)))
                    prose = []
                pre.append(line.strip())
            else:
                if pre:
                    out.append(("pre", "\n".join(pre)))
                    pre = []
                prose.append(line.strip())
        if prose:
            out.append(("p", " ".join(prose)))
        if pre:
            out.append(("pre", "\n".join(pre)))
    return out


def _gravity_text(model, units: UnitSystem) -> str | None:
    if model.gravity is None:
        return None
    g = model.gravity[:2] if model.planar else model.gravity
    return (
        "("
        + ", ".join(units.format(x, "acceleration", unit=False) for x in g)
        + ") "
        + (units.label("acceleration"))
    )


def render_validation(run) -> str:
    """HTML report of a validation run (see engmech.validation)."""
    return _env.get_template("validation.html.j2").render(run=run, summary=run.summary())
