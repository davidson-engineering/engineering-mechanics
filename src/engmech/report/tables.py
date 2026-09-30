"""Presentation tables shared by the terminal output and the HTML report.

Each builder returns a :class:`Table` of display strings in the chosen
output units, so the terminal and the report always show the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engmech.model import GROUND, BuiltModel
from engmech.results import CaseResult, JointResult, Results
from engmech.solver import ROW_NAMES
from engmech.units import UnitSystem, format_number, unit_text

INDETERMINATE = "indet."


@dataclass
class Cell:
    text: str
    style: str = ""  # "", "muted", "warn", "bad", "good", "strong"
    numeric: bool = False


@dataclass
class Table:
    key: str
    title: str
    headers: list[str]
    rows: list[list[Cell]]
    caption: str = ""
    numeric_from: int = 1  # columns from this index are right-aligned
    notes: list[str] = field(default_factory=list)
    units: str = ""  # units of the numeric columns, shown with the title


def _num(value: float | None, units: UnitSystem, kind: str) -> Cell:
    if value is None:
        return Cell(INDETERMINATE, "warn", True)
    return Cell(units.format(value, kind, unit=False), "", True)


def _clean(values: np.ndarray, scale: float) -> np.ndarray:
    out = np.array(values, dtype=float)
    out[np.abs(out) < 1e-10 * max(scale, 1e-300)] = 0.0
    return out


def component_columns(results: Results) -> list[int]:
    return [0, 1, 5] if results.model.planar else [0, 1, 2, 3, 4, 5]


def _header(c: int, units: UnitSystem) -> str:
    kind = "force" if c < 3 else "moment"
    return f"{ROW_NAMES[c]} ({units.label(kind)})"


def _joint_row(j: JointResult, cols, units, scale, resultant: bool) -> list[Cell]:
    wrench = _clean(j.wrench, scale)
    cells = []
    for c in cols:
        if not j.transmits[c]:
            cells.append(Cell("–", "muted", True))
        else:
            kind = "force" if c < 3 else "moment"
            cells.append(_num(wrench[c] if j.determined[c] else None, units, kind))
    if resultant:
        cells.append(_num(j.component("F"), units, "force"))
    return cells


SCALAR_HEADERS = {"N": "N", "T": "T", "drive": "Drive"}


def _scalar_cell(j: JointResult, units: UnitSystem, results: Results, labelled: bool) -> Cell:
    """Scalar results of a joint (N, T, drive). ``labelled`` adds 'N = ' and units,
    for tables that mix several kinds in one column."""
    geo = results.model.joints[j.name].geometry
    parts, style = [], ""
    for label, value in j.scalars.items():
        prefix = f"{label} = " if labelled else ""
        if not j.scalar_determined[label]:
            parts.append(prefix + INDETERMINATE)
            style = style or "warn"
            continue
        parts.append(prefix + units.format(value, j.scalar_kinds[label], unit=labelled))
        if geo.sign_limit and value < 0:
            style = "bad"
    return Cell("; ".join(parts), style, not labelled)


def units_note(units: UnitSystem, kinds) -> str:
    return ", ".join(dict.fromkeys(units.label(k) for k in kinds if units.label(k)))


def joint_table(results: Results, case: CaseResult, units: UnitSystem, kind: str) -> Table | None:
    """kind: support (reactions from the ground) or joint (between bodies).

    Only components that at least one listed connection can carry get a column."""
    items = [j for j in case.joints.values() if j.kind == kind]
    if not items:
        return None
    cols = [c for c in component_columns(results) if any(j.transmits[c] for j in items)]
    force_cols = [c for c in cols if c < 3]
    resultant = len(force_cols) >= 2
    scale = _case_scale(case)
    labels = sorted({label for j in items for label in j.scalars})
    kinds = {j.scalar_kinds[x] for j in items for x in j.scalars}
    single = len(labels) == 1 and len(kinds) == 1
    show_body = kind == "support" and len(results.model.bodies) > 1
    if kind == "support":
        headers = ["Support", "Type"] + (["Body"] if show_body else [])
        title = "Support reactions"
        caption = "Force and moment on the body from the ground, at the support point."
    else:
        headers = ["Joint", "Type", "On", "From"]
        title = "Joint forces"
        caption = "Force and moment on the 'On' body from the 'From' body, at the joint."
    if any(not j.transmits[c] for j in items for c in cols):
        caption += " A dash means that connection cannot carry that component."
    if "T" in labels:
        caption += " T is the axial force, positive in tension."
    if "N" in labels:
        caption += " N is the normal force, positive when pushing on the body."
    first_numeric = len(headers)
    headers += [ROW_NAMES[c] for c in cols] + (["Resultant"] if resultant else [])
    if single:
        headers.append(SCALAR_HEADERS.get(labels[0], labels[0]))
    elif labels:
        headers.append(" / ".join(SCALAR_HEADERS.get(x, x) for x in labels))
    rows = []
    for j in items:
        row = [Cell(j.name, "strong"), Cell(j.type_name, "muted")]
        if show_body:
            row.append(Cell(j.body_b))
        elif kind == "joint":
            row += [Cell(j.body_b), Cell(j.body_a, "muted")]
        row += _joint_row(j, cols, units, scale, resultant)
        if labels:
            row.append(_scalar_cell(j, units, results, labelled=not single))
        rows.append(row)
    note_kinds = ["force"] + (["moment"] if any(c >= 3 for c in cols) or "moment" in kinds else [])
    table = Table(f"{kind}s", title, headers, rows, caption, numeric_from=first_numeric)
    table.units = units_note(units, note_kinds)
    return table


def envelope_table(results: Results, units: UnitSystem, kind: str) -> Table | None:
    """Every load case and combination side by side, for each component of
    each support (kind 'support') or joint between bodies ('joint'), with the
    largest and smallest value and the case it comes from. Components that
    are zero in every case are left out."""
    cases = list(results.cases.values())
    if len(cases) < 2:
        return None
    joints = [j.name for j in cases[0].joints.values() if j.kind == kind]
    if not joints:
        return None
    scale = max(_case_scale(c) for c in cases)
    rows = []
    for name in joints:
        per_case = [c.joints[name] for c in cases]
        first = per_case[0]
        geo = results.model.joints[name].geometry
        entries = []
        for c in component_columns(results):
            if first.transmits[c]:
                values = [j.wrench[c] if j.determined[c] else None for j in per_case]
                entries.append((ROW_NAMES[c], "force" if c < 3 else "moment", values, False))
        for label in first.scalars:
            values = [j.scalars[label] if j.scalar_determined[label] else None for j in per_case]
            kind_of = first.scalar_kinds[label]
            entries.append((SCALAR_HEADERS.get(label, label), kind_of, values, geo.sign_limit))
        shown = 0
        for label, kind_of, values, one_sided in entries:
            known = [(v, c.name) for v, c in zip(values, cases, strict=True) if v is not None]
            if known and all(abs(v) <= 1e-10 * max(scale, 1e-300) for v, _ in known):
                continue  # zero in every case
            row = [Cell(name if shown == 0 else "", "strong"), Cell(label, "muted")]
            for v in values:
                cell = _num(v, units, kind_of)
                if one_sided and v is not None and v < 0:
                    cell.style = "bad"  # a cable or contact would have to push or pull
                row.append(cell)
            for pick in (max, min):
                if known:
                    v, where = pick(known, key=lambda item: item[0])
                    row.append(Cell(f"{units.format(v, kind_of, unit=False)} ({where})", "", True))
                else:
                    row.append(Cell(INDETERMINATE, "warn", True))
            rows.append(row)
            shown += 1
    if not rows:
        return None
    head = "Support" if kind == "support" else "Joint"
    title = "Support reactions by load case" if kind == "support" else "Joint forces by load case"
    caption = (
        "Each load case and combination side by side; Max and Min are the extreme values "
        "and the case they come from. Components that are zero in every case are left out."
    )
    table = Table(
        f"{kind}-envelope",
        title,
        [head, "", *(c.name for c in cases), "Max", "Min"],
        rows,
        caption,
        numeric_from=2,
    )
    table.units = units_note(units, ["force", "moment"])
    return table


def unknown_table(results: Results, case: CaseResult, units: UnitSystem) -> Table | None:
    items = [j for j in case.joints.values() if j.kind == "unknown"]
    if not items:
        return None
    rows = []
    for j in items:
        label = next(iter(j.scalars))
        value = j.scalars[label] if j.scalar_determined[label] else None
        kind = j.scalar_kinds[label]
        comp = results.model.joints[j.name].geometry.components[0]
        direction = _direction_text(comp.direction, results.model.planar)
        rows.append(
            [
                Cell(label, "strong"),
                Cell(j.body_b),
                Cell("force" if kind == "force" else "couple"),
                Cell(direction),
                _num(value, units, kind),
                Cell(units.label(kind), "muted"),
            ]
        )
    return Table(
        "unknowns",
        "Solved loads",
        ["Load", "Body", "Kind", "Along", "Value", "Unit"],
        rows,
        "Applied loads whose magnitude was solved for (positive along the given direction).",
        numeric_from=4,
    )


def _direction_text(d: np.ndarray, planar: bool) -> str:
    for i, name in enumerate("xyz"):
        if abs(abs(d[i]) - 1) < 1e-9:
            return f"{'+' if d[i] > 0 else '-'}{name}"
    comps = d[:2] if planar else d
    return "(" + ", ".join(f"{x:.4g}" for x in comps) + ")"


def _case_scale(case: CaseResult) -> float:
    s = 0.0
    for w in case.loads:
        s += np.linalg.norm(w.force) + np.linalg.norm(w.moment)
    for j in case.joints.values():
        s += np.linalg.norm(j.force) + np.linalg.norm(j.moment)
    return s


def loads_table(results: Results, case: CaseResult, units: UnitSystem) -> Table:
    planar = results.model.planar
    cols = [0, 1, 5] if planar else [0, 1, 2, 3, 4, 5]
    pos_cols = 2 if planar else 3
    headers = ["Load", "Kind", "Body"]
    headers += [f"{a} ({units.label('length')})" for a in "xyz"[:pos_cols]]
    headers += [_header(c, units) for c in cols]
    rows = []
    for w in case.loads:
        wrench = np.concatenate([w.force, w.moment])
        at = w.point[:pos_cols]
        kind = w.kind
        if w.kind == "distributed":
            d = w.detail
            kind = "distributed" + (" (projected)" if d.get("projected") else "")
            if "centroid" in d:  # show the resultant at its line of action
                at = d["centroid"][:pos_cols]
                wrench = np.concatenate([w.force, np.zeros(3)])
        row = [Cell(w.name, "strong"), Cell(kind, "muted"), Cell(w.body)]
        row += [_num(x, units, "length") for x in at]
        row += [_num(wrench[c], units, "force" if c < 3 else "moment") for c in cols]
        rows.append(row)
    caption = (
        "Each load as a force through the listed point plus a couple. Distributed loads "
        "are shown as their resultant through the centroid of the load; weights act at the "
        "centre of gravity; inertia rows are the d'Alembert loads of prescribed motion."
    )
    return Table("loads", "Applied loads", headers, rows, caption, numeric_from=3)


def balance_table(results: Results, case: CaseResult, units: UnitSystem) -> Table:
    rows = []
    for b in case.balance:
        style = "good" if b.ok else "bad"
        rows.append(
            [
                Cell(b.body, "strong"),
                Cell(units.format(float(np.linalg.norm(b.force)), "force")),
                Cell(units.format(float(np.linalg.norm(b.moment)), "moment")),
                Cell(f"{b.relative:.1e}", "", True),
                Cell("✓ balanced" if b.ok else "✗ NOT balanced", style),
            ]
        )
    return Table(
        "balance",
        "Balance of each body",
        ["Body", "Residual force", "Residual moment", "Relative", "Result"],
        rows,
        "Independent check: every load and joint force on each body summed directly "
        "(not from the solver's matrix). Relative residual must be below 1e-6.",
        numeric_from=1,
    )


def resultant_table(results: Results, case: CaseResult, units: UnitSystem) -> Table:
    cols = component_columns(results)
    headers = ["", *[_header(c, units) for c in cols]]
    scale = _case_scale(case)
    applied = _clean(case.applied_resultant, scale)
    reactions = _clean(case.reaction_resultant, scale)
    total = _clean(applied + reactions, scale)
    rows = []
    for label, vec in (
        ("Applied loads", applied),
        ("Support reactions", reactions),
        ("Sum", total),
    ):
        rows.append(
            [Cell(label, "strong")]
            + [_num(vec[c], units, "force" if c < 3 else "moment") for c in cols]
        )
    return Table(
        "resultants",
        "Overall balance",
        headers,
        rows,
        "Resultants about the origin. Loads include weight and inertia.",
    )


def mass_table(model: BuiltModel, units: UnitSystem) -> Table | None:
    bodies = [b for b in model.bodies.values() if b.mass is not None]
    if not bodies:
        return None
    n = 2 if model.planar else 3
    headers = ["Body", "Mass", *[f"cog {a}" for a in "xyz"[:n]], "I1", "I2", "I3"]
    rows = []
    for b in bodies:
        p = b.mass
        moments, _ = p.principal()
        row = [Cell(b.name, "strong"), _num(p.mass, units, "mass")]
        row += [_num(x, units, "length") for x in p.cog[:n]]
        row += [_num(x, units, "inertia") for x in moments]
        rows.append(row)
    table = Table(
        "mass",
        "Mass properties",
        headers,
        rows,
        "I1 ≤ I2 ≤ I3 are the principal moments of inertia about the centre of gravity.",
    )
    table.units = units_note(units, ["mass", "length", "inertia"])
    return table


def inertia_detail(model: BuiltModel, body_name: str, units: UnitSystem) -> dict:
    b = model.bodies[body_name]
    p = b.mass
    moments, axes = p.principal()
    f = units.factor("inertia")
    return {
        "tensor": [[format_number(x * f) for x in row] for row in p.inertia],
        "moments": [format_number(x * f) for x in moments],
        "axes": [[f"{x:.4f}" for x in axes[:, i]] for i in range(3)],
        "radii": [units.format(x, "length") for x in p.radii_of_gyration()],
        "shapes": [
            (
                s.label,
                units.format(s.mass, "mass"),
                ", ".join(units.format(x, "length", unit=False) for x in s.cog),
            )
            for s in b.shapes
        ],
    }


def check_table(results: Results, units: UnitSystem) -> Table | None:
    if not results.checks:
        return None
    rows = []
    multi = len(results.cases) > 1
    for c in results.checks:
        value = "–" if c.value is None else c.display
        rows.append(
            [
                Cell(c.name, "strong"),
                *([Cell(c.case)] if multi else []),
                Cell(value, "", True),
                Cell(c.detail, "muted"),
                Cell("✓ pass" if c.passed else "✗ FAIL", "good" if c.passed else "bad"),
            ]
        )
    headers = ["Check", *(["Case"] if multi else []), "Value", "Criterion", "Result"]
    return Table("checks", "Checks", headers, rows, numeric_from=len(headers))


def determinacy_rows(results: Results) -> list[tuple[str, str]]:
    a = results.analysis
    rows = [
        ("Equilibrium equations", str(a.equations)),
        ("Unknown reaction components", str(a.unknowns)),
        ("Rank", str(a.rank)),
        ("Free motions (mechanism DOF)", str(a.degrees_of_freedom)),
        ("Degree of indeterminacy", str(a.degree_of_indeterminacy)),
        ("Condition number", f"{a.condition_number:.3g}" if a.rank else "–"),
    ]
    if a.degree_of_indeterminacy == 0 and a.degrees_of_freedom == 0:
        verdict = "statically determinate and stable"
    elif a.degrees_of_freedom and a.degree_of_indeterminacy:
        verdict = "partly a mechanism and partly indeterminate"
    elif a.degrees_of_freedom:
        verdict = "a mechanism (not fully supported)"
    else:
        verdict = "statically indeterminate"
    if any(c.solution.used_stiffness for c in results.cases.values()):
        verdict += "; redundant forces shared by joint stiffness (least work)"
    rows.insert(0, ("Structure", verdict))
    return rows


def dropped_notes(results: Results) -> list[str]:
    """Components the user explicitly asked for that the analysis ignores."""
    sys = results.system
    out = []
    grouped: dict[str, list[str]] = {}
    for joint, k, reason in sys.dropped:
        if results.model.joints[joint].type_name != "custom":
            continue
        comp = results.model.joints[joint].geometry.components[k]
        grouped.setdefault(reason, []).append(f"{joint} ({comp.kind})")
    for reason, items in grouped.items():
        if reason == "out of plane":
            out.append(
                "Out-of-plane components are not part of a planar analysis: "
                + ", ".join(sorted(set(items)))
            )
        elif reason == "particle":
            out.append("Particles carry no moments: " + ", ".join(sorted(set(items))))
    return out


def status_summary(results: Results, case: CaseResult) -> tuple[str, str]:
    """(style, text) headline for one case."""
    if case.status == "unbalanced":
        if any(b.motion is not None for b in results.model.bodies.values()):
            return "bad", "Not in dynamic equilibrium with the prescribed motion"
        return "bad", "Not in equilibrium: the supports cannot resist these loads"
    if case.status == "indeterminate":
        return "warn", "Statically indeterminate: some reactions need stiffness to resolve"
    if not case.verified:
        return "bad", f"Equilibrium check failed (residual {case.max_residual:.1e})"
    return "good", f"Equilibrium verified (max relative residual {case.max_residual:.0e})"


def is_ground(name: str) -> bool:
    return name == GROUND


def _point_text(p: np.ndarray, units: UnitSystem, planar: bool) -> str:
    comps = p[:2] if planar else p
    return "(" + ", ".join(units.format(x, "length", unit=False) for x in comps) + ")"


def _transmits(j, model: BuiltModel) -> str:
    geo = j.geometry
    planar = model.planar
    parts = []
    scalar = [c for c in geo.components if c.role == "scalar"]
    if scalar:
        c = scalar[0]
        if c.label == "T":
            return "axial force T (tension +)"
        return f"{c.label} along {_direction_text(c.direction, planar)}"
    active = np.zeros(6, bool)
    for c in geo.components:
        if c.role != "constraint":
            continue
        if planar and (c.kind == "force") == (abs(c.direction[2]) > 1e-9):
            continue  # out of plane
        off = 0 if c.kind == "force" else 3
        active[off : off + 3] |= np.abs(c.direction) > 1e-12
    names = [n for i, n in enumerate(ROW_NAMES) if active[i]]
    if planar:
        names = [n for n in names if n in ("Fx", "Fy", "Mz")]
    parts.append(", ".join(names) if names else "nothing")
    drives = [c for c in geo.components if c.role == "drive"]
    for c in drives:
        what = "torque" if c.kind == "moment" else "force"
        parts.append(f"drive {what} about/along {_direction_text(c.direction, planar)}")
    return "; ".join(parts)


def definition_table(model: BuiltModel, units: UnitSystem) -> Table | None:
    joints = [j for j in model.joints.values() if j.kind != "unknown"]
    if not joints:
        return None
    show_body = len(model.bodies) > 1
    rows = []
    notes_any = False
    for j in joints:
        geo = j.geometry
        bodies = j.body_b if j.kind == "support" else f"{j.body_a} – {j.body_b}"
        loc = _point_text(geo.point_b, units, model.planar)
        if not np.allclose(geo.point_a, geo.point_b):
            loc = f"{_point_text(geo.point_a, units, model.planar)} → {loc}"
        extra = []
        if any(c.compliance > 0 for c in geo.components):
            extra.append("elastic")
        if geo.sign_limit:
            extra.append("push only" if geo.sign_limit == "compression" else "pull only")
        notes_any |= bool(extra)
        row = [
            Cell(j.name, "strong"),
            Cell("support" if j.kind == "support" else "joint", "muted"),
            Cell(j.type_name),
        ]
        if show_body:
            row.append(Cell(bodies))
        row += [Cell(loc), Cell(_transmits(j, model), "muted"), Cell(", ".join(extra), "muted")]
        rows.append(row)
    headers = [
        "Name",
        "Kind",
        "Type",
        *(["Bodies"] if show_body else []),
        f"Location ({units.label('length')})",
        "Transmits",
        "Notes",
    ]
    if not notes_any:
        headers = headers[:-1]
        rows = [r[:-1] for r in rows]
    return Table("definitions", "Supports and joints", headers, rows, numeric_from=99)


def points_table(model: BuiltModel, units: UnitSystem) -> Table | None:
    if not model.points:
        return None
    n = 2 if model.planar else 3
    headers = ["Point", *[f"{a} ({units.label('length')})" for a in "xyz"[:n]]]
    rows = [
        [Cell(name, "strong"), *[_num(x, units, "length") for x in p[:n]]]
        for name, p in model.points.items()
    ]
    return Table("points", "Points", headers, rows)


def parameters_table(model: BuiltModel) -> Table | None:
    if not model.parameters:
        return None
    rows = []
    for name, q in model.parameters.items():
        unit = "" if q.unitless else unit_text(q.units)
        rows.append(
            [
                Cell(name, "strong"),
                Cell(format_number(float(q.magnitude)), "", True),
                Cell(unit, "muted"),
            ]
        )
    return Table("parameters", "Parameters", ["Name", "Value", "Unit"], rows, numeric_from=1)
