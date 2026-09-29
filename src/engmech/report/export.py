"""Machine-readable results (JSON-compatible dictionaries, CSV rows)."""

from __future__ import annotations

from typing import Any

import numpy as np

from engmech import __version__
from engmech.results import Results
from engmech.solver import ROW_NAMES
from engmech.units import UnitSystem


def _vec(v, factor: float) -> list[float]:
    return [float(x) * factor for x in v]


def results_to_dict(results: Results, units=None) -> dict[str, Any]:
    """All results in the given display units (default: the model's output units)."""
    units = results.model.output_units if units is None else UnitSystem.from_spec(units)
    fF, fM, fL = units.factor("force"), units.factor("moment"), units.factor("length")
    m = results.model
    a = results.analysis
    out: dict[str, Any] = {
        "engmech": __version__,
        "model": m.name,
        "analysis": "planar" if m.planar else "spatial",
        "units": {k: units.label(k) for k in ("length", "force", "moment", "mass", "inertia")},
        "status": results.status,
        "provenance": results.provenance(),
        "structure": {
            "equations": a.equations,
            "unknowns": a.unknowns,
            "rank": a.rank,
            "mechanism_dof": a.degrees_of_freedom,
            "degree_of_indeterminacy": a.degree_of_indeterminacy,
            "condition_number": a.condition_number if a.rank else None,
        },
        "notes": list(results.notes),
        "sensitivity": results.sensitivity,
        "bodies": {},
        "cases": {},
        "checks": [
            {
                "name": c.name,
                "target": c.target,
                "case": c.case,
                "passed": c.passed,
                "value": None if c.value is None else c.value * units.factor(c.kind),
                "detail": c.detail,
            }
            for c in results.checks
        ],
    }
    for name, b in m.bodies.items():
        entry: dict[str, Any] = {"particle": b.particle}
        if b.mass is not None:
            moments, axes = b.mass.principal()
            fI = units.factor("inertia")
            entry.update(
                mass=b.mass.mass * units.factor("mass"),
                cog=_vec(b.mass.cog, fL),
                inertia=[_vec(row, fI) for row in b.mass.inertia],
                principal_moments=_vec(moments, fI),
                principal_axes=[_vec(axes[:, i], 1.0) for i in range(3)],
            )
        out["bodies"][name] = entry
    for name, case in results.cases.items():
        joints = {}
        for jname, j in case.joints.items():
            comps = {}
            for i, c in enumerate(ROW_NAMES):
                if not j.active[i]:
                    continue
                f = fF if i < 3 else fM
                comps[c] = float(j.wrench[i]) * f if j.determined[i] else None
            joints[jname] = {
                "type": j.type_name,
                "kind": j.kind,
                "body": j.body_b,
                "from": j.body_a,
                "point": _vec(j.point, fL),
                "components": comps,
                "scalars": {
                    k: (
                        v * (fF if j.scalar_kinds[k] == "force" else fM)
                        if j.scalar_determined[k]
                        else None
                    )
                    for k, v in j.scalars.items()
                },
            }
        out["cases"][name] = {
            "kind": case.kind,
            "factors": case.factors,
            "status": case.status,
            "verified": case.verified,
            "max_residual": case.max_residual,
            "warnings": list(case.warnings),
            "joints": joints,
        }
    return out


def flat_values(results: Results, case: str | None = None, units=None) -> dict[str, float | None]:
    """{"A.Fy": value, "B.N": value, ...} for one case, in display units."""
    units = results.model.output_units if units is None else UnitSystem.from_spec(units)
    c = results.primary if case is None else results[case]
    out: dict[str, float | None] = {}
    for jname, j in c.joints.items():
        for i, comp in enumerate(ROW_NAMES):
            if j.active[i]:
                v = j.component(comp)
                out[f"{jname}.{comp}"] = (
                    None if v is None else v * units.factor("force" if i < 3 else "moment")
                )
        for label, v in j.scalars.items():
            kind = j.scalar_kinds[label]
            out[f"{jname}.{label}"] = v * units.factor(kind) if j.scalar_determined[label] else None
    return {k: (None if v is None else float(np.round(v, 12))) for k, v in out.items()}
