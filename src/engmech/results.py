"""Solution objects: joint forces, equilibrium verification, warnings, checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engmech.diagnostics import (
    BodyMotion,
    describe_mechanism_mode,
    describe_redundancy_mode,
    kinematic_warnings,
)
from engmech.errors import InputError
from engmech.loads import AppliedWrench
from engmech.model import GROUND, BuiltModel, ResolvedCheck
from engmech.solver import ROW_NAMES, Analysis, CaseSolution, System

EQUILIBRIUM_TOL = 1e-6


@dataclass
class JointResult:
    """The wrench a joint applies to body ``b`` (from body ``a``, or the
    ground for supports), about the joint point, in global axes."""

    name: str
    type_name: str
    kind: str  # support | joint | unknown
    body_a: str
    body_b: str
    point: np.ndarray
    point_a: np.ndarray
    force: np.ndarray
    moment: np.ndarray
    determined: np.ndarray  # bool per Fx..Mz
    scalars: dict[str, float] = field(default_factory=dict)
    scalar_kinds: dict[str, str] = field(default_factory=dict)
    scalar_determined: dict[str, bool] = field(default_factory=dict)
    active: np.ndarray = field(default_factory=lambda: np.ones(6, bool))  # analysed comps
    transmits: np.ndarray = field(default_factory=lambda: np.ones(6, bool))  # can be non-zero

    def component(self, name: str) -> float | None:
        """Value in SI, or None if statically indeterminate."""
        if name in ROW_NAMES:
            i = ROW_NAMES.index(name)
            return float(self.wrench[i]) if self.determined[i] else None
        if name == "F":
            return float(np.linalg.norm(self.force)) if self.determined[:3].all() else None
        if name == "M":
            return float(np.linalg.norm(self.moment)) if self.determined[3:].all() else None
        if name in self.scalars:
            return self.scalars[name] if self.scalar_determined[name] else None
        raise KeyError(f"{self.name} has no component {name!r}")

    @property
    def wrench(self) -> np.ndarray:
        return np.concatenate([self.force, self.moment])

    @property
    def fully_determined(self) -> bool:
        return bool(self.determined.all()) and all(self.scalar_determined.values())


@dataclass
class BodyBalance:
    body: str
    force: np.ndarray  # residual force
    moment: np.ndarray  # residual moment about the body reference point
    relative: float
    ok: bool


@dataclass
class CheckResult:
    name: str
    target: str
    case: str
    kind: str
    value: float | None
    passed: bool
    detail: str
    display: str = ""  # the value, in the unit the check was written in


@dataclass
class CaseResult:
    name: str
    kind: str  # case | combination
    factors: dict[str, float]
    status: str  # ok | indeterminate | unbalanced
    joints: dict[str, JointResult]
    loads: list[AppliedWrench]  # already factored
    balance: list[BodyBalance]
    solution: CaseSolution
    warnings: list[str]
    excited_modes: list[list[BodyMotion]]
    redundancy: list[list[tuple[str, str, float]]]
    applied_resultant: np.ndarray  # (Fx..Mz) about the origin
    reaction_resultant: np.ndarray

    def __getitem__(self, joint: str) -> JointResult:
        try:
            return self.joints[joint]
        except KeyError:
            raise KeyError(f"no support or joint named {joint!r}") from None

    @property
    def verified(self) -> bool:
        return all(b.ok for b in self.balance)

    @property
    def max_residual(self) -> float:
        return max((b.relative for b in self.balance), default=0.0)


@dataclass
class Results:
    model: BuiltModel
    system: System
    analysis: Analysis
    cases: dict[str, CaseResult]
    mechanism: list[list[BodyMotion]]
    notes: list[str]
    checks: list[CheckResult]

    def __getitem__(self, name: str) -> CaseResult:
        try:
            return self.cases[name]
        except KeyError:
            raise KeyError(f"no load case or combination {name!r}") from None

    @property
    def primary(self) -> CaseResult:
        """The first load case (the only one in most models)."""
        return next(iter(self.cases.values()))

    def joint(self, name: str, case: str | None = None) -> JointResult:
        return (self.primary if case is None else self[case])[name]

    @property
    def status(self) -> str:
        order = {"ok": 0, "indeterminate": 1, "unbalanced": 2}
        return max((c.status for c in self.cases.values()), key=order.__getitem__)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and all(c.passed for c in self.checks)

    # ---------------------------------------------------------------- output

    def show(self, console=None, units=None, verbose: bool = False) -> None:
        from engmech.report.terminal import print_results

        print_results(self, console=console, units=units, verbose=verbose)

    def to_dict(self, units=None) -> dict:
        from engmech.report.export import results_to_dict

        return results_to_dict(self, units)

    def report(self, path, units=None, **kwargs) -> None:
        from engmech.report.html import write_report

        write_report(self, path, units=units, **kwargs)

    def figure(self, case: str | None = None, **kwargs):
        from engmech.report.figure import model_figure

        return model_figure(self, case=case, **kwargs)


# --------------------------------------------------------------------------- building


def build_results(system: System, analysis: Analysis, solutions: dict) -> Results:
    model = system.model
    mechanism = [
        describe_mechanism_mode(system, analysis.mechanism_modes[:, k])
        for k in range(analysis.mechanism_modes.shape[1])
    ]
    cases = {}
    for name, (kind, sol) in solutions.items():
        factors = {name: 1.0} if kind == "case" else model.combinations[name]
        cases[name] = _case_result(system, analysis, name, kind, factors, sol)

    notes = []
    if analysis.degrees_of_freedom > 0 and all(c.solution.balanced for c in cases.values()):
        notes.append(
            f"The model is a mechanism with {analysis.degrees_of_freedom} degree(s) of freedom, "
            "but the loads do no work on it, so equilibrium still holds."
        )
    notes += kinematic_warnings(model)
    results = Results(model, system, analysis, cases, mechanism, notes, [])
    results.checks = evaluate_checks(results)
    return results


def _case_result(system, analysis, name, kind, factors, sol: CaseSolution) -> CaseResult:
    model = system.model
    L = system.length
    by_joint: dict[str, list[int]] = {}
    for i, u in enumerate(system.unknowns):
        by_joint.setdefault(u.joint, []).append(i)
    free = sol.free_modes
    joints = {}
    for jname, joint in model.joints.items():
        geo = joint.geometry
        force, moment = np.zeros(3), np.zeros(3)
        null_w = np.zeros((6, free.shape[1]))
        transmits = np.zeros(6, bool)
        scalars, scalar_kinds, scalar_det = {}, {}, {}
        for i in by_joint.get(jname, []):
            u = system.unknowns[i]
            c = u.component
            value = sol.values[i]
            null_row = system.col_scale[i] * free[i] if free.size else np.zeros(0)
            offset = 0 if c.kind == "force" else 3
            transmits[offset : offset + 3] |= np.abs(c.direction) > 1e-12
            if c.kind == "force":
                force += value * c.direction
                if free.size:
                    null_w[:3] += np.outer(c.direction, null_row)
            else:
                moment += value * c.direction
                if free.size:
                    null_w[3:] += np.outer(c.direction, null_row) / L
            if c.label:
                scalars[c.label] = float(value)
                scalar_kinds[c.label] = c.kind
                scalar_det[c.label] = bool(sol.determined[i])
        for jn, k, _ in system.dropped:
            if jn != jname:
                continue
            c = geo.components[k]
            if c.label:
                scalars[c.label], scalar_kinds[c.label], scalar_det[c.label] = 0.0, c.kind, True
        determined = np.linalg.norm(null_w, axis=1) <= 1e-8 if free.size else np.ones(6, bool)
        active = np.zeros(6, bool)
        for b, c in system.rows:
            if b == joint.body_b or (b == joint.body_a and joint.body_a != GROUND):
                active[c] = True
        joints[jname] = JointResult(
            jname,
            joint.type_name,
            joint.kind,
            joint.body_a,
            joint.body_b,
            geo.point_b,
            geo.point_a,
            force,
            moment,
            determined,
            scalars,
            scalar_kinds,
            scalar_det,
            active,
            transmits & active,
        )

    loads = [_factored(w, factors) for w in model.loads if factors.get(w.case, 0) != 0]
    balance = _balance(system, joints, loads)

    warnings = []
    excited = []
    if not sol.balanced:
        for k, g in enumerate(sol.excited_modes):
            if abs(g) > 1e-9 * max(np.abs(sol.excited_modes).max(), 1e-300):
                excited.append(describe_mechanism_mode(system, analysis.mechanism_modes[:, k]))
        if any(b.motion is not None for b in model.bodies.values()):
            warnings.append(
                "The loads and the prescribed motion disagree along a free motion of the "
                "model. Make a joint 'actuated: true' to find the effort needed, or correct "
                "the motion. The forces shown are a least-squares fit, not a solution."
            )
        else:
            warnings.append(
                "The loads drive a free motion of the model (it is a mechanism for these "
                "loads). Add or change supports. The forces shown are a least-squares fit, "
                "not a solution."
            )
    redundancy = []
    if not sol.fully_determined:
        for k in range(free.shape[1]):
            redundancy.append(describe_redundancy_mode(system, free[:, k]))
        missing = sorted(
            {
                f"{j.name}.{ROW_NAMES[c]}"
                for j in joints.values()
                for c in range(6)
                if not j.determined[c] and j.active[c]
            }
        )
        warnings.append(
            f"Statically indeterminate (degree {free.shape[1]}): "
            f"{', '.join(missing)} cannot be found from equilibrium alone. "
            "Give the joints involved a 'stiffness' to share the load by stiffness, "
            "or release redundant components."
        )
    # one-sided supports: only meaningful when the forces are a real solution
    for jname, j in joints.items() if sol.balanced else ():
        geo = model.joints[jname].geometry
        scale = max(np.linalg.norm(j.force), 1e-300)
        for label, value in j.scalars.items():
            if not j.scalar_determined[label]:
                continue
            if geo.sign_limit == "compression" and value < -1e-9 * scale:
                warnings.append(
                    f"{jname}: contact force is negative (pulling); the body would lift off "
                    "this support, which can only push."
                )
            if geo.sign_limit == "tension" and value < -1e-9 * scale:
                warnings.append(
                    f"{jname}: cable force is negative (compression); a cable would go slack."
                )

    applied = np.zeros(6)
    for w in loads:
        applied += np.concatenate([w.force, w.moment + np.cross(w.point, w.force)])
    reactions = np.zeros(6)
    for j in joints.values():
        if model.joints[j.name].body_a == GROUND:
            reactions += np.concatenate([j.force, j.moment + np.cross(j.point, j.force)])

    if not sol.balanced:
        status = "unbalanced"
    elif not sol.fully_determined:
        status = "indeterminate"
    else:
        status = "ok"
    return CaseResult(
        name,
        kind,
        dict(factors),
        status,
        joints,
        loads,
        balance,
        sol,
        warnings,
        excited,
        redundancy,
        applied,
        reactions,
    )


def _factored(w: AppliedWrench, factors: dict[str, float]) -> AppliedWrench:
    f = factors.get(w.case, 0.0)
    if f == 1.0:
        return w
    detail = dict(w.detail)
    for key in ("w1", "w2", "resultant"):
        if key in detail:
            detail[key] = detail[key] * f
    return AppliedWrench(w.name, w.body, w.case, w.force * f, w.moment * f, w.point, w.kind, detail)


def _balance(system: System, joints: dict[str, JointResult], loads) -> list[BodyBalance]:
    """Independent equilibrium check: sum every wrench on every body directly."""
    model = system.model
    L = system.length
    out = []
    for name, body in model.bodies.items():
        rows = {c for (b, c) in system.rows if b == name}
        attached = [j for j in joints.values() if name in (j.body_a, j.body_b)]
        ref = body.reference_point
        if not (body.mass is not None and body.mass.mass > 0) and attached:
            ref = np.mean([j.point for j in attached], axis=0)
        F, M = np.zeros(3), np.zeros(3)
        magnitude = 0.0
        for w in loads:
            if w.body == name:
                F += w.force
                M += w.moment + np.cross(w.point - ref, w.force)
                magnitude += np.linalg.norm(w.force) + np.linalg.norm(w.moment) / L
        for j in attached:
            sign, point = (1.0, j.point) if j.body_b == name else (-1.0, j.point_a)
            F += sign * j.force
            M += sign * (j.moment + np.cross(point - ref, j.force))
            magnitude += np.linalg.norm(j.force) + np.linalg.norm(j.moment) / L
        residual = np.concatenate([F, M])
        mask = np.array([c in rows for c in range(6)])
        residual[~mask] = 0.0
        scaled = np.concatenate([residual[:3], residual[3:] / L])
        relative = float(np.linalg.norm(scaled) / magnitude) if magnitude > 0 else 0.0
        out.append(
            BodyBalance(name, residual[:3], residual[3:], relative, relative <= EQUILIBRIUM_TOL)
        )
    return out


# --------------------------------------------------------------------------- checks


def evaluate_checks(results: Results) -> list[CheckResult]:
    out = []
    model = results.model
    for check in model.checks:
        if check.case is not None:
            case_names = [check.case]
        elif check.expect is not None and len(results.cases) > 1:
            raise InputError(
                f"check {check.name!r}: say which 'case' the expected value is for "
                f"(cases: {', '.join(results.cases)})"
            )
        else:
            case_names = list(results.cases)
        for case_name in case_names:
            out.append(_evaluate_check(results, check, results.cases[case_name]))
    return out


def _evaluate_check(results: Results, check: ResolvedCheck, case: CaseResult) -> CheckResult:
    units = results.model.output_units
    if check.unit and check.kind != "dimensionless":
        units = units.with_overrides(**{check.kind: check.unit})
    owner, _, comp = check.target.partition(".")
    if owner in results.model.bodies and comp == "mass":
        props = results.model.bodies[owner].mass
        value = 0.0 if props is None else props.mass
    else:
        value = case[owner].component(comp)

    def fmt(x, digits=4):
        return units.format(x, check.kind, digits=digits)

    if value is None:
        return CheckResult(
            check.name,
            check.target,
            case.name,
            check.kind,
            None,
            False,
            "indeterminate: cannot be checked",
        )
    problems, parts = [], []
    digits = 4
    if check.expect is not None:
        ok = abs(value - check.expect) <= check.tolerance + 1e-12 * max(abs(check.expect), 1.0)
        if not ok:  # show enough digits to see the difference
            digits = 8
        parts.append(f"expected {fmt(check.expect, digits)} ± {fmt(check.tolerance, 3)}")
        if not ok:
            problems.append(f"off by {fmt(value - check.expect, 3)}")
    if check.max is not None:
        parts.append(f"max {fmt(check.max)}")
        if value > check.max:
            problems.append("above max")
    if check.min is not None:
        parts.append(f"min {fmt(check.min)}")
        if value < check.min:
            problems.append("below min")
    detail = ", ".join(parts)
    if problems:
        detail += ": " + ", ".join(problems)
    return CheckResult(
        check.name,
        check.target,
        case.name,
        check.kind,
        value,
        not problems,
        detail,
        fmt(value, digits),
    )
