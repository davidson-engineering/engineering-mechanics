"""Assemble and solve the equilibrium equations of a built model.

Unknowns are the magnitudes of every wrench component transmitted by every
joint. Equations are force and moment equilibrium of every body (six per
rigid body in space, three in a planar analysis, fewer for particles),
written about one common reference point. Inertial (d'Alembert) loads turn
the same equations into Newton-Euler dynamics.

The matrix is scaled by a characteristic length before any rank decision, so
the answer to "is this a mechanism?" or "is this statically indeterminate?"
never depends on whether you work in millimetres or metres.

* Mechanisms (rank < equations): the left null space gives the free motions.
  If the loads do work on a free motion there is no equilibrium; otherwise
  the solution is still valid and the mechanism is reported.
* Redundancy (rank < unknowns): the null space gives self-stress states.
  Components that take part in one cannot be found from statics alone and
  are reported as indeterminate, unless joint stiffnesses are given, in which
  case the least-work (minimum complementary energy) solution is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engmech.errors import InputError
from engmech.joints import Component
from engmech.model import GROUND, BuiltModel

RANK_TOL = 1e-10  # relative singular-value cutoff
NULL_TOL = 1e-8  # participation threshold in orthonormal null-space vectors
BALANCE_TOL = 1e-9  # relative load imbalance tolerated before declaring "unbalanced"
MIN_LENGTH = 1e-9  # m; a model smaller than this is a point (geometry below is round-off)

SPATIAL_ROWS = (0, 1, 2, 3, 4, 5)
PLANAR_ROWS = (0, 1, 5)
ROW_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass
class Unknown:
    joint: str
    index: int  # into joint.geometry.components
    component: Component


@dataclass
class System:
    model: BuiltModel
    ref: np.ndarray
    length: float  # characteristic length used for scaling
    rows: list[tuple[str, int]]  # (body, 0..5)
    unknowns: list[Unknown]
    dropped: list[tuple[str, int, str]]  # (joint, component index, reason)
    A: np.ndarray  # SI
    row_scale: np.ndarray
    col_scale: np.ndarray

    @property
    def scaled(self) -> np.ndarray:
        return self.row_scale[:, None] * self.A * self.col_scale[None, :]

    def body_rows(self, body: str) -> list[int]:
        return [i for i, (b, _) in enumerate(self.rows) if b == body]


@dataclass
class Analysis:
    """Structure of the equations, independent of the loads."""

    equations: int
    unknowns: int
    rank: int
    singular_values: np.ndarray
    mechanism_modes: np.ndarray  # scaled left null space, (m, m - r)
    redundancy_modes: np.ndarray  # scaled null space, (n, n - r)
    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray

    @property
    def degrees_of_freedom(self) -> int:
        return self.equations - self.rank

    @property
    def degree_of_indeterminacy(self) -> int:
        return self.unknowns - self.rank

    @property
    def condition_number(self) -> float:
        if self.rank == 0:
            return float("inf")
        return float(self.s[0] / self.s[self.rank - 1])


@dataclass
class CaseSolution:
    values: np.ndarray  # SI, one per unknown
    determined: np.ndarray  # bool per unknown
    free_modes: np.ndarray  # scaled null-space basis still undetermined, (n, k)
    imbalance: float  # relative
    excited_modes: np.ndarray  # generalised force per mechanism mode (scaled)
    used_stiffness: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def balanced(self) -> bool:
        return self.imbalance <= BALANCE_TOL

    @property
    def fully_determined(self) -> bool:
        return bool(self.determined.all())


# --------------------------------------------------------------------------- assembly


def _classify(component: Component, planar: bool, particle: bool) -> str | None:
    """Return a reason to drop the component from the equations, or None."""
    if particle and component.kind == "moment":
        return "particle"
    if not planar:
        return None
    d = component.direction
    in_plane = abs(d[2]) < 1e-9
    normal = np.linalg.norm(d[:2]) < 1e-9
    if component.kind == "force":
        if in_plane:
            return None
        if normal:
            return "out of plane"
    else:
        if normal:
            return None
        if in_plane:
            return "out of plane"
    raise InputError("a joint direction is neither in the plane nor perpendicular to it")


def assemble(model: BuiltModel) -> System:
    pts = model.all_points()
    ref = pts.mean(axis=0)
    length = float(np.max(np.linalg.norm(pts - ref, axis=1))) if len(pts) else 0.0
    if not length > MIN_LENGTH:
        length = 1.0  # all points coincide: any length scale will do

    rows: list[tuple[str, int]] = []
    for name, body in model.bodies.items():
        if body.particle:
            comps = (0, 1) if model.planar else (0, 1, 2)
        else:
            comps = PLANAR_ROWS if model.planar else SPATIAL_ROWS
        rows += [(name, c) for c in comps]
    row_index = {key: i for i, key in enumerate(rows)}

    unknowns: list[Unknown] = []
    dropped: list[tuple[str, int, str]] = []
    columns: list[np.ndarray] = []
    for jname, joint in model.joints.items():
        geo = joint.geometry
        particle = any(
            b != GROUND and model.bodies[b].particle for b in (joint.body_a, joint.body_b)
        )
        for k, comp in enumerate(geo.components):
            try:
                reason = _classify(comp, model.planar, particle)
            except InputError as exc:
                raise exc.at(f"{jname}") from None
            if reason:
                dropped.append((jname, k, reason))
                continue
            col = np.zeros(len(rows))
            for body, point, sign in (
                (joint.body_b, geo.point_b, 1.0),
                (joint.body_a, geo.point_a, -1.0),
            ):
                if body == GROUND:
                    continue
                w = _unit_wrench(comp, point, ref) * sign
                for c in range(6):
                    i = row_index.get((body, c))
                    if i is not None:
                        col[i] += w[c]
            unknowns.append(Unknown(jname, k, comp))
            columns.append(col)

    A = np.column_stack(columns) if columns else np.zeros((len(rows), 0))
    row_scale = np.array([1.0 if c < 3 else 1.0 / length for _, c in rows])
    col_scale = np.array([1.0 if u.component.kind == "force" else length for u in unknowns])
    return System(model, ref, length, rows, unknowns, dropped, A, row_scale, col_scale)


def _unit_wrench(comp: Component, point: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if comp.kind == "force":
        return np.concatenate([comp.direction, np.cross(point - ref, comp.direction)])
    return np.concatenate([np.zeros(3), comp.direction])


def load_vector(system: System, factors: dict[str, float]) -> np.ndarray:
    """Right-hand side: minus the applied (and inertial) loads, per body row."""
    b = np.zeros(len(system.rows))
    index = {key: i for i, key in enumerate(system.rows)}
    for w in system.model.loads:
        f = factors.get(w.case, 0.0)
        if f == 0:
            continue
        full = np.concatenate([w.force, w.moment + np.cross(w.point - system.ref, w.force)])
        for c in range(6):
            i = index.get((w.body, c))
            if i is not None:
                b[i] -= f * full[c]
    return b


def load_scale(system: System, factors: dict[str, float]) -> float:
    """Sum of scaled load magnitudes: the yardstick for 'small' residuals."""
    total = 0.0
    for w in system.model.loads:
        f = abs(factors.get(w.case, 0.0))
        total += f * (np.linalg.norm(w.force) + np.linalg.norm(w.moment) / system.length)
    return total


# --------------------------------------------------------------------------- analysis


def analyze(system: System) -> Analysis:
    A = system.scaled
    m, n = A.shape
    if n == 0 or m == 0:
        return Analysis(
            m, n, 0, np.zeros(0), np.eye(m), np.eye(n), np.eye(m), np.zeros(0), np.eye(n)
        )
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > RANK_TOL * s[0])) if s.size and s[0] > 0 else 0
    return Analysis(m, n, rank, s, U[:, rank:], Vt[rank:].T, U, s, Vt)


def solve_case(system: System, analysis: Analysis, factors: dict[str, float]) -> CaseSolution:
    b = system.row_scale * load_vector(system, factors)
    scale = max(load_scale(system, factors), 1e-300)
    r = analysis.rank
    n = analysis.unknowns

    excited = analysis.mechanism_modes.T @ b if analysis.mechanism_modes.size else np.zeros(0)
    imbalance = float(np.linalg.norm(excited) / scale) if np.any(b) else 0.0

    if r > 0:
        Ur, sr, Vr = analysis.U[:, :r], analysis.s[:r], analysis.Vt[:r].T
        mu = Vr @ ((Ur.T @ b) / sr)
    else:
        mu = np.zeros(n)

    free = analysis.redundancy_modes
    used_stiffness = False
    if free.shape[1] > 0:
        compliance = np.array([u.component.compliance for u in system.unknowns])
        if np.any(compliance > 0):
            used_stiffness = True
            # minimise complementary energy sum(c * lambda^2) over the self-stress space
            D = system.col_scale
            Cs = D * compliance * D  # compliance in scaled coordinates
            H = free.T @ (Cs[:, None] * free)
            g = free.T @ (Cs * mu)
            hs = np.linalg.eigvalsh(H) if H.size else np.zeros(0)
            hmax = hs.max() if hs.size else 0.0
            if hmax > 0:
                z = np.linalg.lstsq(H, -g, rcond=RANK_TOL)[0]
                mu = mu + free @ z
                w, V = np.linalg.eigh(H)
                free = free @ V[:, w <= RANK_TOL * hmax]
    participation = np.linalg.norm(free, axis=1) if free.size else np.zeros(n)
    determined = participation <= NULL_TOL

    # tidy numerical noise so exact zeros print as zeros
    mu[np.abs(mu) < 1e-14 * scale] = 0.0
    values = system.col_scale * mu
    return CaseSolution(values, determined, free, imbalance, excited, used_stiffness)


def solve(model: BuiltModel):
    """Solve every load case and combination of a built model."""
    from engmech.results import build_results

    system = assemble(model)
    analysis = analyze(system)
    solutions = {}
    for case in model.cases:
        solutions[case] = ("case", solve_case(system, analysis, {case: 1.0}))
    for name, factors in model.combinations.items():
        solutions[name] = ("combination", solve_case(system, analysis, factors))
    return build_results(system, analysis, solutions)
