"""Built-in validation suite, run by ``engmech validate``.

The suite runs on the installed copy of engmech, so an organisation can
confirm that its installation reproduces the documented results (an
installation/operational qualification). It has two parts:

* **Benchmarks**: every bundled example and benchmark model. Each carries
  answers derived by hand in its description; the model must solve, pass
  the independent equilibrium check, and reproduce every expected value.
* **Property checks**: seeded random cases compared against independent
  computations (closed-form results, or linear systems built separately
  from the solver). They exercise the solver far beyond the benchmarks.

Every case records the largest error it saw, so the report shows margins,
not just pass/fail.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

from engmech import __version__
from engmech import mass as mp
from engmech.joints import Ball, Fixed, Link, Pin, Roller
from engmech.loads import Force, Moment, Motion
from engmech.model import Model
from engmech.shapes import Cylinder, Rod
from engmech.spatial import rotation_about_axis

SEED = 20260929
TOL = 1e-9  # relative agreement required for property checks


@dataclass
class Outcome:
    suite: str  # benchmark | property
    name: str
    description: str
    passed: bool
    checks: int
    max_error: float | None  # largest relative error seen (property checks)
    detail: str


@dataclass
class ValidationRun:
    outcomes: list[Outcome]
    environment: dict
    started: str
    seconds: float
    seed: int = SEED
    failures: list[Outcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.outcomes)

    def summary(self) -> dict:
        by_suite: dict[str, dict] = {}
        for o in self.outcomes:
            s = by_suite.setdefault(o.suite, {"cases": 0, "passed": 0, "checks": 0})
            s["cases"] += 1
            s["passed"] += o.passed
            s["checks"] += o.checks
        return by_suite

    def to_dict(self) -> dict:
        return {
            "engmech": __version__,
            "passed": self.passed,
            "started": self.started,
            "seconds": round(self.seconds, 2),
            "seed": self.seed,
            "environment": self.environment,
            "summary": self.summary(),
            "cases": [o.__dict__ for o in self.outcomes],
        }


# --------------------------------------------------------------------------- benchmarks


def benchmark_files() -> list:
    root = resources.files("engmech")
    files = []
    for folder in ("examples", "benchmarks"):
        files += sorted(
            (p for p in (root / folder).iterdir() if p.name.endswith(".yaml")),
            key=lambda p: p.name,
        )
    return files


def run_benchmark(path) -> Outcome:
    from engmech.io.loader import load_model

    name = f"{path.parent.name}/{path.name[:-5]}"
    try:
        model = load_model(str(path))
        results = model.solve()
    except Exception as exc:  # a crash is a failure, and must be reported, not raised
        return Outcome("benchmark", name, "", False, 0, None, f"error: {exc}")
    title = results.model.name
    problems = []
    if results.status != "ok":
        problems.append(f"status {results.status}")
    worst = max((c.max_residual for c in results.cases.values()), default=0.0)
    if not all(c.verified for c in results.cases.values()):
        problems.append(f"equilibrium residual {worst:.1e}")
    failed = [c for c in results.checks if not c.passed]
    for c in failed:
        problems.append(f"{c.name}: {c.display} ({c.detail})")
    if not results.checks:
        problems.append("no hand-derived checks")
    n = len(results.checks)
    detail = (
        "; ".join(problems)
        if problems
        else (
            f"{n} hand-derived value{'s' * (n != 1)} reproduced; equilibrium residual {worst:.0e}"
        )
    )
    return Outcome("benchmark", name, title, not problems, n, None, detail)


# --------------------------------------------------------------------------- properties


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(float(np.abs(b).max(initial=0.0)), 1.0)
    return float(np.abs(a - b).max(initial=0.0) / scale)


def _prop_resultant(rng: np.random.Generator, n: int) -> tuple[int, float]:
    worst = 0.0
    for _ in range(n):
        support = rng.uniform(-5, 5, 3)
        m = Model()
        m.support("A", Fixed(at=support))
        F, M = np.zeros(3), np.zeros(3)
        for _ in range(rng.integers(1, 6)):
            f, p = rng.uniform(-100, 100, 3), rng.uniform(-5, 5, 3)
            m.load(Force(f, at=p))
            F += f
            M += np.cross(p - support, f)
        c = rng.uniform(-50, 50, 3)
        m.load(Moment(c))
        M += c
        A = m.solve().primary["A"]
        worst = max(worst, _rel(A.wrench, -np.concatenate([F, M])))
    return n, worst


def _prop_six_links(rng: np.random.Generator, n: int) -> tuple[int, float]:
    worst, done = 0.0, 0
    while done < n:
        anchors, attach = rng.uniform(-3, 3, (6, 3)), rng.uniform(-3, 3, (6, 3))
        d = anchors - attach
        if np.linalg.norm(d, axis=1).min() < 0.5:
            continue
        u = d / np.linalg.norm(d, axis=1)[:, None]
        A = np.array([np.concatenate([u[i], np.cross(attach[i], u[i])]) for i in range(6)]).T
        if np.linalg.cond(A) > 1e3:
            continue
        load, at = rng.uniform(-100, 100, 3), rng.uniform(-3, 3, 3)
        expected = np.linalg.solve(A, -np.concatenate([load, np.cross(at, load)]))
        m = Model()
        for i in range(6):
            m.support(f"L{i}", Link(at=attach[i], anchor=anchors[i]))
        m.load(Force(load, at=at))
        r = m.solve().primary
        got = [r[f"L{i}"].scalars["T"] for i in range(6)]
        worst = max(worst, _rel(got, expected))
        done += 1
    return n, worst


def _frame(scale: float, length: str, force: str, f: float, xs) -> Model:
    m = Model(planar=True, units={"length": length, "force": force})
    m.body("left")
    m.body("right")
    m.support("A", Pin(at=[0, 0]), body="left")
    m.support("B", Pin(at=[xs[0] * scale, 0]), body="right")
    m.joint("C", Pin(at=[xs[1] * scale, xs[2] * scale]), bodies=("left", "right"))
    m.load(Force([xs[3] * f, -xs[4] * f], at=[xs[5] * scale, xs[6] * scale]), body="left")
    m.load(Moment(xs[7] * f * scale), body="right")
    return m


def _prop_units(rng: np.random.Generator, n: int) -> tuple[int, float]:
    worst = 0.0
    for _ in range(n):
        xs = rng.uniform(0.5, 6, 8)
        a = _frame(1, "m", "kN", 1, xs).solve().primary
        b = _frame(1000, "mm", "N", 1000, xs).solve().primary
        for j in ("A", "B", "C"):
            worst = max(worst, _rel(b[j].wrench, a[j].wrench))
    return n, worst


def _two_bodies(point: Callable, direction: Callable) -> Model:
    m = Model()
    m.body("a")
    m.body("b")
    m.support("S", Ball(at=point([0, 0, 0])), body="a")
    m.support("L1", Link(at=point([2, 0, 0]), anchor=point([2, 0, 3])), body="a")
    m.support("L2", Link(at=point([2, 0, 0]), anchor=point([2, 3, 0])), body="a")
    m.support("L3", Link(at=point([0, 1, 0]), anchor=point([0, 1, 3])), body="a")
    m.joint("H", Pin(at=point([2, 1, 0]), axis=direction([0, 0, 1])), bodies=("a", "b"))
    m.support("R", Roller(at=point([4, 1, 0]), normal=direction([0, 1, 0])), body="b")
    m.load(Force(direction([1, -2, -5]), at=point([3, 1, 0.5])), body="b")
    m.load(Force(direction([0, 0, -3]), at=point([1, 0.5, 0])), body="a")
    m.load(Moment(direction([0.5, 0, 0])), body="a")
    return m


def _prop_rigid_motion(rng: np.random.Generator, n: int) -> tuple[int, float]:
    worst = 0.0
    base = _two_bodies(np.asarray, np.asarray).solve().primary
    for _ in range(n):
        R = rotation_about_axis(rng.normal(size=3), rng.uniform(0, 2 * np.pi))
        shift = rng.uniform(-10, 10, 3)
        moved = (
            _two_bodies(
                lambda p, R=R, s=shift: R @ np.asarray(p, float) + s,
                lambda d, R=R: R @ np.asarray(d, float),
            )
            .solve()
            .primary
        )
        for j in ("S", "L1", "L2", "L3", "H", "R"):
            worst = max(
                worst,
                _rel(moved[j].wrench, np.concatenate([R @ base[j].force, R @ base[j].moment])),
            )
    return n, worst


def _prop_bolt_group(rng: np.random.Generator, n: int) -> tuple[int, float]:
    worst, done = 0.0, 0
    while done < n:
        pts = rng.uniform(-0.1, 0.1, (int(rng.integers(3, 9)), 2))
        if min(np.linalg.norm(p - q) for i, p in enumerate(pts) for q in pts[:i]) < 0.02:
            continue
        c = pts.mean(axis=0)
        r = pts - c
        J = float((r**2).sum())
        P, at = rng.uniform(-10e3, 10e3, 2), rng.uniform(-0.5, 0.5, 2)
        e = at - c
        M = e[0] * P[1] - e[1] * P[0]
        expected = -P / len(pts) - (M / J) * np.column_stack([-r[:, 1], r[:, 0]])
        m = Model(planar=True)
        for i, p in enumerate(pts):
            m.support(f"b{i}", Pin(at=p, stiffness={"translational": 2e8}))
        m.load(Force(P, at=at))
        res = m.solve().primary
        got = np.array([res[f"b{i}"].force[:2] for i in range(len(pts))])
        worst = max(worst, _rel(got, expected))
        done += 1
    return n, worst


def _prop_parallel_axis(rng: np.random.Generator, n: int) -> tuple[int, float]:
    worst = 0.0
    for _ in range(n):
        parts = [
            mp.box(
                rng.uniform(0.05, 1, 3),
                rng.uniform(-2, 2, 3),
                rotation_about_axis(rng.normal(size=3), rng.uniform(0, 6)),
                mass=rng.uniform(0.1, 20),
            )
            for _ in range(int(rng.integers(1, 6)))
        ]
        total = mp.combine(parts)
        point = rng.uniform(-3, 3, 3)
        direct = sum(p.inertia_about(point) for p in parts)
        worst = max(worst, _rel(total.inertia_about(point), direct))
        # the composite is physical: positive definite, triangle inequality
        if total.check_physical():
            return n, float("inf")
    return n, worst


def _prop_dynamics(rng: np.random.Generator, n: int) -> tuple[int, float]:
    """Rod on an actuated pin: drive torque and pivot force against Newton-Euler
    in closed form, for random length, mass, angle, speed and acceleration."""
    g = 9.80665
    worst = 0.0
    for _ in range(n):
        L, m, th = rng.uniform(0.2, 3), rng.uniform(0.5, 20), rng.uniform(-np.pi, np.pi)
        w, al = rng.uniform(-10, 10), rng.uniform(-50, 50)
        end = [L * np.cos(th), L * np.sin(th)]
        model = Model(planar=True, gravity="-y")
        model.body(
            "rod",
            shapes=[Rod(mass=m, start=[0, 0], end=end)],
            motion=Motion(angular_velocity=w, angular_acceleration=al, pivot=[0, 0]),
        )
        model.support("O", Pin(at=[0, 0], actuated=True))
        O = model.solve().primary["O"]
        d = L / 2
        tau = m * L**2 / 3 * al + m * g * d * np.cos(th)
        a_g = al * d * np.array([-np.sin(th), np.cos(th)]) - w**2 * d * np.array(
            [np.cos(th), np.sin(th)]
        )
        R = m * a_g + np.array([0, m * g])
        worst = max(worst, _rel([O.scalars["drive"], *O.force[:2]], [tau, *R]))
    return n, worst


def _prop_gyroscope(rng: np.random.Generator, n: int) -> tuple[int, float]:
    """Spinning disc precessing about z: joint moment I_s * spin * Omega about y."""
    worst = 0.0
    for _ in range(n):
        m, r, l = rng.uniform(0.5, 5), rng.uniform(0.02, 0.3), rng.uniform(0.05, 0.5)
        spin, Om = rng.uniform(50, 500), rng.uniform(-5, 5)
        model = Model()
        model.body(
            "rotor",
            shapes=[Cylinder(mass=m, radius=r, length=0, center=[l, 0, 0], axis="+x")],
            motion=Motion(
                angular_velocity=[spin, 0, Om],
                angular_acceleration=[0, Om * spin, 0],
                pivot=[0, 0, 0],
            ),
        )
        model.support("O", Fixed(at=[0, 0, 0]))
        O = model.solve().primary["O"]
        expected = [-m * Om**2 * l, 0, 0, 0, m * r * r / 2 * spin * Om, 0]
        worst = max(worst, _rel(O.wrench, expected))
    return n, worst


PROPERTIES: list[tuple[str, str, Callable]] = [
    (
        "resultant",
        "A fixed support cancels the resultant of random forces and couples",
        _prop_resultant,
    ),
    (
        "six-links",
        "A body on six random links matches an independently built linear system",
        _prop_six_links,
    ),
    (
        "unit-invariance",
        "A two-body frame gives identical SI results in m/kN and mm/N",
        _prop_units,
    ),
    (
        "rigid-motion",
        "Rotating and translating a whole 3D model rotates every reaction with it",
        _prop_rigid_motion,
    ),
    (
        "bolt-group",
        "Equal-stiffness bolt groups reproduce the elastic method (P/n + M r / sum r^2)",
        _prop_bolt_group,
    ),
    (
        "parallel-axis",
        "Composite inertia about any point equals the sum of the parts' inertias",
        _prop_parallel_axis,
    ),
    (
        "rod-dynamics",
        "Driven rod: motor torque and pivot force match Newton-Euler in closed form",
        _prop_dynamics,
    ),
    ("gyroscope", "Precessing disc: joint moment equals I_s * spin * precession", _prop_gyroscope),
]


def run_properties(seed: int = SEED, n: int = 40) -> list[Outcome]:
    out = []
    for i, (name, description, fn) in enumerate(PROPERTIES):
        rng = np.random.default_rng([seed, i])
        try:
            count, worst = fn(rng, n)
        except Exception as exc:
            out.append(Outcome("property", name, description, False, 0, None, f"error: {exc}"))
            continue
        ok = worst <= TOL
        out.append(
            Outcome(
                "property",
                name,
                description,
                ok,
                count,
                worst,
                f"{count} random cases, largest relative error {worst:.1e} (limit {TOL:.0e})",
            )
        )
    return out


def run(seed: int = SEED, n: int = 40) -> ValidationRun:
    from engmech.provenance import environment

    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    outcomes = [run_benchmark(p) for p in benchmark_files()]
    outcomes += run_properties(seed, n)
    return ValidationRun(outcomes, environment(), started, time.perf_counter() - t0, seed)
