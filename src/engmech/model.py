"""The model: bodies, supports, joints, loads, load cases.

:class:`Model` records raw user input. :meth:`Model.build` resolves all of it
(units, parameters, named points) into a :class:`BuiltModel` of SI arrays,
which is what the solver consumes. Keeping raw input lets parameters be
overridden and the model rebuilt, e.g. for parameter sweeps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engmech import mass as mp
from engmech.errors import InputError
from engmech.inputs import Resolver
from engmech.joints import JointGeometry, JointType, UnknownForce, UnknownMoment
from engmech.loads import (
    DEFAULT_CASE,
    AppliedWrench,
    DistributedLoad,
    Force,
    Moment,
    Motion,
    ResolvedMotion,
    UnknownLoad,
)
from engmech.shapes import CustomMass, Shape
from engmech.spatial import parse_axis_name, unit
from engmech.units import STANDARD_GRAVITY, Context, UnitSystem, evaluate_parameters

GROUND = "ground"


# --------------------------------------------------------------------------- raw


@dataclass
class _BodySpec:
    name: str
    shapes: list[Shape]
    particle: bool
    motion: Motion | None
    outline: list[Any] | None


@dataclass
class _JointSpec:
    name: str
    joint: JointType
    body_a: str
    body_b: str
    kind: str  # support | joint | unknown
    where: str | None = None


@dataclass
class _LoadSpec:
    load: Force | Moment | DistributedLoad
    body: str | None
    where: str | None = None


@dataclass
class Check:
    """A design or verification check on a result.

    ``target`` is ``"<joint>.<component>"`` (components: Fx Fy Fz Mx My Mz,
    F and M for magnitudes, or the joint's scalar such as N, T, drive) or
    ``"<body>.mass"``. Give ``expect`` (with ``tolerance``) and/or ``max``/``min``.
    """

    target: str
    expect: Any = None
    tolerance: Any = "0.1%"
    max: Any = None
    min: Any = None
    case: str | None = None
    name: str | None = None


# --------------------------------------------------------------------------- resolved


@dataclass
class ResolvedBody:
    name: str
    mass: mp.MassProperties | None
    shapes: list[mp.MassProperties]
    particle: bool
    motion: ResolvedMotion | None
    outline: list[np.ndarray]

    @property
    def reference_point(self) -> np.ndarray:
        return self.mass.cog if self.mass is not None and self.mass.mass > 0 else np.zeros(3)


@dataclass
class ResolvedJoint:
    name: str
    type_name: str
    body_a: str
    body_b: str
    geometry: JointGeometry
    kind: str  # support | joint | unknown

    @property
    def point(self) -> np.ndarray:
        return self.geometry.point_b


@dataclass
class ResolvedCheck:
    name: str
    target: str
    kind: str  # quantity kind for display
    case: str | None
    expect: float | None
    tolerance: float  # absolute
    max: float | None
    min: float | None
    unit: str | None = None  # the unit the user wrote the criterion in, for display


@dataclass
class BuiltModel:
    name: str
    description: str
    planar: bool
    units: UnitSystem
    output_units: UnitSystem
    parameters: dict[str, Any]
    points: dict[str, np.ndarray]
    bodies: dict[str, ResolvedBody]
    joints: dict[str, ResolvedJoint]
    loads: list[AppliedWrench]
    gravity: np.ndarray | None
    cases: list[str]
    combinations: dict[str, dict[str, float]]
    checks: list[ResolvedCheck]
    source_path: str | None = None  # the model file, when loaded from one
    source_text: str | None = None
    source_sha256: str | None = None
    source_size: int | None = None
    overrides: dict[str, str] = field(default_factory=dict)  # parameter overrides applied

    def all_points(self) -> np.ndarray:
        pts = [j.geometry.point_a for j in self.joints.values()]
        pts += [j.geometry.point_b for j in self.joints.values()]
        pts += [w.point for w in self.loads]
        for w in self.loads:
            if w.kind == "distributed":
                pts.append(w.detail["end"])
        for b in self.bodies.values():
            pts += list(b.outline)
            if b.mass is not None:
                pts.append(b.mass.cog)
        pts += list(self.points.values())
        return np.array(pts) if pts else np.zeros((1, 3))


# --------------------------------------------------------------------------- model


class Model:
    """A rigid-body mechanics model.

    >>> m = Model("Beam", planar=True, units={"length": "m", "force": "kN"})
    >>> m.support("A", Pin(at=[0, 0]))
    >>> m.support("B", Roller(at=[4, 0], normal="+y"))
    >>> m.load(Force([0, -10], at=[1, 0]))
    >>> result = m.solve()
    """

    def __init__(
        self,
        name: str = "Untitled model",
        *,
        planar: bool = False,
        units: UnitSystem | str | Mapping[str, str] | None = None,
        output_units: UnitSystem | str | Mapping[str, str] | None = None,
        parameters: Mapping[str, Any] | None = None,
        description: str = "",
        gravity: Any = None,
        gravity_case: str | None = None,
    ):
        self.name = name
        self.description = description
        self.planar = planar
        self.units = UnitSystem.from_spec(units)
        self.output_units = (
            self.units if output_units is None else UnitSystem.from_spec(output_units)
        )
        self.parameters: dict[str, Any] = dict(parameters or {})
        self._points: dict[str, Any] = {}
        self._bodies: dict[str, _BodySpec] = {}
        self._joints: dict[str, _JointSpec] = {}
        self._loads: list[_LoadSpec] = []
        self._unknowns: list[tuple[UnknownLoad, str | None]] = []
        self.gravity = gravity
        self.gravity_case = gravity_case
        self.combinations: dict[str, dict[str, Any]] = {}
        self.checks: list[Check] = []
        self.source = None  # set by the file loader: maps input paths to lines

    # ---------------------------------------------------------------- building

    def point(self, name: str, value: Any) -> Model:
        if not name.isidentifier():
            raise InputError(f"point name {name!r} must be a simple identifier like A or P1")
        if name in self._points:
            raise InputError(f"point {name!r} is defined twice")
        self._points[name] = value
        return self

    def body(
        self,
        name: str,
        *,
        mass: Any = None,
        cog: Any = None,
        inertia: Any = None,
        shapes: Sequence[Shape] = (),
        particle: bool = False,
        motion: Motion | None = None,
        outline: Sequence[Any] | None = None,
    ) -> str:
        """Add a rigid body (or a particle). Mass properties come from
        ``mass``/``cog``/``inertia`` and/or a list of ``shapes``."""
        if name == GROUND:
            raise InputError(f"{GROUND!r} is reserved for the fixed ground")
        if name in self._bodies:
            raise InputError(f"body {name!r} is defined twice")
        all_shapes = list(shapes)
        if mass is not None:
            if cog is None:
                raise InputError(f"body {name!r}: 'mass' needs a 'cog'")
            all_shapes.insert(0, CustomMass(mass=mass, cog=cog, inertia=inertia, name=name))
        elif cog is not None or inertia is not None:
            raise InputError(f"body {name!r}: 'cog'/'inertia' need a 'mass'")
        self._bodies[name] = _BodySpec(
            name, all_shapes, particle, motion, None if outline is None else list(outline)
        )
        return name

    def support(self, name: str, joint: JointType, body: str | None = None) -> Model:
        """Connect a body to the ground."""
        self._add_joint(_JointSpec(name, joint, GROUND, body, "support"))
        return self

    def joint(self, name: str, joint: JointType, bodies: Sequence[str]) -> Model:
        """Connect two bodies (``bodies[0]`` is 'a', ``bodies[1]`` is 'b');
        results report the force on ``b`` from ``a``."""
        if len(bodies) != 2 or bodies[0] == bodies[1]:
            raise InputError(f"joint {name!r}: 'bodies' must name two different bodies")
        a, b = bodies
        self._add_joint(_JointSpec(name, joint, a, b, "joint"))
        return self

    def _add_joint(self, spec: _JointSpec) -> None:
        if spec.name in self._joints:
            raise InputError(f"{spec.name!r} is used by more than one support/joint/unknown")
        self._joints[spec.name] = spec

    def load(
        self,
        load: Force | Moment | DistributedLoad | UnknownLoad,
        body: str | None = None,
        where: str | None = None,
    ) -> Model:
        """Apply a load to a body (optional when the model has one body)."""
        if isinstance(load, UnknownLoad):
            if load.name in self._joints:
                raise InputError(
                    f"{load.name!r} is used by more than one support/joint/unknown", where
                )
            joint = (
                UnknownForce(at=load.at, direction=load.direction, label=load.name)
                if load.direction is not None
                else UnknownMoment(at=load.at, axis=load.axis, label=load.name)
            )
            self._joints[load.name] = _JointSpec(load.name, joint, GROUND, body, "unknown", where)
            return self
        if not isinstance(load, Force | Moment | DistributedLoad):
            raise TypeError(f"not a load: {load!r}")
        self._loads.append(_LoadSpec(load, body, where))
        return self

    def combination(self, name: str, factors: Mapping[str, Any]) -> Model:
        self.combinations[name] = dict(factors)
        return self

    def check(self, check: Check) -> Model:
        self.checks.append(check)
        return self

    # ---------------------------------------------------------------- resolve

    def build(self, overrides: Mapping[str, Any] | None = None) -> BuiltModel:
        """Resolve all input to SI. ``overrides`` replaces parameter values.

        For models loaded from a file, errors carry the file position."""
        try:
            return self._build(overrides)
        except InputError as exc:
            if self.source is None or getattr(exc, "problems", None):
                raise
            from engmech.io.loader import ModelFileError

            raise ModelFileError([f"{self.source.describe(exc.where)}: {exc}"]) from None

    def _build(self, overrides: Mapping[str, Any] | None) -> BuiltModel:
        try:
            params = evaluate_parameters(self.parameters, overrides, self.units)
        except InputError as exc:
            raise exc.at("parameters") from None
        ctx = Context(self.units, params)
        r = Resolver(ctx, {}, self.planar)
        points = {}
        for name, value in self._points.items():
            try:
                points[name] = r.position(value, f"points.{name}")
            except InputError as exc:
                raise exc.at(f"points.{name}") from None
            r.points = points

        body_specs = dict(self._bodies)
        if not body_specs:
            body_specs["body"] = _BodySpec("body", [], False, None, None)

        def which_body(given: str | None, what: str) -> str:
            if given is None:
                if len(body_specs) == 1:
                    return next(iter(body_specs))
                raise InputError(
                    f"say which body it acts on (bodies: {', '.join(body_specs)})", what
                )
            if given == GROUND:
                raise InputError("loads and supports act on bodies, not the ground", what)
            if given not in body_specs:
                if not self._bodies:
                    raise InputError(
                        f"unknown body {given!r}: the model has no 'bodies' section, so its "
                        f"one body is called 'body' (or add bodies: {{{given}: {{}}}})",
                        what,
                    )
                raise InputError(f"unknown body {given!r} (bodies: {', '.join(body_specs)})", what)
            return given

        bodies: dict[str, ResolvedBody] = {}
        for name, spec in body_specs.items():
            what = f"bodies.{name}"
            shapes = []
            for i, s in enumerate(spec.shapes):
                try:
                    shapes.append(s.resolve(r, f"{what}.shapes[{i}]"))
                except InputError as exc:
                    raise exc.at(f"{what}.shapes[{i}]") from None
            try:
                props = mp.combine(shapes, label=name) if shapes else None
            except InputError as exc:
                raise exc.at(what) from None
            if props is not None:
                problems = props.check_physical()
                if problems:
                    raise InputError("; ".join(problems), what)
            outline = [r.position(p, f"{what}.outline") for p in (spec.outline or [])]
            motion = None
            if spec.motion is not None:
                if props is None or props.mass <= 0:
                    raise InputError("a body with 'motion' needs mass properties", what)
                try:
                    motion = spec.motion.resolve(r, props.cog, f"{what}.motion")
                except InputError as exc:
                    raise exc.at(f"{what}.motion") from None
            bodies[name] = ResolvedBody(name, props, shapes, spec.particle, motion, outline)

        joints: dict[str, ResolvedJoint] = {}
        for name, spec in self._joints.items():
            section = {"support": "supports", "joint": "joints", "unknown": "loads"}[spec.kind]
            what = spec.where or f"{section}.{name}"
            try:
                body_b = which_body(spec.body_b, what)
                body_a = spec.body_a
                if spec.kind == "joint":
                    body_a = which_body(spec.body_a, what)
                geometry = spec.joint.resolve(r, what)
            except InputError as exc:
                raise exc.at(what) from None
            joints[name] = ResolvedJoint(
                name, spec.joint.type_name, body_a, body_b, geometry, spec.kind
            )

        loads: list[AppliedWrench] = []
        for i, spec in enumerate(self._loads):
            what = spec.where or f"loads[{i}]"
            try:
                body = which_body(spec.body, what)
                loads += spec.load.resolve(r, body, what)
            except InputError as exc:
                raise exc.at(what) from None
        _name_loads(loads)

        gravity = None
        if self.gravity is not None:
            try:
                gravity = _resolve_gravity(r, self.gravity)
            except InputError as exc:
                raise exc.at("gravity") from None
            case = self.gravity_case or DEFAULT_CASE
            for body in bodies.values():
                if body.mass is not None and body.mass.mass != 0:
                    loads.append(
                        AppliedWrench(
                            f"{body.name} weight",
                            body.name,
                            case,
                            body.mass.mass * gravity,
                            np.zeros(3),
                            body.mass.cog,
                            "gravity",
                        )
                    )

        for body in bodies.values():
            if body.motion is None:
                continue
            mprops, motion = body.mass, body.motion
            force = -mprops.mass * motion.acceleration
            if body.particle:
                moment = np.zeros(3)
            else:
                I, w = mprops.inertia, motion.angular_velocity
                moment = -(I @ motion.angular_acceleration + np.cross(w, I @ w))
            loads.append(
                AppliedWrench(
                    f"{body.name} inertia",
                    body.name,
                    motion.case,
                    force,
                    moment,
                    mprops.cog,
                    "inertia",
                )
            )

        for w in loads:
            if bodies[w.body].particle and w.kind in ("moment", "distributed"):
                what = "a couple" if w.kind == "moment" else "a distributed load"
                raise InputError(
                    f"a particle cannot take {what} (load {w.name!r} on {w.body!r}); "
                    "put it on a rigid body"
                )

        cases = []
        for w in loads:
            if w.case not in cases:
                cases.append(w.case)
        if not cases:
            cases = [DEFAULT_CASE]
        combinations = {}
        for cname, factors in self.combinations.items():
            if cname in cases:
                raise InputError(f"combination {cname!r} has the same name as a load case")
            resolved = {}
            for case, factor in factors.items():
                if case not in cases:
                    raise InputError(
                        f"unknown load case {case!r} (cases: {', '.join(cases)})",
                        f"combinations.{cname}",
                    )
                resolved[case] = ctx.scalar(factor, "dimensionless")
            combinations[cname] = resolved

        checks = [
            self._resolve_check(c, i, ctx, bodies, joints, cases, combinations)
            for i, c in enumerate(self.checks)
        ]

        return BuiltModel(
            name=self.name,
            description=self.description,
            planar=self.planar,
            units=self.units,
            output_units=self.output_units,
            parameters=params,
            points=points,
            bodies=bodies,
            joints=joints,
            loads=loads,
            gravity=gravity,
            cases=cases,
            combinations=combinations,
            checks=checks,
            source_path=None if self.source is None else self.source.path,
            source_text=None if self.source is None else self.source.text,
            source_sha256=None if self.source is None else self.source.sha256,
            source_size=None if self.source is None else self.source.size,
            overrides={k: str(v) for k, v in (overrides or {}).items()},
        )

    def _resolve_check(self, c: Check, i, ctx, bodies, joints, cases, combinations):
        what = f"checks[{i}]"
        target = c.target.strip()
        if "." not in target and target in joints and joints[target].kind == "unknown":
            target = f"{target}.{target}"  # a solved load: 'P' means P.P
        owner, _, comp = target.partition(".")
        if owner in bodies and comp == "mass":
            kind = "mass"
        elif owner in joints:
            kind = _check_kind(joints[owner], comp, what)
        else:
            raise InputError(
                f"target {target!r} must be '<support or joint>.<component>', "
                "'<body>.mass' or the name of an unknown load",
                what,
            )
        if c.case is not None and c.case not in cases and c.case not in combinations:
            raise InputError(f"unknown case {c.case!r}", what)
        if c.expect is None and c.max is None and c.min is None:
            raise InputError("a check needs 'expect', 'max' or 'min'", what)
        expect = None if c.expect is None else ctx.scalar(c.expect, kind)
        tol = 0.0
        if expect is not None:
            t = str(c.tolerance).strip()
            if t.endswith("%"):
                tol = abs(expect) * float(t[:-1]) / 100
            else:
                tol = ctx.scalar(c.tolerance, kind)
        unit = None
        for raw in (c.expect, c.max, c.min):
            if raw is not None and not (q := ctx.quantity(raw)).unitless:
                unit = f"{q.units}"
                break
        return ResolvedCheck(
            c.name or c.target.strip(),
            target,
            kind,
            c.case,
            expect,
            tol,
            None if c.max is None else ctx.scalar(c.max, kind),
            None if c.min is None else ctx.scalar(c.min, kind),
            unit,
        )

    def solve(self, overrides: Mapping[str, Any] | None = None):
        from engmech.solver import solve

        return solve(self.build(overrides))


def _check_kind(joint: ResolvedJoint, comp: str, what: str) -> str:
    if comp in ("Fx", "Fy", "Fz", "F"):
        return "force"
    if comp in ("Mx", "My", "Mz", "M"):
        return "moment"
    for c in joint.geometry.components:
        if c.label and c.label == comp:
            return c.kind
    labels = sorted({c.label for c in joint.geometry.components if c.label})
    options = ", ".join(["Fx", "Fy", "Fz", "Mx", "My", "Mz", "F", "M", *labels])
    raise InputError(f"unknown component {comp!r} for {joint.name!r} (use one of {options})", what)


def _resolve_gravity(r: Resolver, value: Any) -> np.ndarray:
    if isinstance(value, str):
        axis = parse_axis_name(value)
        if axis is not None:
            return r._check_in_plane(axis * STANDARD_GRAVITY, "gravity")
    if isinstance(value, Mapping):
        if set(value) - {"direction", "magnitude"} or "direction" not in value:
            raise InputError("expected a vector, an axis like -z, or {direction, magnitude}")
        g = r.scalar(value.get("magnitude", f"{STANDARD_GRAVITY} m/s^2"), "acceleration")
        return g * r.direction(value["direction"], "gravity.direction")
    v = r.vector(value, "acceleration", "gravity")
    unit(v, "gravity")
    return v


def _name_loads(loads: list[AppliedWrench]) -> None:
    """Give default-named loads unique names: F1, F2, M1, w1, ..."""
    counts: dict[str, int] = {}
    for w in loads:
        if w.name in ("F", "M", "w"):
            counts[w.name] = counts.get(w.name, 0) + 1
    seen: dict[str, int] = {}
    for w in loads:
        if w.name in counts and counts[w.name] > 1:
            seen[w.name] = seen.get(w.name, 0) + 1
            w.name = f"{w.name}{seen[w.name]}"
