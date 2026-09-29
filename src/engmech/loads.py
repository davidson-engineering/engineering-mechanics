"""Applied loads and prescribed motion.

Every load resolves to one or more :class:`AppliedWrench` objects: a force
and a moment about a point, acting on one body, in one load case. The solver
only ever sees wrenches; the extra fields describe the load for tables and
plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engmech.errors import InputError
from engmech.inputs import Resolver
from engmech.spatial import unit

DEFAULT_CASE = "default"


@dataclass
class AppliedWrench:
    name: str
    body: str
    case: str
    force: np.ndarray
    moment: np.ndarray  # about `point`
    point: np.ndarray
    kind: str  # force | moment | distributed | gravity | inertia
    detail: dict = field(default_factory=dict)


@dataclass
class Force:
    """A point force. ``value`` is a vector or {magnitude, direction|angle|toward|along}."""

    value: Any
    at: Any
    name: str | None = None
    case: str | None = None

    def resolve(self, r: Resolver, body: str, what: str) -> list[AppliedWrench]:
        p = r.position(self.at, f"{what}.at")
        f = r.force(self.value, p, f"{what}.force")
        return [
            AppliedWrench(
                self.name or "F", body, self.case or DEFAULT_CASE, f, np.zeros(3), p, "force"
            )
        ]


@dataclass
class Moment:
    """A couple (free vector). ``at`` only positions the arrow in plots."""

    value: Any
    at: Any = None
    name: str | None = None
    case: str | None = None

    def resolve(self, r: Resolver, body: str, what: str) -> list[AppliedWrench]:
        m = r.moment(self.value, f"{what}.moment")
        p = r.position(self.at, f"{what}.at") if self.at is not None else None
        return [
            AppliedWrench(
                self.name or "M",
                body,
                self.case or DEFAULT_CASE,
                np.zeros(3),
                m,
                np.zeros(3) if p is None else p,
                "moment",
                {"placed": p is not None},
            )
        ]


@dataclass
class DistributedLoad:
    """Line load from ``start`` to ``end`` varying linearly in intensity.

    ``intensity`` is a force per length: one value (uniform) or
    ``{start: w1, end: w2}`` (linear). ``direction`` is the direction of
    positive intensity. With ``projected=True`` the intensity is per unit
    length of the projection of the span onto the plane normal to
    ``direction`` (e.g. snow on an inclined roof).
    """

    start: Any
    end: Any
    intensity: Any
    direction: Any
    name: str | None = None
    case: str | None = None
    projected: bool = False

    def resolve(self, r: Resolver, body: str, what: str) -> list[AppliedWrench]:
        a = r.position(self.start, f"{what}.start")
        b = r.position(self.end, f"{what}.end")
        d = r.direction(self.direction, f"{what}.direction")
        if isinstance(self.intensity, dict):
            extra = set(self.intensity) - {"start", "end"}
            if extra or len(self.intensity) != 2:
                raise InputError("intensity must be one value or {start: ..., end: ...}", what)
            w1 = r.scalar(self.intensity["start"], "force_per_length")
            w2 = r.scalar(self.intensity["end"], "force_per_length")
        else:
            w1 = w2 = r.scalar(self.intensity, "force_per_length")
        span = b - a
        length = float(np.linalg.norm(span))
        if length == 0:
            raise InputError("start and end must differ", what)
        u = unit(span)
        # load per unit length along the member
        factor = float(np.linalg.norm(np.cross(u, d))) if self.projected else 1.0
        total = factor * length * (w1 + w2) / 2
        first_moment = factor * length**2 * (w1 + 2 * w2) / 6  # ∫ s w(s) ds
        force = total * d
        moment = first_moment * np.cross(u, d)  # about `start`
        detail = {
            "start": a,
            "end": b,
            "w1": w1,
            "w2": w2,
            "direction": d,
            "resultant": total,
            "projected": self.projected,
        }
        if abs(w1 + w2) > 0:
            detail["centroid"] = a + span * (w1 + 2 * w2) / (3 * (w1 + w2))
        return [
            AppliedWrench(
                self.name or "w",
                body,
                self.case or DEFAULT_CASE,
                force,
                moment,
                a,
                "distributed",
                detail,
            )
        ]


@dataclass
class UnknownLoad:
    """A load whose magnitude is solved for: a force along ``direction`` at
    ``at``, or a couple about ``axis``. Solved like a support reaction."""

    name: str
    at: Any = None
    direction: Any = None
    axis: Any = None

    def __post_init__(self):
        if (self.direction is None) == (self.axis is None):
            raise InputError(
                f"unknown load {self.name!r}: give 'direction' (a force) or 'axis' (a couple)"
            )


@dataclass
class Motion:
    """Instantaneous motion of a body for inverse dynamics.

    Give ``angular_velocity`` and ``angular_acceleration`` of the body, and
    either the ``acceleration`` of its centre of gravity, or a ``pivot``
    point whose acceleration (``pivot_acceleration``, default zero) is known;
    the cog acceleration then follows from rigid-body kinematics.
    """

    angular_velocity: Any = None
    angular_acceleration: Any = None
    acceleration: Any = None
    pivot: Any = None
    pivot_acceleration: Any = None
    case: str | None = None

    def resolve(self, r: Resolver, cog: np.ndarray, what: str) -> ResolvedMotion:
        w = self._angular(r, self.angular_velocity, "angular_velocity", what)
        al = self._angular(r, self.angular_acceleration, "angular_acceleration", what)
        if self.acceleration is not None and self.pivot is not None:
            raise InputError("give either 'acceleration' (of the cog) or 'pivot', not both", what)
        if self.pivot_acceleration is not None and self.pivot is None:
            raise InputError("'pivot_acceleration' needs a 'pivot'", what)
        if self.pivot is not None:
            p = r.position(self.pivot, f"{what}.pivot")
            ap = (
                np.zeros(3)
                if self.pivot_acceleration is None
                else r.vector(self.pivot_acceleration, "acceleration", f"{what}.pivot_acceleration")
            )
            rel = cog - p
            a = ap + np.cross(al, rel) + np.cross(w, np.cross(w, rel))
        elif self.acceleration is not None:
            a = r.vector(self.acceleration, "acceleration", f"{what}.acceleration")
        else:
            a = np.zeros(3)
        return ResolvedMotion(w, al, a, self.case or DEFAULT_CASE)

    @staticmethod
    def _angular(r: Resolver, value, kind: str, what: str) -> np.ndarray:
        if value is None:
            return np.zeros(3)
        if isinstance(value, dict):
            if set(value) != {"magnitude", "axis"}:
                raise InputError(f"{kind} must be a vector or {{magnitude, axis}}", what)
            axis = r.axis(value["axis"], f"{what}.{kind}")
            if r.planar and np.linalg.norm(axis[:2]) > 1e-12:
                raise InputError(f"{kind} must be about z in a planar analysis", f"{what}.{kind}")
            return r.scalar(value["magnitude"], kind) * axis
        from engmech.units import split_vector

        try:
            n = len(split_vector(value)[0])
        except InputError:
            n = 1  # a single value
        if n == 1 and not isinstance(value, list | tuple):
            if not r.planar:
                raise InputError(
                    f"{kind} must be a 3-vector or {{magnitude, axis}}", f"{what}.{kind}"
                )
            return np.array([0.0, 0.0, r.scalar(value, kind)])
        if n != 3:
            raise InputError(f"{kind} needs 3 components", f"{what}.{kind}")
        v = r.ctx.vector(value, kind, 3)
        if r.planar and np.linalg.norm(v[:2]) > 0:
            raise InputError(f"{kind} must be about z in a planar analysis", f"{what}.{kind}")
        return v


@dataclass
class ResolvedMotion:
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray
    acceleration: np.ndarray  # of the cog
    case: str

    def point_acceleration(self, cog: np.ndarray, point: np.ndarray) -> np.ndarray:
        rel = np.asarray(point) - cog
        w, al = self.angular_velocity, self.angular_acceleration
        return self.acceleration + np.cross(al, rel) + np.cross(w, np.cross(w, rel))
