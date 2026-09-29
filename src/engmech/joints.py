"""Joint and support types.

A joint connects body ``a`` to body ``b`` (for a support, ``a`` is the
ground). Each joint type defines which force and moment components it can
transmit, in a local frame at the joint. The solver treats the magnitude of
each transmitted component as an unknown; the wrench the joint applies to
body ``b`` is the sum of those components, and body ``a`` receives the
opposite.

Joint specs hold raw user input; :meth:`JointType.resolve` turns them into
SI geometry through a :class:`~engmech.inputs.Resolver`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from engmech.errors import InputError
from engmech.inputs import Resolver
from engmech.spatial import EZ, frame_from_axis, unit

LOCAL = ("x", "y", "z")
COMPONENT_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass(frozen=True)
class Component:
    """One transmitted wrench component: a unit force or unit moment on body b."""

    kind: str  # "force" | "moment"
    direction: np.ndarray  # global unit vector
    compliance: float = 0.0  # 1/stiffness in SI; 0 = rigid
    role: str = "constraint"  # constraint | drive | scalar | unknown
    label: str = ""


@dataclass
class JointGeometry:
    point_a: np.ndarray
    point_b: np.ndarray
    frame: np.ndarray
    components: list[Component]
    scalar_label: str | None = None  # headline scalar result ("N", "T", ...)
    sign_limit: str | None = None  # "compression" (push only) | "tension" (pull only)


@dataclass
class JointType:
    """Base class for joint specifications."""

    type_name: ClassVar[str] = "joint"
    description: ClassVar[str] = ""
    at: Any = None
    stiffness: Any = None

    def resolve(self, r: Resolver, what: str) -> JointGeometry:  # pragma: no cover
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers

    def _point(self, r: Resolver, what: str) -> np.ndarray:
        if self.at is None:
            raise InputError("needs a location 'at'", what)
        return r.position(self.at, f"{what}.at")

    def _local_components(
        self, r: Resolver, frame: np.ndarray, names: Sequence[str], what: str
    ) -> list[Component]:
        kt, kr = self._stiffness_vectors(r, what)
        comps = []
        for name in names:
            i = LOCAL.index(name[1])
            if name[0] == "F":
                comps.append(Component("force", frame[:, i], _compliance(kt[i])))
            else:
                comps.append(Component("moment", frame[:, i], _compliance(kr[i])))
        return comps

    def _stiffness_vectors(self, r: Resolver, what: str) -> tuple[np.ndarray, np.ndarray]:
        kt = np.full(3, np.inf)
        kr = np.full(3, np.inf)
        s = self.stiffness
        if s is None:
            return kt, kr
        if not isinstance(s, Mapping):
            raise InputError(
                "stiffness must be {translational: ..., rotational: ...}", f"{what}.stiffness"
            )
        extra = set(s) - {"translational", "rotational"}
        if extra:
            raise InputError(
                f"unexpected {', '.join(sorted(extra))} (allowed: translational, rotational)",
                f"{what}.stiffness",
            )
        if "translational" in s:
            kt = _stiffness_triplet(r, s["translational"], "stiffness", f"{what}.stiffness")
        if "rotational" in s:
            kr = _stiffness_triplet(r, s["rotational"], "rotational_stiffness", f"{what}.stiffness")
        return kt, kr

    def _scalar_compliance(self, r: Resolver, what: str) -> float:
        if self.stiffness is None:
            return 0.0
        k = r.scalar(self.stiffness, "stiffness")
        if not k > 0:
            raise InputError("stiffness must be positive", f"{what}.stiffness")
        return 1.0 / k

    def _axis(self, r: Resolver, value, what: str, planar_default=None) -> np.ndarray:
        if value is None:
            if r.planar and planar_default is not None:
                return planar_default
            raise InputError(f"a {self.type_name} joint needs an 'axis'", what)
        return r.axis(value, f"{what}.axis")


def _compliance(k: float) -> float:
    return 0.0 if np.isinf(k) else 1.0 / k


def _stiffness_triplet(r: Resolver, value, kind: str, what: str) -> np.ndarray:
    if isinstance(value, str) and value.strip().lower() in ("rigid", "inf", "infinite"):
        return np.full(3, np.inf)
    if isinstance(value, list | tuple) or (isinstance(value, str) and value.strip()[:1] in "[("):
        from engmech.units import evaluate, split_vector

        items, unit_text = split_vector(value)
        if len(items) != 3:
            raise InputError("expected one stiffness or three (local x, y, z)", what)
        unit_q = evaluate(unit_text, r.ctx.parameters, units_only=True) if unit_text else None
        out = []
        for item in items:
            if isinstance(item, str) and item.strip().lower() in ("rigid", "inf"):
                out.append(np.inf)
                continue
            q = r.ctx.quantity(item, kind)
            if unit_q is not None:
                if not q.unitless:
                    raise InputError("give units once after the brackets or on each item", what)
                q = q.to("dimensionless").magnitude * unit_q
            out.append(r.scalar(q, kind))
        k = np.array(out)
    else:
        k = np.full(3, r.scalar(value, kind))
    if np.any(k <= 0):
        raise InputError("stiffness must be positive", what)
    return k


# --------------------------------------------------------------------------- types


@dataclass
class Fixed(JointType):
    """Welded / built-in: transmits all three forces and all three moments."""

    type_name: ClassVar[str] = "fixed"
    description: ClassVar[str] = "all forces and moments"
    orientation: Any = None

    def resolve(self, r, what):
        p = self._point(r, what)
        frame = r.orientation(self.orientation, f"{what}.orientation")
        names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        return JointGeometry(p, p, frame, self._local_components(r, frame, names, what))


@dataclass
class Pin(JointType):
    """Revolute hinge: free rotation about ``axis``, everything else held.

    With ``actuated=True`` a motor/brake torque about the axis is solved for.
    """

    type_name: ClassVar[str] = "pin"
    description: ClassVar[str] = "all forces, moments except about the pin axis"
    axis: Any = None
    actuated: bool = False

    def resolve(self, r, what):
        p = self._point(r, what)
        axis = self._axis(r, self.axis, what, planar_default=EZ)
        if r.planar and abs(abs(axis @ EZ) - 1) > 1e-9:
            raise InputError("a pin's axis must be the z-axis in a planar analysis", what)
        frame = frame_from_axis(axis)
        comps = self._local_components(r, frame, ["Fx", "Fy", "Fz", "Mx", "My"], what)
        if self.actuated:
            comps.append(Component("moment", frame[:, 2], role="drive", label="drive"))
        return JointGeometry(p, p, frame, comps)


@dataclass
class Ball(JointType):
    """Ball-and-socket (spherical): transmits forces only."""

    type_name: ClassVar[str] = "ball"
    description: ClassVar[str] = "all forces, no moments"

    def resolve(self, r, what):
        p = self._point(r, what)
        frame = np.eye(3)
        return JointGeometry(
            p, p, frame, self._local_components(r, frame, ["Fx", "Fy", "Fz"], what)
        )


@dataclass
class Slider(JointType):
    """Prismatic: free translation along ``axis``, no rotation.

    With ``actuated=True`` the drive force along the axis is solved for.
    """

    type_name: ClassVar[str] = "slider"
    description: ClassVar[str] = "forces across the slide axis and all moments"
    axis: Any = None
    actuated: bool = False

    def resolve(self, r, what):
        p = self._point(r, what)
        axis = self._axis(r, self.axis, what)
        frame = frame_from_axis(axis)
        if r.planar and abs(axis @ EZ) > 1e-9:
            raise InputError("a slider's axis must lie in the xy-plane in a planar analysis", what)
        comps = self._local_components(r, frame, ["Fx", "Fy", "Mx", "My", "Mz"], what)
        if self.actuated:
            comps.append(Component("force", frame[:, 2], role="drive", label="drive"))
        return JointGeometry(p, p, frame, comps)


@dataclass
class Cylindrical(JointType):
    """Free rotation about and translation along ``axis``."""

    type_name: ClassVar[str] = "cylindrical"
    description: ClassVar[str] = "forces and moments across the axis"
    axis: Any = None

    def resolve(self, r, what):
        p = self._point(r, what)
        axis = self._axis(r, self.axis, what)
        if r.planar and abs(axis @ EZ) > 1e-9:
            raise InputError("the axis must lie in the xy-plane in a planar analysis", what)
        frame = frame_from_axis(axis)
        comps = self._local_components(r, frame, ["Fx", "Fy", "Mx", "My"], what)
        return JointGeometry(p, p, frame, comps)


@dataclass
class Bearing(JointType):
    """Shaft bearing or textbook hinge: carries forces across ``axis`` but no
    moments; with ``thrust=True`` it also carries axial force. This is the
    usual idealisation for properly aligned bearings and hinges, which are
    assumed not to take couples."""

    type_name: ClassVar[str] = "bearing"
    description: ClassVar[str] = "forces across the axis (and along it with thrust), no moments"
    axis: Any = None
    thrust: bool = False

    def resolve(self, r, what):
        p = self._point(r, what)
        axis = self._axis(r, self.axis, what, planar_default=EZ)
        frame = frame_from_axis(axis)
        names = ["Fx", "Fy", "Fz"] if self.thrust else ["Fx", "Fy"]
        return JointGeometry(p, p, frame, self._local_components(r, frame, names, what))


@dataclass
class Universal(JointType):
    """Hooke joint: rotation about two perpendicular ``axes`` is free."""

    type_name: ClassVar[str] = "universal"
    description: ClassVar[str] = "all forces, moment about the cross axis"
    axes: Any = None

    def resolve(self, r, what):
        p = self._point(r, what)
        if not isinstance(self.axes, list | tuple) or len(self.axes) != 2:
            raise InputError("a universal joint needs 'axes: [axis1, axis2]'", what)
        a1, a2 = (r.axis(a, f"{what}.axes") for a in self.axes)
        if abs(a1 @ a2) > 1e-9:
            raise InputError("the two axes of a universal joint must be perpendicular", what)
        frame = np.column_stack([a1, a2, np.cross(a1, a2)])
        comps = self._local_components(r, frame, ["Fx", "Fy", "Fz", "Mz"], what)
        return JointGeometry(p, p, frame, comps)


@dataclass
class Roller(JointType):
    """Single force along ``normal`` (roller, rocker, frictionless surface).

    Reported as N, positive when pushing on the body along ``normal``. A
    roller can also pull; use ``contact`` when it can only push.
    """

    type_name: ClassVar[str] = "roller"
    description: ClassVar[str] = "one force along the normal"
    normal: Any = None

    def resolve(self, r, what):
        p = self._point(r, what)
        if self.normal is None:
            raise InputError(f"a {self.type_name} needs a 'normal' direction", what)
        n = r.direction(self.normal, f"{what}.normal")
        comp = Component("force", n, self._scalar_compliance(r, what), "scalar", "N")
        geometry = JointGeometry(p, p, frame_from_axis(n), [comp], scalar_label="N")
        if self.type_name == "contact":
            geometry.sign_limit = "compression"
        return geometry


@dataclass
class Contact(Roller):
    """Like a roller, but can only push (N >= 0); a pull is flagged as lift-off."""

    type_name: ClassVar[str] = "contact"
    description: ClassVar[str] = "one pushing force along the normal"


@dataclass
class Link(JointType):
    """Two-force member (strut, rod, link) along the line between two points.

    Reported as T, positive in tension. As a support give ``at`` (the point
    on the body) and ``anchor`` (the fixed end); between two bodies give
    ``ends: [point_on_first_body, point_on_second_body]``.
    """

    type_name: ClassVar[str] = "link"
    description: ClassVar[str] = "one axial force (tension or compression)"
    anchor: Any = None
    ends: Any = None

    def resolve(self, r, what):
        if self.ends is not None:
            if self.at is not None or self.anchor is not None:
                raise InputError("give either 'ends' or 'at' + 'anchor', not both", what)
            if not isinstance(self.ends, list | tuple) or len(self.ends) != 2:
                raise InputError("'ends' must be [point_on_first, point_on_second]", what)
            pa, pb = (r.position(e, f"{what}.ends") for e in self.ends)
        else:
            if self.at is None or self.anchor is None:
                raise InputError(
                    f"a {self.type_name} support needs 'at' (on the body) and 'anchor' "
                    "(the fixed end); between bodies use 'ends'",
                    what,
                )
            pb = r.position(self.at, f"{what}.at")
            pa = r.position(self.anchor, f"{what}.anchor")
        if np.linalg.norm(pa - pb) == 0:
            raise InputError("the two ends must be different points", what)
        # tension pulls body b toward end a
        u = unit(pa - pb)
        comp = Component("force", u, self._scalar_compliance(r, what), "scalar", "T")
        geometry = JointGeometry(pa, pb, frame_from_axis(u), [comp], scalar_label="T")
        if self.type_name == "cable":
            geometry.sign_limit = "tension"
        return geometry


@dataclass
class Cable(Link):
    """Like a link, but can only pull (T >= 0); compression is flagged as slack."""

    type_name: ClassVar[str] = "cable"
    description: ClassVar[str] = "one tensile force"


@dataclass
class Custom(JointType):
    """Hold any subset of Fx, Fy, Fz, Mx, My, Mz in a chosen local frame."""

    type_name: ClassVar[str] = "custom"
    description: ClassVar[str] = "chosen components"
    constrain: Sequence[str] = field(default_factory=list)
    orientation: Any = None

    def resolve(self, r, what):
        p = self._point(r, what)
        names = [str(c).strip() for c in self.constrain]
        bad = [c for c in names if c not in COMPONENT_NAMES]
        if bad or not names:
            raise InputError(
                f"'constrain' must list components from {', '.join(COMPONENT_NAMES)}"
                + (f" (got {', '.join(bad)})" if bad else ""),
                what,
            )
        if len(set(names)) != len(names):
            raise InputError("'constrain' lists a component twice", what)
        frame = r.orientation(self.orientation, f"{what}.orientation")
        return JointGeometry(p, p, frame, self._local_components(r, frame, names, what))


@dataclass
class UnknownForce(JointType):
    """An applied force of unknown magnitude along a known direction."""

    type_name: ClassVar[str] = "unknown force"
    direction: Any = None
    label: str = "P"

    def resolve(self, r, what):
        p = self._point(r, what)
        if self.direction is None:
            raise InputError("an unknown force needs a 'direction'", what)
        d = r.direction(self.direction, f"{what}.direction")
        comp = Component("force", d, role="unknown", label=self.label)
        return JointGeometry(p, p, frame_from_axis(d), [comp], scalar_label=self.label)


@dataclass
class UnknownMoment(JointType):
    """An applied couple of unknown magnitude about a known axis."""

    type_name: ClassVar[str] = "unknown moment"
    axis: Any = None
    label: str = "M"

    def resolve(self, r, what):
        p = r.position(self.at, f"{what}.at") if self.at is not None else None
        axis = self._axis(r, self.axis, what, planar_default=EZ)
        if r.planar and abs(abs(axis @ EZ) - 1) > 1e-9:
            raise InputError("the axis must be z in a planar analysis", what)
        comp = Component("moment", axis, role="unknown", label=self.label)
        p = np.zeros(3) if p is None else p
        return JointGeometry(p, p, frame_from_axis(axis), [comp], scalar_label=self.label)


JOINT_TYPES: dict[str, type[JointType]] = {
    "fixed": Fixed,
    "weld": Fixed,
    "pin": Pin,
    "revolute": Pin,
    "hinge": Pin,
    "ball": Ball,
    "spherical": Ball,
    "slider": Slider,
    "prismatic": Slider,
    "cylindrical": Cylindrical,
    "bearing": Bearing,
    "universal": Universal,
    "roller": Roller,
    "contact": Contact,
    "link": Link,
    "strut": Link,
    "cable": Cable,
    "custom": Custom,
}
