"""Turn raw user values into SI geometry.

A :class:`Resolver` knows the unit system, parameters, named points and
whether the analysis is planar. Every input the user writes, in a file or
through the Python API, passes through it, so the same flexible forms work
everywhere:

* positions: ``[1, 2, 0]``, ``"[100, 0] mm"``, or a named point ``"A"``
* directions: ``"+x"``, ``"-z"``, ``[1, 1, 0]``
* forces: vectors, or ``{magnitude, direction | angle | toward | along}``
* moments: vectors, planar scalars, or ``{magnitude, axis}``
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engmech.errors import InputError
from engmech.spatial import EZ, frame_from_axes, parse_axis_name, rotation_about_axis, unit
from engmech.units import Context, split_vector


@dataclass
class Resolver:
    ctx: Context = field(default_factory=Context)
    points: Mapping[str, np.ndarray] = field(default_factory=dict)
    planar: bool = False

    # ------------------------------------------------------------ scalars

    def scalar(self, value: Any, kind: str) -> float:
        return self.ctx.scalar(value, kind)

    def positive(self, value: Any, kind: str, what: str) -> float:
        v = self.scalar(value, kind)
        if not v > 0:
            raise InputError(f"{what} must be positive")
        return v

    # ------------------------------------------------------------ vectors

    def _components(self, value: Any) -> int:
        try:
            items, _ = split_vector(value)
        except InputError:
            return -1
        return len(items)

    def vector(self, value: Any, kind: str, what: str = "vector") -> np.ndarray:
        """A 3D vector; planar models also accept 2 components (z = 0)."""
        n = self._components(value)
        if self.planar and n == 2:
            return np.append(self.ctx.vector(value, kind, 2), 0.0)
        if n == 3:
            v = self.ctx.vector(value, kind, 3)
            if self.planar and abs(v[2]) > 0:
                raise InputError(f"{what} must lie in the xy-plane in a planar analysis")
            return v
        expected = "2 or 3" if self.planar else "3"
        raise InputError(f"{what} needs {expected} components, got {value!r}")

    def position(self, value: Any, what: str = "position") -> np.ndarray:
        if isinstance(value, str) and value.strip() in self.points:
            return np.array(self.points[value.strip()], dtype=float)
        if isinstance(value, str) and value.strip().isidentifier():
            known = ", ".join(self.points) or "none defined"
            raise InputError(f"unknown point {value!r} (points: {known})")
        return self.vector(value, "length", what)

    def direction(self, value: Any, what: str = "direction") -> np.ndarray:
        """A unit direction. Accepts '+x'/'-y'/'z', vectors of any length unit,
        or a planar angle like '30 deg' measured from +x towards +y."""
        if isinstance(value, str):
            axis = parse_axis_name(value)
            if axis is not None:
                return self._check_in_plane(axis, what)
            if self._components(value) < 0:
                angle = self.scalar(value, "angle")
                return np.array([np.cos(angle), np.sin(angle), 0.0])
        v = self.vector(value, "dimensionless", what) if _is_bare(value) else None
        if v is None:
            v = self.vector(value, "length", what)
        return self._check_in_plane(unit(v, what), what)

    def axis(self, value: Any, what: str = "axis") -> np.ndarray:
        """Like direction, but out-of-plane (z) axes are allowed in planar models."""
        if isinstance(value, str) and (a := parse_axis_name(value)) is not None:
            return a
        items = self._components(value)
        if items in (2, 3):
            v = (
                self.ctx.vector(value, "dimensionless", items)
                if _is_bare(value)
                else (self.ctx.vector(value, "length", items))
            )
            return unit(np.append(v, 0.0) if items == 2 else v, what)
        raise InputError(f"{what}: expected an axis like +z or [0, 0, 1], got {value!r}")

    def _check_in_plane(self, v: np.ndarray, what: str) -> np.ndarray:
        if self.planar and abs(v[2]) > 1e-12:
            raise InputError(f"{what} must lie in the xy-plane in a planar analysis")
        return v

    # ------------------------------------------------------------ loads

    def force(self, value: Any, at: np.ndarray | None, what: str = "force") -> np.ndarray:
        return self._directed(value, "force", at, what)

    def moment(self, value: Any, what: str = "moment") -> np.ndarray:
        if isinstance(value, Mapping):
            _only(value, {"magnitude", "axis"}, what)
            if "magnitude" not in value or "axis" not in value:
                raise InputError(f"{what}: needs magnitude and axis")
            axis = self.axis(value["axis"], what)
            if self.planar and np.linalg.norm(axis[:2]) > 1e-12:
                raise InputError(f"{what} must be about the z-axis in a planar analysis")
            return self.scalar(value["magnitude"], "moment") * axis
        n = self._components(value)
        if n < 0:  # a single value
            if not self.planar:
                raise InputError(
                    f"{what}: a scalar moment is only allowed in planar analysis; "
                    "use a vector or {magnitude, axis}"
                )
            return np.array([0.0, 0.0, self.scalar(value, "moment")])
        if n != 3:
            raise InputError(f"{what} needs 3 components, got {value!r}")
        v = self.ctx.vector(value, "moment", 3)
        if self.planar and np.linalg.norm(v[:2]) > 0:
            raise InputError(f"{what} must be about the z-axis in a planar analysis")
        return v

    def _directed(self, value: Any, kind: str, at, what: str) -> np.ndarray:
        if not isinstance(value, Mapping):
            return self.vector(value, kind, what)
        keys = {"magnitude", "direction", "angle", "toward", "along"}
        _only(value, keys, what)
        if "magnitude" not in value:
            raise InputError(f"{what}: needs a magnitude")
        magnitude = self.scalar(value["magnitude"], kind)
        given = [k for k in ("direction", "angle", "toward", "along") if k in value]
        if len(given) != 1:
            raise InputError(f"{what}: give exactly one of direction, angle, toward or along")
        how = given[0]
        if how == "direction":
            d = self.direction(value["direction"], f"{what}.direction")
        elif how == "angle":
            a = self.scalar(value["angle"], "angle")
            d = np.array([np.cos(a), np.sin(a), 0.0])
        elif how == "toward":
            if at is None:
                raise InputError(f"{what}: 'toward' needs the load's 'at' point")
            d = unit(self.position(value["toward"]) - at, f"{what}.toward (points coincide)")
        else:
            ends = value["along"]
            if not isinstance(ends, list | tuple) or len(ends) != 2:
                raise InputError(f"{what}.along: expected [from_point, to_point]")
            p, q = (self.position(e) for e in ends)
            d = unit(q - p, f"{what}.along (points coincide)")
        return magnitude * self._check_in_plane(d, what)

    # ------------------------------------------------------------ orientation

    def orientation(self, value: Any, what: str = "orientation") -> np.ndarray:
        """Rotation matrix from {x|y|z: axis, ...} (two axes), {axis, angle}
        or {euler: [a, b, c], sequence: 'xyz'} (extrinsic, lowercase) /
        'XYZ' (intrinsic, uppercase)."""
        if value is None:
            return np.eye(3)
        if not isinstance(value, Mapping):
            raise InputError(f"{what}: expected a mapping")
        if {"x", "y", "z"} & set(value):
            _only(value, {"x", "y", "z"}, what)
            return frame_from_axes(**{k: self.axis(v, f"{what}.{k}") for k, v in value.items()})
        if "axis" in value:
            _only(value, {"axis", "angle"}, what)
            return rotation_about_axis(
                self.axis(value["axis"]), self.scalar(value.get("angle", 0), "angle")
            )
        if "euler" in value:
            _only(value, {"euler", "sequence"}, what)
            from scipy.spatial.transform import Rotation

            seq = value.get("sequence", "xyz")
            rad = self.ctx.vector(value["euler"], "angle", 3)
            try:
                return Rotation.from_euler(seq, rad).as_matrix()
            except ValueError as exc:
                raise InputError(f"{what}: {exc}") from None
        raise InputError(f"{what}: expected two axes, {{axis, angle}} or {{euler, sequence}}")

    def check_planar_frame(self, frame: np.ndarray, what: str) -> None:
        if self.planar and abs(abs(frame[:, 2] @ EZ) - 1) > 1e-9:
            raise InputError(f"{what}: in a planar analysis the local z-axis must be the global z")


def _is_bare(value: Any) -> bool:
    """True if a vector is given without any units (a pure direction)."""
    try:
        items, unit_text = split_vector(value)
    except InputError:
        return False
    if unit_text:
        return False
    for item in items:
        if isinstance(item, int | float | np.integer | np.floating):
            continue
        if isinstance(item, str):
            try:
                float(item)
                continue
            except ValueError:
                return False
        return False
    return True


def _only(mapping: Mapping, allowed: set[str], what: str) -> None:
    extra = set(mapping) - allowed
    if extra:
        raise InputError(
            f"{what}: unexpected {', '.join(sorted(extra))} (allowed: {', '.join(sorted(allowed))})"
        )
