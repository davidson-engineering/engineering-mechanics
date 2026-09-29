"""Mass properties of rigid bodies.

A :class:`MassProperties` holds mass, centre of gravity and the inertia tensor
about the centre of gravity, all in global axes and SI units. Shapes are
placed directly in global coordinates, and composites combine them with the
parallel-axis theorem. Holes are shapes with ``subtract=True`` (negative mass).

Inertia tensors are the true tensor, so the off-diagonal entries are the
negated products of inertia: I_xy = -∫ x y dm.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from engmech.errors import InputError
from engmech.spatial import frame_from_axis, unit


@dataclass(frozen=True)
class MassProperties:
    mass: float
    cog: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # about cog
    label: str = ""

    def __post_init__(self):
        object.__setattr__(self, "cog", np.asarray(self.cog, dtype=float).reshape(3))
        I = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        object.__setattr__(self, "inertia", 0.5 * (I + I.T))

    def inertia_about(self, point) -> np.ndarray:
        """Inertia tensor about ``point`` (global axes), by the parallel-axis theorem."""
        r = np.asarray(point, dtype=float) - self.cog
        return self.inertia + self.mass * ((r @ r) * np.eye(3) - np.outer(r, r))

    def principal(self) -> tuple[np.ndarray, np.ndarray]:
        """Principal moments (ascending) and axes (columns) about the cog."""
        moments, axes = np.linalg.eigh(self.inertia)
        moments[np.abs(moments) < 1e-12 * max(np.abs(moments).max(), 1e-300)] = 0.0
        # make the frame right-handed and the largest component of each axis positive
        for i in range(3):
            if axes[np.argmax(np.abs(axes[:, i])), i] < 0:
                axes[:, i] *= -1
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
        return moments, axes

    def radii_of_gyration(self) -> np.ndarray:
        moments, _ = self.principal()
        if self.mass <= 0:
            return np.full(3, np.nan)
        return np.sqrt(np.clip(moments, 0, None) / self.mass)

    def __add__(self, other: MassProperties) -> MassProperties:
        return combine([self, other])

    def check_physical(self) -> list[str]:
        """Return problems that make these properties physically impossible."""
        problems = []
        if self.mass <= 0:
            problems.append(f"total mass must be positive (got {self.mass:g} kg)")
            return problems
        moments, _ = self.principal()
        scale = max(abs(moments).max(), 1e-300)
        if moments.min() < -1e-9 * scale:
            problems.append("inertia tensor is not positive semi-definite")
        a, b, c = moments
        if a + b < c * (1 - 1e-9) - 1e-12 * scale:
            problems.append("principal moments violate the triangle inequality (I1 + I2 >= I3)")
        return problems


def combine(parts: Iterable[MassProperties], label: str = "") -> MassProperties:
    parts = list(parts)
    if not parts:
        return MassProperties(0.0, label=label)
    mass = sum(p.mass for p in parts)
    if abs(mass) < 1e-300:
        if any(p.mass != 0 for p in parts):
            raise InputError("combined mass is zero; cannot locate a centre of gravity")
        return MassProperties(0.0, label=label)
    cog = sum(p.mass * p.cog for p in parts) / mass
    inertia = sum(p.inertia_about(cog) for p in parts)
    return MassProperties(mass, cog, inertia, label=label)


# --------------------------------------------------------------------------- shapes


def _resolve_mass(mass, density, volume, what: str, subtract: bool) -> float:
    if (mass is None) == (density is None):
        raise InputError(f"{what}: give exactly one of mass or density")
    m = float(mass) if mass is not None else float(density) * volume
    if m < 0:
        raise InputError(f"{what}: mass must not be negative (use subtract: true for holes)")
    return -m if subtract else m


def _place(local_inertia: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return frame @ local_inertia @ frame.T


def point_mass(mass: float, at, *, subtract=False, label="point") -> MassProperties:
    m = _resolve_mass(mass, None, 0.0, label, subtract)
    return MassProperties(m, at, np.zeros((3, 3)), label)


def rod(
    start, end, *, mass=None, linear_density=None, subtract=False, label="rod"
) -> MassProperties:
    """Slender rod between two points (no inertia about its own axis)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    if length == 0:
        raise InputError(f"{label}: start and end must differ")
    if (mass is None) == (linear_density is None):
        raise InputError(f"{label}: give exactly one of mass or linear_density")
    m = float(mass) if mass is not None else float(linear_density) * length
    if m < 0:
        raise InputError(f"{label}: mass must not be negative")
    m = -m if subtract else m
    u = unit(end - start)
    inertia = m * length**2 / 12 * (np.eye(3) - np.outer(u, u))
    return MassProperties(m, (start + end) / 2, inertia, label)


def box(
    size, center, frame=None, *, mass=None, density=None, subtract=False, label="box"
) -> MassProperties:
    """Solid cuboid with edge lengths ``size`` along the local x, y, z axes."""
    a, b, c = (float(s) for s in size)
    if min(a, b, c) < 0 or (a * b * c == 0 and density is not None):
        raise InputError(f"{label}: size must be positive")
    m = _resolve_mass(mass, density, a * b * c, label, subtract)
    local = m / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    return MassProperties(m, center, _place(local, _frame(frame)), label)


def cylinder(
    radius,
    length,
    center,
    axis=(0, 0, 1),
    *,
    inner_radius=0.0,
    mass=None,
    density=None,
    subtract=False,
    label="cylinder",
) -> MassProperties:
    """Solid or hollow cylinder; a disc is a cylinder with length 0."""
    ro, ri, h = float(radius), float(inner_radius), float(length)
    if ro <= 0 or ri < 0 or ri >= ro or h < 0:
        raise InputError(f"{label}: need radius > inner_radius >= 0 and length >= 0")
    if h == 0 and density is not None:
        raise InputError(
            f"{label}: a thin disc (length 0) has no volume; give its 'mass', "
            "or a 'length' (thickness) with the density"
        )
    m = _resolve_mass(mass, density, np.pi * (ro**2 - ri**2) * h, label, subtract)
    r2 = ro**2 + ri**2
    local = np.diag([m * (3 * r2 + h**2) / 12, m * (3 * r2 + h**2) / 12, m * r2 / 2])
    return MassProperties(m, center, _place(local, frame_from_axis(axis)), label)


def sphere(
    radius, center, *, inner_radius=0.0, mass=None, density=None, subtract=False, label="sphere"
) -> MassProperties:
    ro, ri = float(radius), float(inner_radius)
    if ro <= 0 or ri < 0 or ri >= ro:
        raise InputError(f"{label}: need radius > inner_radius >= 0")
    m = _resolve_mass(mass, density, 4 / 3 * np.pi * (ro**3 - ri**3), label, subtract)
    i = 2 / 5 * m * (ro**5 - ri**5) / (ro**3 - ri**3)
    return MassProperties(m, center, i * np.eye(3), label)


def cone(
    radius,
    height,
    base_center,
    axis=(0, 0, 1),
    *,
    mass=None,
    density=None,
    subtract=False,
    label="cone",
) -> MassProperties:
    """Solid right circular cone; ``axis`` points from the base to the apex."""
    r, h = float(radius), float(height)
    if r <= 0 or h <= 0:
        raise InputError(f"{label}: radius and height must be positive")
    m = _resolve_mass(mass, density, np.pi * r * r * h / 3, label, subtract)
    frame = frame_from_axis(axis)
    cog = np.asarray(base_center, float) + frame[:, 2] * h / 4
    transverse = 3 / 20 * m * r * r + 3 / 80 * m * h * h
    local = np.diag([transverse, transverse, 3 / 10 * m * r * r])
    return MassProperties(m, cog, _place(local, frame), label)


def custom(mass, cog, inertia=None, about=None, frame=None, *, label="custom") -> MassProperties:
    """Mass properties from CAD or a datasheet.

    ``inertia`` is a 3x3 tensor, or three principal/diagonal values, expressed
    in ``frame`` (default: global axes) about ``about`` (default: the cog).
    """
    m = float(mass)
    cog = np.asarray(cog, float)
    if inertia is None:
        I = np.zeros((3, 3))
    else:
        I = np.asarray(inertia, float)
        if I.shape == (3,):
            I = np.diag(I)
        if I.shape != (3, 3):
            raise InputError(f"{label}: inertia must be 3 values or a 3x3 matrix")
        if not np.allclose(I, I.T, rtol=1e-9, atol=1e-12 * max(1.0, abs(I).max())):
            raise InputError(f"{label}: inertia matrix must be symmetric")
        I = _place(I, _frame(frame))
    if about is not None:
        # shift from `about` back to the cog
        r = np.asarray(about, float) - cog
        I = I - m * ((r @ r) * np.eye(3) - np.outer(r, r))
    return MassProperties(m, cog, I, label)


def _frame(frame) -> np.ndarray:
    if frame is None:
        return np.eye(3)
    frame = np.asarray(frame, dtype=float)
    if frame.shape != (3, 3) or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-9):
        raise InputError("orientation must be a rotation matrix")
    return frame
