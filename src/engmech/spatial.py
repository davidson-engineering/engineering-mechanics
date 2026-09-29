"""Vector helpers: skew matrices, frames, directions and wrench transport."""

from __future__ import annotations

import numpy as np

from engmech.errors import InputError

EX, EY, EZ = np.eye(3)
AXES = {"x": EX, "y": EY, "z": EZ}

# Relative tolerance for geometric tests (parallel, perpendicular, in-plane).
GEOM_TOL = 1e-9


def skew(v) -> np.ndarray:
    """Matrix [v]x such that [v]x @ u == cross(v, u)."""
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def unit(v, what: str = "direction") -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise InputError(f"{what} must be a non-zero vector")
    return v / n


def parse_axis_name(text: str) -> np.ndarray | None:
    """'+x', '-z', 'y' -> unit vector; anything else -> None."""
    t = text.strip().lower().replace(" ", "")
    sign = 1.0
    if t[:1] in "+-" and len(t) == 2:
        sign = -1.0 if t[0] == "-" else 1.0
        t = t[1:]
    if t in AXES:
        return sign * AXES[t]
    return None


def frame_from_axis(axis) -> np.ndarray:
    """Right-handed frame (columns = local x, y, z) with local z along ``axis``.

    The choice of local x is deterministic and planar-friendly: for an axis
    along global z, local x/y are global x/y; for an axis in the xy-plane,
    local y is global z so the local x axis stays in the plane.
    """
    z = unit(axis, "axis")
    if abs(z[2]) > 1 - GEOM_TOL:
        x = EX.copy()
        x = x - z * (x @ z)
    else:
        x = np.cross(EZ, z)
    x = unit(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def frame_from_axes(z=None, x=None, y=None) -> np.ndarray:
    """Frame from any two of the local axes (the second is orthogonalised)."""
    given = {k: v for k, v in (("x", x), ("y", y), ("z", z)) if v is not None}
    if len(given) == 1 and "z" in given:
        return frame_from_axis(z)
    if len(given) != 2:
        raise InputError("a frame needs exactly two of its axes, e.g. {z: ..., x: ...}")
    (a_name, a), (b_name, b) = sorted(given.items(), key=lambda kv: "zxy".index(kv[0]))
    a = unit(a, f"{a_name} axis")
    b = np.asarray(b, dtype=float) - a * (np.asarray(b, dtype=float) @ a)
    if np.linalg.norm(b) < GEOM_TOL:
        raise InputError(f"frame axes {a_name} and {b_name} must not be parallel")
    b = unit(b)
    cols = {a_name: a, b_name: b}
    missing = ({"x", "y", "z"} - set(cols)).pop()
    # right-handed completion: x = y cross z, y = z cross x, z = x cross y
    nxt = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}[missing]
    cols[missing] = np.cross(cols[nxt[0]], cols[nxt[1]])
    return np.column_stack([cols["x"], cols["y"], cols["z"]])


def transport_moment(force, moment, from_point, to_point) -> np.ndarray:
    """Moment of the wrench (force, moment@from_point) about to_point."""
    r = np.asarray(from_point, dtype=float) - np.asarray(to_point, dtype=float)
    return np.asarray(moment, dtype=float) + np.cross(r, force)


def is_parallel(a, b) -> bool:
    a, b = unit(a), unit(b)
    return np.linalg.norm(np.cross(a, b)) < 1e-7


def rotation_about_axis(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    k = unit(axis, "rotation axis")
    K = skew(k)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
