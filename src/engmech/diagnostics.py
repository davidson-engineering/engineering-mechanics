"""Human-readable explanations of mechanisms, redundancy and kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engmech.model import GROUND, BuiltModel
from engmech.solver import ROW_NAMES, System
from engmech.units import UnitSystem


@dataclass
class BodyMotion:
    """How one body moves in a mechanism mode."""

    body: str
    kind: str  # rotation | translation | screw
    direction: np.ndarray  # rotation axis or translation direction
    point: np.ndarray | None = None  # a point on the rotation axis
    pitch: float = 0.0
    point_name: str | None = None

    def describe(self, units: UnitSystem, planar: bool) -> str:
        if self.kind == "translation":
            return f"{self.body} translates along {_dir(self.direction, planar)}"
        where = _pt(self.point, units, planar)
        if self.point_name:
            where = f"{self.point_name} {where}"
        if planar:
            return f"{self.body} rotates about the point {where}"
        axis = _dir(self.direction, planar)
        text = f"{self.body} rotates about an axis along {axis} through {where}"
        if self.kind == "screw":
            text += f" (screw pitch {units.format(self.pitch, 'length')} per rad)"
        return text


def describe_mechanism_mode(system: System, mode: np.ndarray) -> list[BodyMotion]:
    """Interpret a scaled left-null-space vector as a rigid motion of each body."""
    L = system.length
    twists = {}
    for body in system.model.bodies:
        v, w = np.zeros(3), np.zeros(3)
        for i in system.body_rows(body):
            _, c = system.rows[i]
            if c < 3:
                v[c] = mode[i]
            else:
                w[c - 3] = mode[i] / L
        twists[body] = (v, w)
    size = max(np.linalg.norm(v) + np.linalg.norm(w) * L for v, w in twists.values())
    out = []
    for body, (v, w) in twists.items():
        if np.linalg.norm(v) + np.linalg.norm(w) * L < 1e-6 * size:
            continue
        if np.linalg.norm(w) * L < 1e-6 * size:
            out.append(BodyMotion(body, "translation", v / np.linalg.norm(v)))
            continue
        w2 = w @ w
        point = system.ref + np.cross(w, v) / w2
        pitch = (w @ v) / w2
        kind = "screw" if abs(pitch) > 1e-6 * L else "rotation"
        # prefer a positive-leaning axis direction for readability
        axis = w / np.sqrt(w2)
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
            pitch = -pitch
        on_axis, name = _nearest_on_axis(point, axis, system)
        out.append(BodyMotion(body, kind, axis, on_axis, pitch, name))
    return out


def _nearest_on_axis(point, axis, system: System) -> tuple[np.ndarray, str | None]:
    """A readable point on the axis: a model point that lies on it if there is
    one (named points first, then joint locations), else the point closest to
    the model's centre."""
    model = system.model
    candidates = list(model.points.items()) + [
        (None, j.geometry.point_b) for j in model.joints.values()
    ]
    for name, q in candidates:
        off = (q - point) - axis * ((q - point) @ axis)
        if np.linalg.norm(off) < 1e-9 * system.length:
            return np.array(q, dtype=float), name
    c = system.ref
    return point + axis * ((c - point) @ axis), None


def describe_redundancy_mode(system: System, mode: np.ndarray) -> list[tuple[str, str, float]]:
    """Joint components taking part in a self-stress state: (joint, component, weight)."""
    L = system.length
    lam = system.col_scale * mode
    per_joint: dict[str, np.ndarray] = {}
    for u, value in zip(system.unknowns, lam, strict=True):
        w = per_joint.setdefault(u.joint, np.zeros(6))
        if u.component.kind == "force":
            w[:3] += value * u.component.direction
        else:
            w[3:] += value * u.component.direction
    items = []
    for joint, w in per_joint.items():
        scaled = np.concatenate([w[:3], w[3:] / L])
        for c in range(6):
            if abs(scaled[c]) > 1e-8:
                items.append((joint, ROW_NAMES[c], float(scaled[c])))
    top = max((abs(x[2]) for x in items), default=1.0)
    return [(j, c, v / top) for j, c, v in items]


# --------------------------------------------------------------------------- kinematics


def kinematic_warnings(model: BuiltModel) -> list[str]:
    """Check that prescribed motions agree at pins and welds.

    Only joints that hold two points together are checked; sliding joints
    need relative velocities that inverse dynamics does not ask for.
    """
    if not any(b.motion is not None for b in model.bodies.values()):
        return []
    warnings = []
    units = model.output_units
    for joint in model.joints.values():
        if joint.kind != "joint" and joint.kind != "support":
            continue
        geo = joint.geometry
        if not np.allclose(geo.point_a, geo.point_b):
            continue
        forces = np.array(
            [c.direction for c in geo.components if c.kind == "force" and c.role == "constraint"]
        )
        if forces.size == 0:
            continue
        if model.planar:  # only in-plane constraint forces hold the point in the plane
            forces = forces[:, :2]
        needed = 2 if model.planar else 3
        if np.linalg.matrix_rank(forces, tol=1e-9) < needed:
            continue
        p = geo.point_b
        a_a, w_a, al_a = _point_state(model, joint.body_a, p)
        a_b, w_b, al_b = _point_state(model, joint.body_b, p)
        scale = max(np.linalg.norm(a_a), np.linalg.norm(a_b), 1e-9)
        rel = a_b - a_a
        if model.planar:
            rel[2] = 0.0
        if np.linalg.norm(rel) > 1e-6 * scale:
            warnings.append(
                f"{joint.name}: the prescribed motions give the joint point different "
                f"accelerations on {joint.body_a} ({_vec(a_a, units, 'acceleration')}) and "
                f"{joint.body_b} ({_vec(a_b, units, 'acceleration')}); check the motion input"
            )
        if joint.type_name not in ("fixed", "pin"):
            continue
        moments = [
            c.direction for c in geo.components if c.kind == "moment" and c.role == "constraint"
        ]
        if model.planar:
            moments = [d for d in moments if abs(d[2]) > 0.5]
        if not moments:
            continue
        Dm = np.array(moments)
        w_rel = w_b - w_a
        al_rel = al_b - al_a - np.cross(w_a, w_rel)
        wscale = max(np.linalg.norm(w_a), np.linalg.norm(w_b), 1e-9)
        ascale = max(np.linalg.norm(al_a), np.linalg.norm(al_b), wscale**2, 1e-9)
        if (
            np.linalg.norm(Dm @ w_rel) > 1e-6 * wscale
            or np.linalg.norm(Dm @ al_rel) > 1e-6 * ascale
        ):
            warnings.append(
                f"{joint.name}: the angular motions of {joint.body_a} and {joint.body_b} "
                f"are not compatible with a {joint.type_name} joint; check the motion input"
            )
    return warnings


def _point_state(model: BuiltModel, body: str, point: np.ndarray):
    if body == GROUND:
        return np.zeros(3), np.zeros(3), np.zeros(3)
    b = model.bodies[body]
    if b.motion is None:
        return np.zeros(3), np.zeros(3), np.zeros(3)
    m = b.motion
    return m.point_acceleration(b.mass.cog, point), m.angular_velocity, m.angular_acceleration


# --------------------------------------------------------------------------- formatting


def _dir(d: np.ndarray, planar: bool) -> str:
    for i, name in enumerate("xyz"):
        if abs(abs(d[i]) - 1) < 1e-9:
            return f"{'+' if d[i] > 0 else '-'}{name}"
    comps = d[:2] if planar else d
    return "(" + ", ".join(f"{x:.3f}" for x in comps) + ")"


def _pt(p: np.ndarray, units: UnitSystem, planar: bool) -> str:
    comps = p[:2] if planar else p
    return (
        "("
        + ", ".join(units.format(x, "length", unit=False) for x in comps)
        + ") "
        + units.label("length")
    )


def _vec(v: np.ndarray, units: UnitSystem, kind: str) -> str:
    return "(" + ", ".join(units.format(x, kind, unit=False) for x in v) + ") " + units.label(kind)
