"""Interactive free-body diagrams with plotly.

Planar models are drawn in 2D with engineering support symbols, force
arrows and moment arcs; spatial models in 3D with cones for arrowheads and
double-headed arrows for moments. A dropdown switches between the whole
model and the free-body diagram of each body.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go

from engmech.errors import InputError
from engmech.model import GROUND
from engmech.results import CaseResult, Results
from engmech.solver import ROW_NAMES
from engmech.spatial import frame_from_axis, parse_axis_name
from engmech.units import UnitSystem

PALETTE = {
    "applied": "#c2410c",  # orange-red: applied loads
    "weight": "#4d7c0f",  # olive green: gravity
    "inertia": "#a16207",  # amber: d'Alembert loads
    "reaction": "#0f766e",  # teal: support reactions
    "joint": "#7e22ce",  # purple: forces between bodies
    "solved": "#be185d",  # magenta: solved loads and actuator efforts
    "tension": "#1d4ed8",  # blue: members in tension
    "compression": "#b91c1c",  # red: members in compression
    "support": "#475569",  # slate: support symbols
    "grid": "rgba(100,116,139,0.18)",
}
BODY_COLORS = ["#64748b", "#0369a1", "#9333ea", "#b45309", "#be123c", "#78716c", "#4d7c0f"]
GROUP_NAMES = {
    "applied": "Applied loads",
    "weight": "Weight",
    "inertia": "Inertia (d'Alembert)",
    "reaction": "Support reactions",
    "joint": "Joint forces",
    "solved": "Solved loads / actuators",
}
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# Default 3D view, orthographic as in engineering drawings, looking down from
# the front right: where the camera sits, as components across the page, into
# the page and up (see _camera). With z up: x to the lower right, y into the page.
EYE = np.array([0.95, -1.05, 0.62])
# Projected height of the scene box at the default view, in plotly's aspect
# units: large enough to fill the plot, small enough to keep tick labels in.
FILL = 1.8


@dataclass
class Arrow:
    kind: str  # force | moment
    point: np.ndarray
    vector: np.ndarray
    label: str
    hover: str
    group: str
    body: str
    distributed: dict | None = None
    pull: bool = False  # drawn from the point outwards (members in tension)
    joint: str = ""  # the support or joint this force comes from, if any


@dataclass
class View:
    name: str
    bodies: list[str]
    arrows: list[Arrow] = field(default_factory=list)
    supports: bool = True


# --------------------------------------------------------------------------- arrows


def _vec_text(v, units: UnitSystem, kind: str, planar: bool) -> str:
    comps = ("x", "y", "z")
    idx = (0, 1) if planar and kind == "force" else ((2,) if planar else (0, 1, 2))
    prefix = "F" if kind == "force" else "M"
    return "<br>".join(f"{prefix}{comps[i]} = {units.format(v[i], kind)}" for i in idx)


def _collect(results: Results, case: CaseResult, units: UnitSystem) -> list[View]:
    model = results.model
    planar = model.planar
    loads: list[Arrow] = []
    for w in case.loads:
        group = {"gravity": "weight", "inertia": "inertia"}.get(w.kind, "applied")
        if w.kind == "distributed":
            d = w.detail
            label = (
                units.format(d["w1"], "force_per_length")
                if abs(d["w1"] - d["w2"]) < 1e-12 * max(abs(d["w1"]), abs(d["w2"]), 1e-300)
                else f"{units.format(d['w1'], 'force_per_length', unit=False)} → "
                f"{units.format(d['w2'], 'force_per_length')}"
            )
            hover = (
                f"<b>{w.name}</b> (distributed, on {w.body})<br>"
                f"resultant {units.format(d['resultant'], 'force')}"
            )
            loads.append(Arrow("force", d["start"], w.force, label, hover, group, w.body, d))
            continue
        if np.linalg.norm(w.force) > 0:
            label = units.format(float(np.linalg.norm(w.force)), "force")
            hover = f"<b>{w.name}</b> on {w.body}<br>" + _vec_text(w.force, units, "force", planar)
            loads.append(Arrow("force", w.point, w.force, label, hover, group, w.body))
        if np.linalg.norm(w.moment) > 0:
            m = w.moment
            label = units.format(float(m[2] if planar else np.linalg.norm(m)), "moment")
            hover = f"<b>{w.name}</b> on {w.body}<br>" + _vec_text(m, units, "moment", planar)
            point = w.point
            if w.kind == "moment" and not w.detail.get("placed", True):
                point = model.bodies[w.body].reference_point
            loads.append(Arrow("moment", point, m, label, hover, group, w.body))

    def joint_arrows(j, sign: float, body: str, point, group: str) -> list[Arrow]:
        """Arrows for the determined part of a joint's wrench; indeterminate
        components are left out and named in the hover text."""
        out = []
        det = j.determined
        F = np.where(det[:3], sign * j.force, 0.0)
        M = np.where(det[3:], sign * j.moment, 0.0)
        who = "ground" if j.body_a == GROUND else (j.body_a if sign > 0 else j.body_b)
        title = f"<b>{j.name}</b> ({j.type_name}) on {body} from {who}"
        unknown = [n for n, d, a in zip(ROW_NAMES, det, j.active, strict=True) if a and not d]
        note = f"<br><i>indeterminate: {', '.join(unknown)}</i>" if unknown else ""
        if np.linalg.norm(F) > 0:
            pull = j.scalars.get("T", 0.0) > 0
            hover = title + "<br>" + _vec_text(F, units, "force", planar) + note
            if j.scalars:
                hover += "<br>" + "<br>".join(
                    f"{k} = {units.format(sign * v if k != 'T' else v, j.scalar_kinds[k])}"
                    for k, v in j.scalars.items()
                    if j.scalar_determined[k]
                )
            label = units.format(float(np.linalg.norm(F)), "force")
            if not det[:3].all():
                label += "*"
            out.append(Arrow("force", point, F, label, hover, group, body, pull=pull, joint=j.name))
        if np.linalg.norm(M) > 0:
            label = units.format(float(M[2] if planar else np.linalg.norm(M)), "moment")
            if not det[3:].all():
                label += "*"
            hover = title + "<br>" + _vec_text(M, units, "moment", planar) + note
            out.append(Arrow("moment", point, M, label, hover, group, body, joint=j.name))
        return out

    reactions: dict[str, list[Arrow]] = {}
    for j in case.joints.values():
        group = (
            "reaction" if j.kind == "support" else ("solved" if j.kind == "unknown" else "joint")
        )
        if j.kind == "support" and any(
            c.role == "drive" for c in model.joints[j.name].geometry.components
        ):
            group = "reaction"
        reactions.setdefault(j.body_b, []).extend(joint_arrows(j, 1.0, j.body_b, j.point, group))
        if j.body_a != GROUND:
            reactions.setdefault(j.body_a, []).extend(
                joint_arrows(j, -1.0, j.body_a, j.point_a, group)
            )

    # the whole-model view shows the force in a link or cable by colouring the
    # member, so it leaves out arrows for those, as for joints between bodies
    members = {name for name, *_ in _members(case, results, list(model.bodies))}
    everything = [a for a in loads]
    for arrows in reactions.values():
        everything += [a for a in arrows if a.group != "joint" and a.joint not in members]
    views = [View("Whole model", list(model.bodies), everything)]
    if len(model.bodies) > 1:
        for name in model.bodies:
            arrows = [a for a in loads if a.body == name] + reactions.get(name, [])
            views.append(View(f"Free body: {name}", [name], arrows, supports=True))
    return views


# --------------------------------------------------------------------------- geometry


def _body_points(results: Results, name: str) -> list[np.ndarray]:
    model = results.model
    body = model.bodies[name]
    if body.outline:
        return list(body.outline)
    pts = []
    for j in model.joints.values():
        if j.body_b == name:
            pts.append(j.geometry.point_b)
        if j.body_a == name:
            pts.append(j.geometry.point_a)
    for w in model.loads:
        if w.body != name:
            continue
        if w.kind == "distributed":
            pts += [w.detail["start"], w.detail["end"]]
        elif w.kind == "force" or (w.kind == "moment" and w.detail.get("placed")):
            pts.append(w.point)
    for s in body.shapes:
        pts.append(s.cog)
    if not pts and body.mass is not None:
        pts.append(body.mass.cog)
    return pts


def _skeleton(results: Results, name: str) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Segments a body is drawn as, when it is drawn as lines (None for a filled
    outline, which already contains every point of the body)."""
    body = results.model.bodies[name]
    L = _extent(results)
    if body.outline:
        pts = [np.asarray(p, float) for p in body.outline]
        return list(itertools.pairwise(pts)) or None
    pts = _dedupe(_body_points(results, name), 1e-9 * L)
    if len(pts) < 2:
        return None
    centered = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    if len(pts) >= 3 and sv[1] > 1e-6 * sv[0]:
        return None
    u = np.linalg.svd(centered, full_matrices=False)[2][0]
    t = centered @ u
    return [(pts[np.argmin(t)], pts[np.argmax(t)])]


def _closest_on_segments(p: np.ndarray, segments) -> np.ndarray:
    best, best_d = None, np.inf
    for a, b in segments:
        q = _closest_on_segment(p, a, b)
        d = np.linalg.norm(p - q)
        if d < best_d:
            best, best_d = q, d
    return best


def _closest_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = 0.0 if not ab @ ab else float(np.clip((p - a) @ ab / (ab @ ab), 0, 1))
    return a + t * ab


def _distance_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(p - _closest_on_segment(p, a, b)))


def _away(n: np.ndarray, at: np.ndarray, centre: np.ndarray, tol: float) -> np.ndarray:
    """Of the 2D directions n and -n, the one pointing away from ``centre``, so
    that a label beside a line at ``at`` faces out of the drawing, where there
    is room. Near the centre: the one pointing up, or right."""
    side = float(n @ (np.asarray(at) - np.asarray(centre)))
    if abs(side) > tol:
        return n if side > 0 else -n
    if n[1] < -1e-9 or (abs(n[1]) <= 1e-9 and n[0] < 0):
        return -n
    return n


def _lever_lines(results: Results, arrows: list[Arrow]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dotted lever arms from load points that are off a line-drawn body."""
    L = _extent(results)
    out = []
    cache: dict[str, list | None] = {}
    for a in arrows:
        if a.distributed or a.group not in ("applied", "solved"):
            continue
        if a.body not in cache:
            cache[a.body] = _skeleton(results, a.body)
        segs = cache[a.body]
        if not segs:
            continue
        q = _closest_on_segments(np.asarray(a.point, float), segs)
        if np.linalg.norm(a.point - q) > 0.01 * L:
            out.append((q, np.asarray(a.point, float)))
    return out


def _members(case: CaseResult, results: Results, bodies: list[str]):
    """Two-force members (links, cables) with their axial force, for colouring."""
    out = []
    for j in case.joints.values():
        if "T" not in j.scalars or not j.scalar_determined["T"]:
            continue
        if j.body_b not in bodies and j.body_a not in bodies:
            continue
        geo = results.model.joints[j.name].geometry
        out.append((j.name, geo.point_a, geo.point_b, j.scalars["T"]))
    return out


def _coloured_members(case: CaseResult, results: Results, bodies: list[str]) -> set[str]:
    """The members drawn coloured by their force (those that carry one)."""
    members = _members(case, results, bodies)
    tmax = max((abs(t) for *_, t in members), default=0.0) or 1.0
    return {name for name, *_, t in members if abs(t) > 1e-9 * tmax}


def _dedupe(points: list[np.ndarray], tol: float) -> np.ndarray:
    out: list[np.ndarray] = []
    for p in points:
        if all(np.linalg.norm(p - q) > tol for q in out):
            out.append(np.asarray(p, float))
    return np.array(out) if out else np.zeros((0, 3))


def _hull2d(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of 2D points (returns a closed loop)."""
    P = sorted(map(tuple, pts))
    if len(P) <= 2:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    loop = lower[:-1] + upper[:-1]
    return np.array([*loop, loop[0]])


def _extent(results: Results) -> float:
    pts = results.model.all_points()
    span = np.ptp(pts, axis=0) if len(pts) > 1 else np.zeros(3)
    L = float(np.max(span))
    return L if L > 0 else 1.0


# --------------------------------------------------------------------------- 2D


def _rot(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class _Planar:
    def __init__(self, results: Results, units: UnitSystem, views: list[View]):
        self.results = results
        self.units = units
        self.views = views
        self.L = _extent(results)
        self.f_len = self.units.factor("length")
        forces = [
            np.linalg.norm(a.vector)
            for v in views
            for a in v.arrows
            if a.kind == "force" and a.distributed is None
        ]
        moments = [abs(a.vector[2]) for v in views for a in v.arrows if a.kind == "moment"]
        wmax = [
            max(abs(a.distributed["w1"]), abs(a.distributed["w2"]))
            for v in views
            for a in v.arrows
            if a.distributed
        ]
        self.fmax = max(forces, default=0.0) or 1.0
        self.mmax = max(moments, default=0.0) or 1.0
        self.wmax = max(wmax, default=0.0) or 1.0
        self.glyph = 0.035 * self.L
        # reaction arrows stop short of the support symbol they act through
        self.clearance: dict = {}
        g = self.glyph * self.f_len
        for j in results.model.joints.values():
            if j.kind != "support":
                continue
            depth = {
                "pin": 1.6,
                "revolute": 1.6,
                "hinge": 1.6,
                "ball": 1.6,
                "spherical": 1.6,
                "roller": 2.35,
                "contact": 1.6,
            }.get(j.type_name, 0.0)
            if depth:
                key = ("reaction", tuple(np.round(j.geometry.point_b, 12)))
                self.clearance[key] = depth * g + 0.25 * g

    def xy(self, p) -> tuple[float, float]:
        return float(p[0] * self.f_len), float(p[1] * self.f_len)

    # --- bodies
    def body_traces(self, name: str, color: str) -> list[go.Scatter]:
        pts = _dedupe(_body_points(self.results, name), 1e-9 * self.L)
        if len(pts) == 0:
            return []
        body = self.results.model.bodies[name]
        traces = []
        xy = pts[:, :2] * self.f_len
        explicit = bool(body.outline)
        closed = False
        if explicit:
            # the outline exactly as given (a repeated first point closes it)
            loop = np.array([p[:2] for p in body.outline], dtype=float) * self.f_len
            closed = len(loop) >= 4 and np.allclose(loop[0], loop[-1])
        else:
            loop = xy if len(xy) < 3 else _hull2d(xy)
        area = 0.0
        if len(loop) >= 4:
            x, y = loop[:, 0], loop[:, 1]
            area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        L = self.L * self.f_len
        if body.particle or len(xy) == 1:
            c = xy.mean(axis=0)
            if not body.particle:  # a single-point rigid body: draw a small block
                h = self.glyph * self.f_len
                block = (
                    c
                    + np.array([[-1.4, -0.8], [1.4, -0.8], [1.4, 0.8], [-1.4, 0.8], [-1.4, -0.8]])
                    * h
                )
                traces.append(
                    go.Scatter(
                        x=block[:, 0],
                        y=block[:, 1],
                        mode="lines",
                        fill="toself",
                        fillcolor=_rgba(color, 0.25),
                        line=dict(color=color, width=2.5),
                        name=name,
                        legendgroup=f"body:{name}",
                        hoverinfo="skip",
                    )
                )
                return traces
            traces.append(
                go.Scatter(
                    x=[c[0]],
                    y=[c[1]],
                    mode="markers",
                    name=name,
                    legendgroup="particles",
                    marker=dict(size=11, color="white", line=dict(color=color, width=2.5)),
                    hovertemplate=f"<b>{name}</b> (particle)<extra></extra>",
                    showlegend=False,
                )
            )
            return traces
        if area > 1e-4 * L * L and (closed or not explicit):
            traces.append(
                go.Scatter(
                    x=loop[:, 0],
                    y=loop[:, 1],
                    mode="lines",
                    fill="toself",
                    fillcolor=_rgba(color, 0.14),
                    line=dict(color=color, width=2.5),
                    name=name,
                    legendgroup=f"body:{name}",
                    hoverinfo="skip",
                )
            )
        else:
            if not explicit and len(xy) >= 2:
                # collinear: draw between the two extreme points
                d = xy - xy.mean(axis=0)
                u = np.linalg.svd(d, full_matrices=False)[2][0]
                s = d @ u
                loop = np.array([xy[np.argmin(s)], xy[np.argmax(s)]])
            traces.append(
                go.Scatter(
                    x=loop[:, 0],
                    y=loop[:, 1],
                    mode="lines",
                    line=dict(color=color, width=7),
                    opacity=0.85,
                    name=name,
                    legendgroup=f"body:{name}",
                    hoverinfo="skip",
                )
            )
        if body.mass is not None and body.mass.mass > 0:
            cx, cy = self.xy(body.mass.cog)
            traces.append(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="markers",
                    legendgroup=f"body:{name}",
                    showlegend=False,
                    marker=dict(
                        symbol="circle-cross-open", size=13, color="#0f172a", line=dict(width=1.6)
                    ),
                    hovertemplate=(
                        f"<b>{name}</b> centre of gravity<br>"
                        f"m = {self.units.format(body.mass.mass, 'mass')}<extra></extra>"
                    ),
                )
            )
        return traces

    # --- supports
    def support_traces(self, bodies: list[str], hidden=()) -> list[go.Scatter]:
        """Support symbols, joints, and link and cable lines (except ``hidden``
        ones, drawn coloured as members instead)."""
        model = self.results.model
        g = self.glyph * self.f_len
        segs: list[np.ndarray] = []
        fills: list[np.ndarray] = []
        hover_pts, hover_text = [], []
        joints_open, joints_solid = [], []
        for j in model.joints.values():
            if j.kind == "unknown":
                continue
            if j.body_b not in bodies and j.body_a not in bodies:
                continue
            p = np.array(self.xy(j.geometry.point_b))
            geo = j.geometry
            if j.kind == "joint":
                target = joints_solid if j.type_name == "fixed" else joints_open
                if j.type_name in ("link", "cable", "strut"):
                    pa = np.array(self.xy(geo.point_a))
                    if j.name not in hidden:
                        segs.append(np.array([pa, p]))
                    joints_open += [pa, p]
                else:
                    target.append(p)
                hover_pts.append(p)
                hover_text.append(f"<b>{j.name}</b>: {j.type_name} joint ({j.body_a} – {j.body_b})")
                continue
            hover_pts.append(p)
            hover_text.append(f"<b>{j.name}</b>: {j.type_name} support on {j.body_b}")
            up = self._support_up(j)
            side = _rot(up, np.pi / 2)
            t = j.type_name
            if t in (
                "pin",
                "ball",
                "bearing",
                "revolute",
                "hinge",
                "spherical",
                "roller",
                "contact",
            ):
                base = p - up * g * 1.6
                tri = np.array([p, base + side * g, base - side * g, p])
                fills.append(tri)
                ground = base
                if t == "roller":
                    r = g * 0.32
                    for k in (-0.55, 0.55):
                        c = base + side * g * k - up * r
                        fills.append(_circle(c, r))
                    ground = base - up * 2 * r
                segs.append(self._ground(ground, up, side, g))
                segs.append(self._hatch(ground, up, side, g * 1.3))
                if t in ("pin", "roller", "revolute", "hinge"):
                    joints_open.append(p)
            elif t in ("fixed", "weld", "custom"):
                segs.append(np.array([p + side * g * 1.4, p - side * g * 1.4]))
                segs.append(self._hatch(p, up, side, g * 1.4))
            elif t in ("slider", "prismatic", "cylindrical"):
                axis = np.array(geo.frame[:2, 2])
                n = _rot(axis, np.pi / 2)
                box = [
                    p + axis * g * 1.2 + n * g * 0.6,
                    p - axis * g * 1.2 + n * g * 0.6,
                    p - axis * g * 1.2 - n * g * 0.6,
                    p + axis * g * 1.2 - n * g * 0.6,
                ]
                fills.append(np.array([*box, box[0]]))
                for sgn in (1, -1):
                    segs.append(
                        np.array(
                            [
                                p - axis * g * 3 + sgn * n * g * 0.75,
                                p + axis * g * 3 + sgn * n * g * 0.75,
                            ]
                        )
                    )
            elif t in ("link", "strut", "cable"):
                a = np.array(self.xy(geo.point_a))
                segs.append(np.array([a, p]))
                d = (a - p) / max(np.linalg.norm(a - p), 1e-300)
                segs.append(self._ground(a, -d, _rot(-d, np.pi / 2), g * 0.8))
                segs.append(self._hatch(a, -d, _rot(-d, np.pi / 2), g * 0.8))
                joints_open += [a, p]
            else:
                joints_solid.append(p)
        traces = []
        if segs:
            x, y = _join(segs)
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=PALETTE["support"], width=1.8),
                    name="Supports",
                    legendgroup="supports",
                    hoverinfo="skip",
                )
            )
        if fills:
            x, y = _join(fills)
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(148,163,184,0.35)",
                    line=dict(color=PALETTE["support"], width=1.6),
                    legendgroup="supports",
                    showlegend=not segs,
                    name="Supports",
                    hoverinfo="skip",
                )
            )
        for pts, solid in ((joints_open, False), (joints_solid, True)):
            if pts:
                arr = np.array(pts)
                traces.append(
                    go.Scatter(
                        x=arr[:, 0],
                        y=arr[:, 1],
                        mode="markers",
                        legendgroup="supports",
                        showlegend=False,
                        hoverinfo="skip",
                        marker=dict(
                            symbol="square" if solid else "circle",
                            size=9 if solid else 8,
                            color=PALETTE["support"] if solid else "white",
                            line=dict(color=PALETTE["support"], width=2),
                        ),
                    )
                )
        if hover_pts:
            arr = np.array(hover_pts)
            traces.append(
                go.Scatter(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    mode="markers",
                    legendgroup="supports",
                    showlegend=False,
                    marker=dict(size=18, opacity=0),
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        return traces

    def _support_up(self, j) -> np.ndarray:
        geo = j.geometry
        if j.type_name in ("roller", "contact"):
            n = geo.components[0].direction[:2]
            return n / np.linalg.norm(n)
        # point from the support toward the body it holds
        body_pts = _body_points(self.results, j.body_b)
        p = geo.point_b[:2]
        c = np.mean([q[:2] for q in body_pts], axis=0) if body_pts else p
        d = c - p
        if j.type_name in ("fixed", "weld", "custom") and np.linalg.norm(d) > 1e-9 * self.L:
            return d / np.linalg.norm(d)
        return np.array([0.0, 1.0])

    def _ground(self, center, up, side, g) -> np.ndarray:
        return np.array([center + side * g * 1.5, center - side * g * 1.5])

    def _hatch(self, center, up, side, half) -> np.ndarray:
        """Short diagonal strokes under a ground line of half-width ``half``."""
        segs = []
        g = self.glyph * self.f_len
        n = max(4, round(2 * half / (0.55 * g)))
        for i in range(n):
            s = center + side * (half - 2 * half * (i + 0.5) / n)
            segs += [s, s - up * 0.42 * g + side * 0.32 * g, np.full(2, np.nan)]
        return np.array(segs)

    def member_traces(self, case: CaseResult, bodies: list[str]) -> list[go.Scatter]:
        members = _members(case, self.results, bodies)
        if not members:
            return []
        tmax = max(abs(t) for *_, t in members) or 1.0
        centre = np.mean([self.xy(p) for _, a, b, _ in members for p in (a, b)], axis=0)
        traces = []
        for sign, key in ((1, "tension"), (-1, "compression")):
            chosen = [m for m in members if np.sign(m[3]) == sign and abs(m[3]) > 1e-9 * tmax]
            if not chosen:
                continue
            x, y, tx, ty, text, hover, where = [], [], [], [], [], [], []
            for name, a, b, t in chosen:
                (ax_, ay_), (bx, by) = self.xy(a), self.xy(b)
                x += [ax_, bx, None]
                y += [ay_, by, None]
                label = f"{self.units.format(abs(t), 'force')} {'T' if t > 0 else 'C'}"
                hover += [f"<b>{name}</b>: {label}"] * 2 + [""]
                # label beside the member, on the side facing out of the drawing,
                # anchored so the text grows away from the line
                d = np.array([bx - ax_, by - ay_])
                n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-300)
                mid = np.array([(ax_ + bx) / 2, (ay_ + by) / 2])
                n = _away(n, mid, centre, 0.05 * self.L * self.f_len)
                off = n * self.glyph * 0.35 * self.f_len
                tx.append(mid[0] + off[0])
                ty.append(mid[1] + off[1])
                text.append(label)
                where.append(_text_position(*n))
            color = PALETTE[key]
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=color, width=3.2),
                    opacity=0.85,
                    name=f"Members in {key}",
                    legendgroup=key,
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            traces.append(
                go.Scatter(
                    x=tx,
                    y=ty,
                    mode="text",
                    text=text,
                    textposition=where,
                    legendgroup=key,
                    showlegend=False,
                    textfont=dict(color=color, size=11, family=FONT),
                    hoverinfo="skip",
                )
            )
        return traces

    def lever_traces(self, arrows: list[Arrow]) -> list[go.Scatter]:
        lines = _lever_lines(self.results, arrows)
        if not lines:
            return []
        x, y = [], []
        for a, b in lines:
            x += [a[0] * self.f_len, b[0] * self.f_len, None]
            y += [a[1] * self.f_len, b[1] * self.f_len, None]
        return [
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="#94a3b8", width=1.4, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        ]

    # --- arrows
    def _segments(self, bodies: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
        """The lines drawn in a view: bodies drawn as lines, and links and
        cables (in plot units)."""
        segments = []
        for name in bodies:
            for a, b in _skeleton(self.results, name) or []:
                segments.append((np.array(self.xy(a)), np.array(self.xy(b))))
        for j in self.results.model.joints.values():
            if (
                j.type_name in ("link", "strut", "cable")
                and bodies
                and (j.body_a in bodies or j.body_b in bodies)
            ):
                a, b = j.geometry.point_a, j.geometry.point_b
                segments.append((np.array(self.xy(a)), np.array(self.xy(b))))
        return segments

    def arrow_traces(self, arrows: list[Arrow], bodies: list[str] = ()) -> list[go.Scatter]:
        traces = []
        by_group: dict[str, list[Arrow]] = {}
        for a in arrows:
            by_group.setdefault(a.group, []).append(a)
        segments = self._segments(list(bodies))
        clear = self.glyph * 0.8 * self.f_len
        drawn = [p for s in segments for p in s] + [np.array(self.xy(a.point)) for a in arrows]
        centre = np.mean(drawn, axis=0) if drawn else np.zeros(2)
        for group, items in by_group.items():
            color = PALETTE[group]
            mx, my, sizes, htext = [], [], [], []
            tx, ty, ttext = [], [], []
            beside: dict[int, str] = {}  # labels moved beside their arrow: textposition
            for a in items:
                if a.distributed:
                    others = [self.xy(b.point) for b in arrows if b.distributed is None]
                    self._distributed(a, mx, my, sizes, tx, ty, ttext, htext, others)
                    continue
                if a.kind == "force":
                    F = a.vector[:2]
                    mag = np.linalg.norm(F)
                    if mag < 1e-12 * self.fmax:
                        continue
                    length = self.L * (0.07 + 0.11 * mag / self.fmax) * self.f_len
                    u = F / mag
                    if a.pull:
                        tail = np.array(self.xy(a.point))
                        tip = tail + u * length
                    else:
                        tip = np.array(self.xy(a.point))
                        key = (a.group, tuple(np.round(a.point, 12)))
                        tip = tip - u * self.clearance.get(key, 0.0)
                        tail = tip - u * length
                    mx += [tail[0], tip[0], None]
                    my += [tail[1], tip[1], None]
                    sizes += [0, 13, 0]
                    htext += [a.hover, a.hover, ""]
                    # the label goes beyond the free end of the arrow, or beside
                    # it where that would put it on a drawn line (an arrow along
                    # a link, as in a truss joint's free body)
                    far = tip if a.pull else tail
                    pad = far + (u if a.pull else -u) * self.glyph * 0.9 * self.f_len
                    if any(_distance_to_segment(pad, p, q) < clear for p, q in segments):
                        n = _away(np.array([-u[1], u[0]]), far, centre, clear)
                        pad = far + n * self.glyph * 0.3 * self.f_len
                        beside[len(ttext)] = _text_position(*n)
                    tx.append(pad[0])
                    ty.append(pad[1])
                    ttext.append(a.label)
                else:
                    Mz = a.vector[2]
                    if abs(Mz) < 1e-12 * self.mmax:
                        continue
                    r = self.L * (0.045 + 0.035 * abs(Mz) / self.mmax) * self.f_len
                    c = np.array(self.xy(a.point))
                    start, sweep = np.deg2rad(-60), np.deg2rad(260) * np.sign(Mz)
                    ts = start + np.linspace(0, sweep, 40)
                    arc = c + r * np.column_stack([np.cos(ts), np.sin(ts)])
                    mx += [*arc[:, 0], None]
                    my += [*arc[:, 1], None]
                    sizes += [0] * 39 + [13, 0]
                    htext += [a.hover] * 40 + [""]
                    # label in the gap of the arc (below it), clear of loads drawn
                    # above, growing sideways: a vertical force at the same point
                    # (a fixed support's reaction) runs through the gap too
                    gap = start - np.deg2rad(50) * np.sign(Mz)
                    d = np.array([np.cos(gap), np.sin(gap)])
                    lab = c + r * 1.3 * d
                    vertical = "bottom" if d[1] < -0.38 else ("top" if d[1] > 0.38 else "middle")
                    beside[len(ttext)] = f"{vertical} {'left' if d[0] < 0 else 'right'}"
                    tx.append(lab[0])
                    ty.append(lab[1])
                    ttext.append(a.label)
            if not mx:
                continue
            traces.append(
                go.Scatter(
                    x=mx,
                    y=my,
                    mode="lines+markers",
                    name=GROUP_NAMES[group],
                    legendgroup=group,
                    line=dict(color=color, width=2.6),
                    marker=dict(
                        symbol="arrow",
                        angleref="previous",
                        size=sizes,
                        color=color,
                        line=dict(width=0),
                    ),
                    text=htext,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            traces.append(
                go.Scatter(
                    x=tx,
                    y=ty,
                    mode="text",
                    text=ttext,
                    textposition=[beside.get(i, "middle center") for i in range(len(ttext))],
                    cliponaxis=False,  # a label near the edge may run into the margin
                    legendgroup=group,
                    showlegend=False,
                    textfont=dict(color=color, size=12, family=FONT),
                    hoverinfo="skip",
                )
            )
        return traces

    def _distributed(self, a, mx, my, sizes, tx, ty, ttext, htext, others=()):
        d = a.distributed
        start, end = np.array(self.xy(d["start"])), np.array(self.xy(d["end"]))
        direction = d["direction"][:2]
        span = np.linalg.norm(end - start)
        n = int(np.clip(round(span / (self.L * self.f_len) * 14), 4, 16))
        tails = []
        for i in range(n + 1):
            s = i / n
            w = d["w1"] + (d["w2"] - d["w1"]) * s
            base = start + (end - start) * s
            length = self.L * 0.10 * abs(w) / self.wmax * self.f_len
            tail = base - np.sign(w) * direction * length
            tails.append(tail)
            if abs(w) < 1e-12 * self.wmax:
                continue
            mx += [tail[0], base[0], None]
            my += [tail[1], base[1], None]
            sizes += [0, 9, 0]
            htext += [a.hover, a.hover, ""]
        tails = np.array(tails)
        mx += [*tails[:, 0], None]
        my += [*tails[:, 1], None]
        sizes += [0] * len(tails) + [0]
        htext += [a.hover] * len(tails) + [""]
        # the label goes over the middle of the load, or the nearest point
        # along it that no other load acts at (self-weight often acts mid-span)
        along = (end - start) / max(span, 1e-300)
        taken = []
        for q in others:
            v = np.asarray(q) - start
            if abs(along[0] * v[1] - along[1] * v[0]) < 2 * self.glyph * self.f_len:
                taken.append(v @ along / max(span, 1e-300))
        order = sorted(range(n + 1), key=lambda i: abs(i - n / 2))
        free = [i for i in order if all(abs(i / n - t) > 0.12 for t in taken)]
        mid = tails[free[0] if free else n // 2]
        away = -direction * self.glyph * 1.1 * self.f_len * (1 if d["w1"] + d["w2"] >= 0 else -1)
        tx.append(mid[0] + away[0])
        ty.append(mid[1] + away[1])
        ttext.append(a.label)


def _circle(c, r, n=24) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])


def _join(parts: list[np.ndarray]) -> tuple[list, list]:
    x, y = [], []
    for p in parts:
        p = np.asarray(p)
        x += [None if np.isnan(v) else float(v) for v in p[:, 0]] + [None]
        y += [None if np.isnan(v) else float(v) for v in p[:, 1]] + [None]
    return x, y


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- 3D


class _Spatial:
    def __init__(self, results: Results, units: UnitSystem, views: list[View], up: np.ndarray):
        self.results = results
        self.units = units
        self.views = views
        self.L = _extent(results)
        self.f_len = units.factor("length")
        forces = [
            np.linalg.norm(a.vector)
            for v in views
            for a in v.arrows
            if a.kind == "force" and a.distributed is None
        ]
        moments = [np.linalg.norm(a.vector) for v in views for a in v.arrows if a.kind == "moment"]
        wmax = [
            max(abs(a.distributed["w1"]), abs(a.distributed["w2"]))
            for v in views
            for a in v.arrows
            if a.distributed
        ]
        self.fmax = max(forces, default=0.0) or 1.0
        self.mmax = max(moments, default=0.0) or 1.0
        self.wmax = max(wmax, default=0.0) or 1.0
        self.head = 0.045 * self.L * self.f_len  # arrowhead length
        self.right, self.up = _screen_axes(*_camera(up))

    def p(self, v) -> np.ndarray:
        return np.asarray(v, float) * self.f_len

    def _side(self, away) -> str:
        """The textposition that puts a label on the side ``away`` (a
        direction in model axes) of its point in the default view, so that
        it grows away from the line it labels."""
        away = np.asarray(away, float)
        dx, dy = away @ self.right, away @ self.up
        if np.hypot(dx, dy) <= 1e-9 * max(np.linalg.norm(away), 1e-300):
            return "top center"  # pointing at the viewer
        return _text_position(dx, dy)

    def _beside(self, a, b, centre) -> np.ndarray:
        """The side of segment ab its label goes on, as seen on screen: facing
        away from ``centre`` (out of the drawing), else above it, or right of
        it when it is vertical on screen."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        normal = np.array([-(d @ self.up), d @ self.right])
        if not normal.any():
            return self.up
        normal /= np.linalg.norm(normal)
        mid = np.array([(a + b) / 2 @ self.right, (a + b) / 2 @ self.up])
        at_centre = np.array([centre @ self.right, centre @ self.up])
        nx, ny = _away(normal, mid, at_centre, 0.05 * self.L * self.f_len)
        return nx * self.right + ny * self.up

    def body_traces(self, name: str, color: str) -> list:
        body = self.results.model.bodies[name]
        pts = _dedupe(_body_points(self.results, name), 1e-9 * self.L)
        traces = []
        if body.outline:
            arr = self.p(np.array(body.outline))
            traces.append(
                go.Scatter3d(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    z=arr[:, 2],
                    mode="lines",
                    name=name,
                    legendgroup=f"body:{name}",
                    line=dict(color=color, width=9),
                    hoverinfo="skip",
                )
            )
        elif len(pts) >= 2:
            arr = self.p(pts)
            centered = arr - arr.mean(axis=0)
            sv = np.linalg.svd(centered, compute_uv=False)
            if len(pts) >= 4 and sv[2] > 1e-6 * sv[0]:
                from scipy.spatial import ConvexHull

                hull = ConvexHull(arr)
                i, j, k = hull.simplices.T
                traces.append(
                    go.Mesh3d(
                        x=arr[:, 0],
                        y=arr[:, 1],
                        z=arr[:, 2],
                        i=i,
                        j=j,
                        k=k,
                        color=color,
                        opacity=0.22,
                        name=name,
                        legendgroup=f"body:{name}",
                        showlegend=True,
                        hoverinfo="skip",
                        flatshading=True,
                    )
                )
                edges = set()
                for tri in hull.simplices:
                    for a, b in ((0, 1), (1, 2), (0, 2)):
                        edges.add(tuple(sorted((tri[a], tri[b]))))
                ex, ey, ez = [], [], []
                for a, b in edges:
                    ex += [arr[a, 0], arr[b, 0], None]
                    ey += [arr[a, 1], arr[b, 1], None]
                    ez += [arr[a, 2], arr[b, 2], None]
                traces.append(
                    go.Scatter3d(
                        x=ex,
                        y=ey,
                        z=ez,
                        mode="lines",
                        legendgroup=f"body:{name}",
                        showlegend=False,
                        line=dict(color=color, width=2),
                        hoverinfo="skip",
                        opacity=0.5,
                    )
                )
            elif len(pts) >= 3 and sv[1] > 1e-6 * sv[0]:
                basis = np.linalg.svd(centered, full_matrices=False)[2][:2]
                loop = _hull2d(centered @ basis.T)
                ring = loop @ basis + arr.mean(axis=0)
                n = len(ring) - 1
                traces.append(
                    go.Mesh3d(
                        x=ring[:n, 0],
                        y=ring[:n, 1],
                        z=ring[:n, 2],
                        i=[0] * (n - 2),
                        j=list(range(1, n - 1)),
                        k=list(range(2, n)),
                        color=color,
                        opacity=0.25,
                        name=name,
                        legendgroup=f"body:{name}",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )
                traces.append(
                    go.Scatter3d(
                        x=ring[:, 0],
                        y=ring[:, 1],
                        z=ring[:, 2],
                        mode="lines",
                        legendgroup=f"body:{name}",
                        showlegend=False,
                        line=dict(color=color, width=4),
                        hoverinfo="skip",
                    )
                )
            else:
                u = np.linalg.svd(centered, full_matrices=False)[2][0]
                s = centered @ u
                seg = arr[[np.argmin(s), np.argmax(s)]]
                traces.append(
                    go.Scatter3d(
                        x=seg[:, 0],
                        y=seg[:, 1],
                        z=seg[:, 2],
                        mode="lines",
                        name=name,
                        legendgroup=f"body:{name}",
                        line=dict(color=color, width=10),
                        hoverinfo="skip",
                    )
                )
        for s in body.shapes:
            traces += self._shape_mesh(s, color, name)
        if body.mass is not None and body.mass.mass > 0:
            c = self.p(body.mass.cog)
            traces.append(
                go.Scatter3d(
                    x=[c[0]],
                    y=[c[1]],
                    z=[c[2]],
                    mode="markers",
                    legendgroup=f"body:{name}",
                    showlegend=not traces,
                    name=name,
                    marker=dict(symbol="diamond", size=5, color="#0f172a"),
                    hovertemplate=(
                        f"<b>{name}</b> centre of gravity<br>"
                        f"m = {self.units.format(body.mass.mass, 'mass')}<extra></extra>"
                    ),
                )
            )
        return traces

    def _shape_mesh(self, s, color, name) -> list:
        # draw shapes as translucent ellipsoids of equivalent inertia (uniform look for all)
        if s.mass <= 0:
            return []
        moments, axes = np.linalg.eigh(s.inertia)
        if moments.max() <= 0:
            return []
        a2 = 5 / (2 * s.mass) * (moments.sum() / 2 - moments)
        radii = np.sqrt(np.clip(a2, (0.01 * self.L) ** 2, None))
        u, v = np.mgrid[0 : 2 * np.pi : 18j, 0 : np.pi : 10j]
        sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], axis=-1)
        pts = (sphere * radii) @ axes.T + s.cog
        pts = pts.reshape(-1, 3) * self.f_len
        return [
            go.Mesh3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                alphahull=0,
                color=color,
                opacity=0.12,
                legendgroup=f"body:{name}",
                showlegend=False,
                hoverinfo="skip",
            )
        ]

    def support_traces(self, bodies: list[str], hidden=()) -> list:
        """Support markers, and link and cable lines (except ``hidden`` ones,
        drawn coloured as members instead: two lines in one place flicker)."""
        model = self.results.model
        symbols = {
            "fixed": "square",
            "weld": "square",
            "pin": "circle-open",
            "revolute": "circle-open",
            "hinge": "circle-open",
            "ball": "circle",
            "spherical": "circle",
            "roller": "diamond-open",
            "contact": "diamond",
            "slider": "square-open",
            "prismatic": "square-open",
            "cylindrical": "square-open",
            "bearing": "square-open",
            "universal": "cross",
            "custom": "diamond-open",
            "link": "circle-open",
            "strut": "circle-open",
            "cable": "circle-open",
        }
        traces = []
        lx, ly, lz = [], [], []
        labels: dict[tuple, list[str]] = {}
        for j in model.joints.values():
            if j.kind == "unknown" or (j.body_b not in bodies and j.body_a not in bodies):
                continue
            geo = j.geometry
            p = self.p(geo.point_b)
            if j.type_name in ("link", "strut", "cable") and j.name not in hidden:
                a = self.p(geo.point_a)
                lx += [a[0], p[0], None]
                ly += [a[1], p[1], None]
                lz += [a[2], p[2], None]
            where = (
                "support on " + j.body_b
                if j.kind == "support"
                else (f"joint {j.body_a} – {j.body_b}")
            )
            key = tuple(np.round(p, 9))
            labels.setdefault(key, []).append(j.name)
            traces.append(
                go.Scatter3d(
                    x=[p[0]],
                    y=[p[1]],
                    z=[p[2]],
                    mode="markers",
                    marker=dict(
                        symbol=symbols.get(j.type_name, "circle"),
                        size=7,
                        color=PALETTE["support"],
                        line=dict(color=PALETTE["support"], width=2),
                    ),
                    legendgroup="supports",
                    showlegend=False,
                    hovertemplate=f"<b>{j.name}</b>: {j.type_name} {where}<extra></extra>",
                )
            )
        if lx:
            traces.append(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    legendgroup="supports",
                    showlegend=False,
                    line=dict(color=PALETTE["support"], width=3, dash="dash"),
                    hoverinfo="skip",
                )
            )
        if labels:
            pts = np.array(list(labels))
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="text",
                    text=[", ".join(names) for names in labels.values()],
                    textposition="bottom center",
                    textfont=dict(size=11, color=PALETTE["support"], family=FONT),
                    legendgroup="supports",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if traces:
            traces[0].showlegend = True
            traces[0].name = "Supports & joints"
        return traces

    def arrow_traces(self, arrows: list[Arrow], bodies: list[str] = ()) -> list:
        by_group: dict[str, list[Arrow]] = {}
        for a in arrows:
            by_group.setdefault(a.group, []).append(a)
        traces = []
        for group, items in by_group.items():
            color = PALETTE[group]
            lx, ly, lz, htext = [], [], [], []
            cones = []
            labels = []
            for a in items:
                if a.distributed:
                    d = a.distributed
                    s0, s1 = self.p(d["start"]), self.p(d["end"])
                    n = 8
                    tails = []
                    for i in range(n + 1):
                        s = i / n
                        w = d["w1"] + (d["w2"] - d["w1"]) * s
                        base = s0 + (s1 - s0) * s
                        length = self.L * 0.12 * abs(w) / self.wmax * self.f_len
                        tail = base - np.sign(w) * d["direction"] * length
                        tails.append(tail)
                        if abs(w) > 1e-12 * self.wmax:
                            lx += [tail[0], base[0], None]
                            ly += [tail[1], base[1], None]
                            lz += [tail[2], base[2], None]
                            htext += [a.hover] * 2 + [""]
                            cones.append((base, np.sign(w) * d["direction"], 0.6))
                    tails = np.array(tails)
                    lx += [*tails[:, 0], None]
                    ly += [*tails[:, 1], None]
                    lz += [*tails[:, 2], None]
                    htext += [a.hover] * len(tails) + [""]
                    away = -np.sign(d["w1"] + d["w2"] or 1.0) * np.asarray(d["direction"], float)
                    labels.append((tails[len(tails) // 2], a.label, self._side(away)))
                    continue
                vec = a.vector
                mag = np.linalg.norm(vec)
                vmax = self.fmax if a.kind == "force" else self.mmax
                if mag < 1e-12 * vmax:
                    continue
                u = vec / mag
                length = self.L * (0.10 + 0.14 * mag / vmax) * self.f_len
                if a.pull:
                    tail = self.p(a.point)
                    tip = tail + u * length
                else:
                    tip = self.p(a.point)
                    tail = tip - u * length
                lx += [tail[0], tip[0], None]
                ly += [tail[1], tip[1], None]
                lz += [tail[2], tip[2], None]
                htext += [a.hover] * 2 + [""]
                cones.append((tip, u, 1.0))
                if a.kind == "moment":  # double-headed: right-hand rule vector
                    cones.append((tip - u * self.head * 0.9, u, 1.0))
                # beyond the free end of the arrow, growing away from it
                labels.append((tip if a.pull else tail, a.label, self._side(u if a.pull else -u)))
            if not lx:
                continue
            traces.append(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    name=GROUP_NAMES[group],
                    legendgroup=group,
                    line=dict(color=color, width=5),
                    text=htext,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            if cones:
                traces.append(self._heads(cones, color, group))
            traces += _text_traces_3d(labels, color, group)
        return traces

    def member_traces(self, case: CaseResult, bodies: list[str]) -> list:
        members = _members(case, self.results, bodies)
        coloured = _coloured_members(case, self.results, bodies)
        centre = self.p(self.results.model.all_points().mean(axis=0))
        traces = []
        for sign, key in ((1, "tension"), (-1, "compression")):
            chosen = [m for m in members if np.sign(m[3]) == sign and m[0] in coloured]
            if not chosen:
                continue
            x, y, z, labels = [], [], [], []
            for _, a, b, t in chosen:
                a, b = self.p(a), self.p(b)
                x += [a[0], b[0], None]
                y += [a[1], b[1], None]
                z += [a[2], b[2], None]
                text = f"{self.units.format(abs(t), 'force')} {'T' if t > 0 else 'C'}"
                side = self._side(self._beside(a, b, centre))
                labels.append(((a + b) / 2, text, side))
            color = PALETTE[key]
            traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(color=color, width=6),
                    name=f"Members in {key}",
                    legendgroup=key,
                    hoverinfo="skip",
                )
            )
            traces += _text_traces_3d(labels, color, key)
        return traces

    def lever_traces(self, arrows: list[Arrow]) -> list:
        lines = _lever_lines(self.results, arrows)
        if not lines:
            return []
        x, y, z = [], [], []
        for a, b in lines:
            a, b = self.p(a), self.p(b)
            x += [a[0], b[0], None]
            y += [a[1], b[1], None]
            z += [a[2], b[2], None]
        return [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color="#94a3b8", width=3, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        ]

    def _heads(self, cones, color: str, group: str) -> go.Mesh3d:
        """Arrowheads as one mesh of cones (tip, direction, relative size)."""
        xs, ys, zs, ii, jj, kk = [], [], [], [], [], []
        n = 14
        for tip, u, scale in cones:
            h = self.head * scale
            base = tip - u * h
            frame = frame_from_axis(u)
            ring = [
                base + 0.38 * h * (np.cos(t) * frame[:, 0] + np.sin(t) * frame[:, 1])
                for t in np.linspace(0, 2 * np.pi, n, endpoint=False)
            ]
            k0 = len(xs)
            for p in [tip, base, *ring]:
                xs.append(p[0])
                ys.append(p[1])
                zs.append(p[2])
            for m in range(n):
                a, b = k0 + 2 + m, k0 + 2 + (m + 1) % n
                ii += [k0, k0 + 1]
                jj += [a, b]
                kk += [b, a]
        return go.Mesh3d(
            x=xs,
            y=ys,
            z=zs,
            i=ii,
            j=jj,
            k=kk,
            color=color,
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=0.8, diffuse=0.4, specular=0.1),
            legendgroup=group,
            showlegend=False,
            hoverinfo="skip",
        )


def model_figure(
    results: Results, case: str | None = None, units=None, height: int = 620, up=None
) -> go.Figure:
    """The free-body diagram. ``up`` is the axis that points up in a 3D view
    ('x', 'y', 'z', or signed like '-y'); by default the model file's
    report.up, else z. Planar models are always drawn in the xy-plane."""
    units = results.model.output_units if units is None else UnitSystem.from_spec(units)
    case_result = results.primary if case is None else results[case]
    views = _collect(results, case_result, units)
    planar = results.model.planar
    if planar:
        painter = _Planar(results, units, views)
    else:
        view_up = up_axis(results.model.report_up if up is None else up)
        painter = _Spatial(results, units, views, view_up)
    body_color = {n: BODY_COLORS[i % len(BODY_COLORS)] for i, n in enumerate(results.model.bodies)}

    fig = go.Figure()
    visibility: list[list[bool]] = [[] for _ in views]

    def add(traces, in_views: set[int]):
        for tr in traces:
            fig.add_trace(tr)
            for k in range(len(views)):
                visibility[k].append(k in in_views)

    single = len(results.model.bodies) == 1
    for name in results.model.bodies:
        in_views = {k for k, v in enumerate(views) if name in v.bodies}
        traces = painter.body_traces(name, body_color[name])
        if single:
            for tr in traces:
                tr.showlegend = False
        add(traces, in_views)
    coloured = _coloured_members(case_result, results, views[0].bodies)
    for k, view in enumerate(views):
        add(painter.support_traces(view.bodies, coloured if k == 0 else ()), {k})
    add(painter.member_traces(case_result, views[0].bodies), {0})
    for k, view in enumerate(views):
        add(painter.lever_traces(view.arrows), {k})
    for k, view in enumerate(views):
        add(painter.arrow_traces(view.arrows, view.bodies), {k})

    for i, tr in enumerate(fig.data):
        tr.visible = visibility[0][i]

    L = units.label("length")
    layout = dict(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        font=dict(family=FONT, size=13, color="#0f172a"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,  # clear of the plot's frame
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12),
        ),
    )
    if planar:
        layout.update(
            xaxis=dict(
                title=dict(text=f"x ({L})"),
                gridcolor=PALETTE["grid"],
                zeroline=False,
                showline=True,
                linecolor="#cbd5e1",
                mirror=True,
                ticks="outside",
                tickcolor="#cbd5e1",
            ),
            yaxis=dict(
                title=dict(text=f"y ({L})"),
                gridcolor=PALETTE["grid"],
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
                showline=True,
                linecolor="#cbd5e1",
                mirror=True,
                ticks="outside",
                tickcolor="#cbd5e1",
            ),
        )
    else:
        right, up = _screen_axes(*_camera(view_up))  # the screen's axes, in model axes
        data_lo, data_hi = _scene_bounds(fig)
        labels = _label_extents(fig)
        plot_height = height - layout["margin"]["t"] - layout["margin"]["b"]
        fit = SceneFit(data_lo, data_hi, labels, right, up)
        lo, hi, ratio = fit.solve(PLOT_ASPECT, plot_height)
        # Everything so far is in model axes. Plotly only turns a 3D view about
        # its own z axis (its turntable ignores any other camera up), so each
        # model axis is drawn on the plotly axis that runs the same way on
        # screen: plotly's z carries the up axis, and an axis that runs the
        # other way is drawn reversed (x, z, y with y up, z reversed).
        shown, signs = _plotly_axes(view_up)
        for tr in fig.data:
            if tr.type in ("scatter3d", "mesh3d") and tr.x is not None:
                xyz = (tr.x, tr.y, tr.z)
                tr.x, tr.y, tr.z = (xyz[a] for a in shown)
        base = dict(
            backgroundcolor="white", gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showspikes=False
        )
        axes = {
            name: dict(title=f"{'xyz'[a]} ({L})", **base)
            for name, a in zip(("xaxis", "yaxis", "zaxis"), shown, strict=True)
        }
        layout.update(
            scene=dict(
                **axes,
                aspectmode="manual",
                camera=dict(
                    eye=dict(x=EYE[0], y=EYE[1], z=EYE[2]),
                    up=dict(x=0, y=0, z=1),
                    projection=dict(type="orthographic"),
                ),
            ),
            # what the HTML report needs to fit the scene to other plot sizes
            meta={"engmech_fit": fit.to_json(shown, signs)},
        )
        _apply_scene_fit(layout["scene"], lo, hi, ratio, shown, signs)
    view_layouts = [{} for _ in views]
    if planar:
        for k in range(len(views)):
            x, y = _view_ranges_2d(fig, visibility[k])
            view_layouts[k] = {"xaxis.range": x, "yaxis.range": y}
        layout["xaxis"]["range"], layout["yaxis"]["range"] = _view_ranges_2d(fig, visibility[0])
    if len(views) > 1:
        layout["updatemenus"] = [
            dict(
                type="dropdown",
                direction="down",
                x=0.0,
                xanchor="left",
                y=1.0,
                yanchor="bottom",
                pad=dict(t=4, b=4),
                bgcolor="white",
                bordercolor="#cbd5e1",
                font=dict(size=12),
                buttons=[
                    dict(
                        label=v.name,
                        method="update",
                        args=[{"visible": visibility[k]}, view_layouts[k]],
                    )
                    for k, v in enumerate(views)
                ],
            )
        ]
    fig.update_layout(**layout)
    return fig


# Plot width / height that the 3D scene is sized for in the figure itself; the
# HTML report refits it to the size it is actually shown (or printed) at.
PLOT_ASPECT = 1.6
SCENE_PAD = 0.08  # space around the content, as a fraction of its largest extent
FLAT_AXIS = 0.25  # the shortest a box side may be, relative to the longest


@dataclass
class SceneFit:
    """Axis ranges and aspect ratio of a 3D scene that fit its content and its
    labels (whose size is fixed in pixels) into a plot of a given size.
    Everything is in model axes. The report's script (fit3d in
    report.html.j2) repeats :meth:`solve`, so keep the two in step."""

    lo: np.ndarray
    hi: np.ndarray
    labels: list  # (anchor, x extent in px, y extent in px), see _label_extents
    right: np.ndarray  # screen right and up, in model axes
    up: np.ndarray

    def box(self, lo, hi, aspect: float):
        """Padded ranges and aspect ratio for content within lo..hi."""
        pad = SCENE_PAD * max((hi - lo).max(), 1e-9)
        lo, hi = lo - pad, hi + pad
        ratio = np.maximum((hi - lo) / (hi - lo).max(), FLAT_AXIS)
        # an orthographic view does not zoom with the eye's distance, so size
        # the box instead: its projection fills the plot's height, or its width
        ratio *= FILL * min(1 / (ratio @ np.abs(self.up)), aspect / (ratio @ np.abs(self.right)))
        return lo, hi, ratio

    def solve(self, aspect: float, plot_height: float):
        lo, hi, ratio = self.box(self.lo, self.hi, aspect)
        # the labels' extent in model units depends on the zoom, which depends
        # on the extent: a second pass settles it (the plot spans 2 aspect units)
        for _ in range(2):
            if not self.labels:
                break
            per_px = (hi - lo).max() / ratio.max() / (plot_height / 2)
            corners = _label_corners(self.labels, self.right, self.up, per_px)
            lo, hi, ratio = self.box(
                np.minimum(self.lo, corners.min(axis=0)),
                np.maximum(self.hi, corners.max(axis=0)),
                aspect,
            )
        return lo, hi, ratio

    def to_json(self, shown, signs) -> dict:
        def r(v):
            return [round(float(x), 9) for x in v]

        return {
            "lo": r(self.lo),
            "hi": r(self.hi),
            "right": r(self.right),
            "up": r(self.up),
            "labels": [[r(p), r(xs), r(ys)] for p, xs, ys in self.labels],
            "fill": FILL,
            "pad": SCENE_PAD,
            "flat": FLAT_AXIS,
            "shown": list(shown),
            "signs": list(signs),
        }


def _apply_scene_fit(scene: dict, lo, hi, ratio, shown, signs) -> None:
    """Set a scene's axis ranges (reversed where a model axis runs the other
    way), aspect ratio and tick counts from a fit in model axes."""
    # fewer ticks on shorter axes, whose labels would otherwise run together
    nticks = [int(n) for n in np.clip(np.round(9 * ratio / ratio.max()), 4, 9)]
    for name, a, sign in zip(("xaxis", "yaxis", "zaxis"), shown, signs, strict=True):
        scene[name]["range"] = [lo[a], hi[a]] if sign > 0 else [hi[a], lo[a]]
        scene[name]["nticks"] = nticks[a]
    scene["aspectratio"] = dict(x=ratio[shown[0]], y=ratio[shown[1]], z=ratio[shown[2]])


def _view_ranges_2d(fig: go.Figure, visible: list[bool]) -> tuple[list, list]:
    """Axis ranges that frame what one view of a planar figure shows, with a
    margin (labels near the edge may also spill into it: see cliponaxis)."""
    xs, ys = [], []
    for tr, shown in zip(fig.data, visible, strict=True):
        if shown and tr.x is not None:
            xs += [v for v in tr.x if v is not None]
            ys += [v for v in tr.y if v is not None]
    if not xs:
        return [-1, 1], [-1, 1]
    lo, hi = np.array([min(xs), min(ys)]), np.array([max(xs), max(ys)])
    pad = 0.08 * max((hi - lo).max(), 1e-9)
    return [lo[0] - pad, hi[0] + pad], [lo[1] - pad, hi[1] + pad]


def _text_traces_3d(labels, color: str, group: str) -> list[go.Scatter3d]:
    """Text traces for (point, text, textposition) labels: one trace per
    position, since plotly misplaces 3D text given a textposition array of
    one or two entries."""
    traces = []
    for where in dict.fromkeys(pos for *_, pos in labels):
        chosen = [(p, text) for p, text, pos in labels if pos == where]
        traces.append(
            go.Scatter3d(
                x=[p[0] for p, _ in chosen],
                y=[p[1] for p, _ in chosen],
                z=[p[2] for p, _ in chosen],
                mode="text",
                text=[text for _, text in chosen],
                textposition=where,
                legendgroup=group,
                showlegend=False,
                hoverinfo="skip",
                textfont=dict(color=color, size=11, family=FONT),
            )
        )
    return traces


def _label_extents(fig: go.Figure) -> list:
    """Each 3D text label as (anchor, x extent, y extent): the box its text
    covers on screen, in pixels from the anchor (x right, y up), estimated
    from the text length and position."""
    labels = []
    for tr in fig.data:
        if tr.type != "scatter3d" or tr.mode != "text" or tr.x is None:
            continue
        size = tr.textfont.size or 12
        vertical, horizontal = (tr.textposition or "middle center").split()
        for x, y, z, text in zip(tr.x, tr.y, tr.z, tr.text, strict=True):
            w, h = 0.62 * size * len(str(text)), 1.3 * size
            gap_x, gap_y = 1.2 * size, 0.7 * size  # plotly's offset from the point
            xs = {"left": (-gap_x - w, -gap_x), "right": (gap_x, gap_x + w)}.get(
                horizontal, (-w / 2, w / 2)
            )
            ys = {"bottom": (-gap_y - h, -gap_y), "top": (gap_y, gap_y + h)}.get(
                vertical, (-h / 2, h / 2)
            )
            labels.append((np.array([x, y, z], float), xs, ys))
    return labels


def _label_corners(labels, right, up, per_px: float) -> np.ndarray:
    """Corners of labels (see _label_extents) in model axes, at ``per_px``
    model units per screen pixel."""
    corners = [
        p + (sx * np.asarray(right) + sy * np.asarray(up)) * per_px
        for p, xs, ys in labels
        for sx in xs
        for sy in ys
    ]
    return np.array(corners) if corners else np.zeros((0, 3))


def _label_boxes(fig: go.Figure, right, up, per_px: float) -> np.ndarray:
    """Corners of a figure's 3D text labels in model axes. Plotly clips 3D
    text at the axis ranges, so the ranges must take these in."""
    return _label_corners(_label_extents(fig), right, up, per_px)


def up_axis(value=None) -> np.ndarray:
    """The unit vector of an up axis given as 'x', 'y', 'z' or signed like
    '-y' (any case); z when None."""
    if value is None:
        return np.array([0.0, 0.0, 1.0])
    axis = parse_axis_name(value) if isinstance(value, str) else None
    if axis is None:
        raise InputError(f"up axis must be x, y or z, optionally signed like -y; got {value!r}")
    return axis + 0.0  # -0.0 components (from '-x') become 0.0


def _view_axes(up) -> np.ndarray:
    """The model axes that run across the page, into the page and up in the
    3D view (rows). The view is the same for every up axis, as for a rotated
    model: x runs across (y when x is up) and the third axis completes a
    right-handed set. With y up, x runs to the lower right and z out of the
    page, as in CAD."""
    up = np.asarray(up, float)
    across = np.array([0.0, 1.0, 0.0]) if abs(up[0]) > 0.5 else np.array([1.0, 0.0, 0.0])
    return np.array([across, np.cross(up, across), up]) + 0.0


def _camera(up) -> tuple[np.ndarray, np.ndarray]:
    """Eye position and up vector of the default 3D view, in model axes."""
    across, into, up = _view_axes(up)
    return EYE[0] * across + EYE[1] * into + EYE[2] * up, up


def _plotly_axes(up) -> tuple[list[int], list[float]]:
    """For plotly's x, y and z axes: the model axis each one shows (0, 1, 2
    for x, y, z) and +1, or -1 where that model axis runs the other way."""
    rows = _view_axes(up)
    shown = [int(np.argmax(np.abs(r))) for r in rows]
    return shown, [float(r[a]) for r, a in zip(rows, shown, strict=True)]


def _screen_axes(eye, up) -> tuple[np.ndarray, np.ndarray]:
    """Screen right and screen up, as unit vectors in model axes, for a camera
    at ``eye`` looking at the centre of the scene with ``up`` vertical."""
    view = -np.asarray(eye, float) / np.linalg.norm(eye)
    right = np.cross(view, up)
    right /= np.linalg.norm(right)
    return right, np.cross(right, view)


def _text_position(dx: float, dy: float) -> str:
    """The plotly textposition that puts text on the side (dx, dy) of its
    point, in one of eight directions."""
    s = np.sin(np.pi / 8) * np.hypot(dx, dy)
    vertical = "top" if dy > s else ("bottom" if dy < -s else "middle")
    horizontal = "right" if dx > s else ("left" if dx < -s else "center")
    return f"{vertical} {horizontal}"


def _scene_bounds(fig: go.Figure) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for tr in fig.data:
        if tr.type not in ("scatter3d", "mesh3d") or tr.x is None:
            continue
        xyz = np.array(
            [[np.nan if v is None else v for v in c] for c in (tr.x, tr.y, tr.z)], dtype=float
        ).T
        pts.append(xyz[~np.isnan(xyz).any(axis=1)])
    allp = np.vstack(pts) if pts else np.zeros((1, 3))
    return allp.min(axis=0), allp.max(axis=0)


def sweep_figure(x, x_label, series, labels, title="", y_label="") -> go.Figure:
    fig = go.Figure()
    for ys, label in zip(series, labels, strict=True):
        fig.add_trace(go.Scatter(x=x, y=ys, mode="lines+markers", name=label))
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=16)),
        font=dict(family=FONT, size=13),
        xaxis_title=x_label,
        yaxis_title=y_label,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(gridcolor=PALETTE["grid"]),
        yaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor="#94a3b8"),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"),
        margin=dict(l=10, r=10, t=100, b=10),
    )
    return fig
