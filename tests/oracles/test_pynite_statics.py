"""Cross-check engmech statics against PyNite, an independent 3D frame FE solver.

PyNite (PyNiteFEA) solves elastic frames by the direct stiffness method, which
shares no code or formulation with engmech's rigid-body equilibrium solver, so
agreement between the two is strong evidence that both are right.

What is compared, and why the two must agree:

* Statically determinate structures (trees welded to one fixed support,
  pin + roller frames, three-hinged frames, simple trusses): reactions and
  member end forces follow from equilibrium alone, so they do not depend on
  member stiffness. A linear elastic analysis must reproduce engmech's
  rigid-body answer to solver round-off.
* Statically indeterminate structures with elastic supports (a beam on
  springs, a plate on springs, a bolt group, a block on redundant inclined
  links): engmech treats the body as rigid and shares the redundant reactions
  by least work. The FE model uses springs of the same stiffness and members
  stiffened by a factor rho over the springs; its answer converges to
  engmech's as 1/rho, and the tests check that convergence rate as well as
  the size of the difference at every rho.

Conventions used to line the two models up:

* engmech reports the wrench on body ``b`` from body ``a`` (or from the
  ground) about the joint point in global axes. PyNite's reactions are the
  forces on the structure from the support, about the node, in global axes,
  so they compare directly.
* PyNite's member end forces ``Member3D.f()`` are the forces ON the member
  FROM the nodes, in member local axes. They are rotated to global axes with
  the member's transformation matrix (``T.T @ f``; T is orthogonal). A joint
  force on body ``b`` from body ``a`` at a node is then the sum of those
  end forces over b's members at that node, provided every nodal load at that
  node is applied to body ``a`` in the engmech model.
* Truss bars and links are frame members with all end rotations released
  (and torsion at one end), with the node rotations restrained, so they carry
  axial force only. The axial force at the j end in local x is the tension.
* A roller with an arbitrary normal is a released link to a fixed ground
  node; the ground node's reaction is the force the roller puts on the body.
* Planar models are PyNite models in the XY plane with DZ, RX and RY
  restrained at every node.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

import engmech as em

FEModel3D = pytest.importorskip("Pynite").FEModel3D

pytestmark = pytest.mark.oracle

COMBO = "Combo 1"  # PyNite's default load combination (factor 1 on 'Case 1')
SEEDS = range(20)

# Determinate structures: the FE answer equals the rigid-body answer up to round-off
# in the stiffness solve. Observed differences are below 1e-11 of the load scale;
# 1e-9 leaves a wide margin while still catching any modelling slip.
DETERMINATE_RTOL = 1e-9

# Elastic supports: the FE model's reactions converge to the rigid-body limit as
# C / rho, with rho the member-to-support stiffness ratio. C measures how much the
# members' own flexibility (~1/rho of the supports') redistributes the reactions;
# over all seeds C < 0.08 (relative to the load scale), so C_MAX = 0.2 bounds it
# with margin. At rho = 1e4 that allows 2e-5. The ratios stop there: beyond ~1e6
# round-off in the stiffness solve (condition number ~ rho) takes over, and PyNite
# rejects rho >~ 1e7 as singular.
RHOS = (1e1, 1e2, 1e3, 1e4)
C_MAX = 0.2


# --------------------------------------------------------------------------- PyNite helpers


def fe_model(E=200e9, G=80e9, A=1e-2, Iy=1e-5, Iz=1e-5, J=2e-5) -> FEModel3D:
    fe = FEModel3D()
    fe.add_material("mat", E, G, 0.3, 0.0)
    fe.add_section("sec", A, Iy, Iz, J)
    return fe


def fe_node(fe: FEModel3D, name: str, p) -> None:
    p = np.asarray(p, dtype=float)
    fe.add_node(name, float(p[0]), float(p[1]), float(p[2]) if p.size == 3 else 0.0)


def fe_member(fe: FEModel3D, name: str, i: str, j: str, *, axial_only: bool = False) -> None:
    fe.add_member(name, i, j, "mat", "sec")
    if axial_only:  # a pin-ended bar: release bending at both ends and torsion at one
        fe.def_releases(name, Rxi=True, Ryi=True, Rzi=True, Ryj=True, Rzj=True)


def fe_ground_link(fe: FEModel3D, name: str, node: str, direction) -> str:
    """A pin-ended link from ``node`` to a new fixed ground node, one unit
    along ``-direction``: it pushes the node along ``+direction`` in compression,
    like a roller with that normal. Returns the ground node's name."""
    n = fe.nodes[node]
    p = np.array([n.X, n.Y, n.Z]) - np.asarray(direction, float)
    ground = f"{name}_ground"
    fe_node(fe, ground, p)
    fe.def_support(ground, True, True, True, True, True, True)
    fe_member(fe, name, ground, node, axial_only=True)
    return ground


def fe_restrain(fe: FEModel3D, node: str, dofs: str) -> None:
    """Add restraints (e.g. 'DZ RX RY') to a node, keeping those already there."""
    n = fe.nodes[node]
    for dof in dofs.split():
        setattr(n, f"support_{dof}", True)


def fe_planar(fe: FEModel3D) -> None:
    """Hold every node in the XY plane (DZ, RX, RY) for a planar analysis."""
    for name in fe.nodes:
        fe_restrain(fe, name, "DZ RX RY")


def fe_solve(fe: FEModel3D) -> None:
    fe.analyze_linear(check_stability=True)


def fe_reaction(fe: FEModel3D, node: str) -> np.ndarray:
    """Reaction on the structure from the support at ``node`` (Fx..Mz, global)."""
    n = fe.nodes[node]
    return np.array([getattr(n, f"Rxn{c}")[COMBO] for c in ("FX", "FY", "FZ", "MX", "MY", "MZ")])


def fe_end_wrench(fe: FEModel3D, member: str, node: str) -> np.ndarray:
    """Force and moment (about the node) that ``node`` applies to the end of
    ``member`` there, in global axes."""
    mem = fe.members[member]
    T = mem.T()  # 12x12, diagonal blocks = direction cosines (rows: local x, y, z)
    f_global = T.T @ mem.f(COMBO)  # local end forces rotated to global axes
    if mem.i_node.name == node:
        return f_global[:6, 0]
    assert mem.j_node.name == node, (member, node)
    return f_global[6:, 0]


def fe_tension(fe: FEModel3D, member: str) -> float:
    """Axial force of an unloaded bar, positive in tension: the j end is pulled
    along local +x in tension."""
    return float(fe.members[member].f(COMBO)[6, 0])


# --------------------------------------------------------------------------- comparison


@dataclass
class Comparison:
    """Collects engmech-vs-PyNite differences, each relative to a load scale."""

    force_scale: float
    length_scale: float
    diffs: dict[str, float] = field(default_factory=dict)

    def wrench(self, what: str, engmech_value, oracle_value) -> None:
        a, b = np.asarray(engmech_value, float), np.asarray(oracle_value, float)
        scale = np.array([1, 1, 1, self.length_scale, self.length_scale, self.length_scale])
        scale = scale[: a.size] * self.force_scale
        self.diffs[what] = float(np.max(np.abs(a - b) / scale))

    def force(self, what: str, engmech_value, oracle_value) -> None:
        a, b = np.asarray(engmech_value, float), np.asarray(oracle_value, float)
        self.diffs[what] = float(np.max(np.abs(a - b)) / self.force_scale)

    @property
    def worst(self) -> tuple[str, float]:
        return max(self.diffs.items(), key=lambda kv: kv[1])

    def check(self, rtol: float) -> None:
        name, diff = self.worst
        assert diff < rtol, f"{name}: relative difference {diff:.3g} (tolerance {rtol:g})"


def solved(model: em.Model):
    results = model.solve()
    assert results.status == "ok", results.primary.warnings
    assert results.primary.verified, results.primary.max_residual
    return results.primary


def unit_vector(rng) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def planar_unit(rng) -> np.ndarray:
    a = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(a), np.sin(a), 0.0])


def cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


# --------------------------------------------------------------------------- random frames


@dataclass
class Frame:
    """Members welded at nodes, with loads. Node indices refer to ``points``."""

    points: np.ndarray
    members: list[tuple[int, int]]  # PyNite (i, j) nodes, in random orientation
    node_loads: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    # member, position as a fraction of the length from node i, force, couple
    point_loads: list[tuple[int, float, np.ndarray, np.ndarray]] = field(default_factory=list)
    # member, start and end as fractions from node i, intensities, unit direction
    line_loads: list[tuple[int, float, float, float, float, np.ndarray]] = field(
        default_factory=list
    )

    def at(self, member: int, fraction: float) -> np.ndarray:
        i, j = self.members[member]
        return self.points[i] + fraction * (self.points[j] - self.points[i])

    def length(self, member: int) -> float:
        i, j = self.members[member]
        return float(np.linalg.norm(self.points[j] - self.points[i]))

    @property
    def size(self) -> float:
        return float(np.max(np.linalg.norm(self.points - self.points.mean(axis=0), axis=1)))

    @property
    def load_scale(self) -> float:
        total = sum(np.linalg.norm(f) for _, f, _ in self.node_loads)
        total += sum(np.linalg.norm(f) for _, _, f, _ in self.point_loads)
        total += sum(
            self.length(e) * (b - a) * (abs(w1) + abs(w2)) / 2
            for e, a, b, w1, w2, _ in self.line_loads
        )
        return float(total)

    def add_random_loads(self, rng, nodes, planar: bool = False) -> None:
        """1-3 nodal forces and couples, member point forces and couples, and
        line loads (full or partial span, linearly varying, any direction)."""

        def force():
            return 1e3 * (planar_unit(rng) if planar else unit_vector(rng)) * rng.uniform(0.2, 2)

        def couple():
            if planar:
                return np.array([0.0, 0.0, 1e3 * rng.normal()])
            return 1e3 * rng.normal(size=3)

        n_members = len(self.members)
        for node in rng.choice(nodes, size=min(len(nodes), rng.integers(1, 4)), replace=False):
            self.node_loads.append((int(node), force(), couple()))
        for _ in range(rng.integers(1, 4)):
            e = int(rng.integers(n_members))
            self.point_loads.append((e, float(rng.uniform(0.1, 0.9)), force(), couple()))
        for _ in range(rng.integers(1, 4)):
            e = int(rng.integers(n_members))
            a, b = (0.0, 1.0) if rng.random() < 0.5 else sorted(rng.uniform(0, 1, 2))
            if b - a < 0.1:
                a, b = 0.0, 1.0
            w1, w2 = 1e3 * rng.normal(size=2)
            if rng.random() < 0.3:  # perpendicular to the member, like wind on a rafter
                i, j = self.members[e]
                axis = self.points[j] - self.points[i]
                d = np.cross(axis, [0.0, 0.0, 1.0] if planar else unit_vector(rng))
                d /= np.linalg.norm(d)
            else:
                d = planar_unit(rng) if planar else unit_vector(rng)
            self.line_loads.append((e, float(a), float(b), float(w1), float(w2), d))


def fe_frame(fe: FEModel3D, frame: Frame) -> None:
    """Add a frame's nodes 'N<k>', members 'M<e>' and loads to a PyNite model."""
    for k, p in enumerate(frame.points):
        fe_node(fe, f"N{k}", p)
    for e, (i, j) in enumerate(frame.members):
        fe_member(fe, f"M{e}", f"N{i}", f"N{j}")
    for node, f, c in frame.node_loads:
        for axis, fv, cv in zip("XYZ", f, c, strict=True):
            fe.add_node_load(f"N{node}", f"F{axis}", float(fv))
            fe.add_node_load(f"N{node}", f"M{axis}", float(cv))
    for e, x, f, c in frame.point_loads:
        at = x * frame.length(e)
        for axis, fv, cv in zip("XYZ", f, c, strict=True):
            fe.add_member_pt_load(f"M{e}", f"F{axis}", float(fv), at)
            fe.add_member_pt_load(f"M{e}", f"M{axis}", float(cv), at)
    for e, a, b, w1, w2, d in frame.line_loads:
        L = frame.length(e)
        # a load in an arbitrary direction is the sum of its global components
        for axis, dc in zip("XYZ", d, strict=True):
            fe.add_member_dist_load(f"M{e}", f"F{axis}", w1 * dc, w2 * dc, a * L, b * L)


def em_frame_loads(model: em.Model, frame: Frame, member_body, node_body, planar=False) -> None:
    """Apply a frame's loads to engmech bodies (per member / per node)."""

    def v(x):
        return x[:2] if planar else x

    def couple(c):
        return float(c[2]) if planar else c

    for node, f, c in frame.node_loads:
        p = frame.points[node]
        model.load(em.Force(v(f), at=v(p)), body=node_body[node])
        model.load(em.Moment(couple(c), at=v(p)), body=node_body[node])
    for e, x, f, c in frame.point_loads:
        p = frame.at(e, x)
        model.load(em.Force(v(f), at=v(p)), body=member_body[e])
        model.load(em.Moment(couple(c), at=v(p)), body=member_body[e])
    for e, a, b, w1, w2, d in frame.line_loads:
        load = em.DistributedLoad(
            v(frame.at(e, a)), v(frame.at(e, b)), {"start": w1, "end": w2}, v(d)
        )
        model.load(load, body=member_body[e])


def one_body(model: em.Model, frame: Frame, body: str, planar: bool = False) -> None:
    """Apply all of a frame's loads to one engmech body."""
    members = dict.fromkeys(range(len(frame.members)), body)
    em_frame_loads(model, frame, members, dict.fromkeys(range(len(frame.points)), body), planar)


# --------------------------------------------------------------------------- 1, 2: 3D trees


@dataclass
class Tree(Frame):
    parent: dict[int, int] = field(default_factory=dict)  # node -> member towards node 0


def random_tree(seed: int) -> Tree:
    """A random 3D tree of 2-8 welded members; node 0 is the fixed support."""
    rng = np.random.default_rng(seed)
    n_nodes = int(rng.integers(3, 10))
    points = [rng.uniform(-1, 1, 3)]
    members, parent = [], {}
    while len(points) < n_nodes:
        base = int(rng.integers(len(points)))
        q = points[base] + rng.uniform(0.8, 2.5) * unit_vector(rng)
        if min(np.linalg.norm(q - p) for p in points) < 0.5:
            continue
        k = len(points)
        points.append(q)
        members.append((base, k) if rng.random() < 0.5 else (k, base))
        parent[k] = len(members) - 1
    tree = Tree(np.array(points), members, parent=parent)
    tree.add_random_loads(rng, nodes=np.arange(1, n_nodes))
    return tree


def fe_tree(tree: Tree) -> FEModel3D:
    fe = fe_model()
    fe_frame(fe, tree)
    fe.def_support("N0", True, True, True, True, True, True)
    fe_solve(fe)
    return fe


def compare_tree_single_body(seed: int) -> Comparison:
    """A welded 3D tree on one fixed support, as one engmech body."""
    tree = random_tree(seed)
    m = em.Model()
    m.body("frame")
    m.support("S", em.Fixed(at=tree.points[0]), body="frame")
    one_body(m, tree, "frame")
    S = solved(m)["S"]

    fe = fe_tree(tree)
    cmp = Comparison(tree.load_scale, tree.size)
    cmp.wrench("S", S.wrench, fe_reaction(fe, "N0"))
    return cmp


@pytest.mark.parametrize("seed", SEEDS)
def test_tree_frame_single_body(seed):
    """All six reaction components of a welded 3D tree."""
    compare_tree_single_body(seed).check(DETERMINATE_RTOL)


def split_tree(tree: Tree, rng):
    """Cut the tree into bodies at random nodes.

    Each node has a 'hub' body: the body of the first member placed there
    (its parent member, or the root's first member). At a cut node the other
    members start new bodies (one body for all of them, or one each), welded
    to the hub by Fixed joints. Nodal loads go on the hub body, so the joint
    force on a new body is exactly the sum of its members' FE end forces there.
    """
    n = len(tree.points)
    degree = np.zeros(n, int)
    for i, j in tree.members:
        degree[i] += 1
        degree[j] += 1
    candidates = [k for k in range(n) if degree[k] >= 2]
    cut = {k for k in candidates if rng.random() < 0.6} or {int(rng.choice(candidates))}
    one_each = {k: rng.random() < 0.5 for k in cut}

    member_body: dict[int, str] = {}
    hub: dict[int, str] = {}
    groups: dict[tuple, str] = {}
    for k in range(1, n):  # creation order: a member's inner node is placed before k
        e = tree.parent[k]
        i, j = tree.members[e]
        base = i if j == k else j
        if base not in hub:  # the root's first member
            body = f"b{len(groups)}"
            groups[(base, "hub")] = body
        elif base in cut:
            key = (base, k if one_each[base] else "rest")
            body = groups.setdefault(key, f"b{len(groups)}")
        else:
            body = hub[base]
        member_body[e] = body
        hub.setdefault(base, body)
        hub[k] = body

    joints = []  # (node, body a, body b): Fixed joint reporting the force on b from a
    for node in sorted(cut):
        others = sorted(
            {member_body[e] for e, (i, j) in enumerate(tree.members) if node in (i, j)}
            - {hub[node]}
        )
        for other in others:
            pair = (hub[node], other) if rng.random() < 0.5 else (other, hub[node])
            joints.append((node, *pair))
    return member_body, hub, joints


def compare_tree_split(seed: int) -> Comparison:
    """The same trees cut into bodies joined by Fixed joints."""
    tree = random_tree(seed)
    rng = np.random.default_rng(1000 + seed)
    member_body, hub, joints = split_tree(tree, rng)

    m = em.Model()
    for body in sorted(set(member_body.values())):
        m.body(body)
    m.support("S", em.Fixed(at=tree.points[0]), body=hub[0])
    for node, a, b in joints:
        m.joint(f"J{node}_{a}_{b}", em.Fixed(at=tree.points[node]), bodies=(a, b))
    em_frame_loads(m, tree, member_body, hub)
    result = solved(m)
    assert joints

    fe = fe_tree(tree)
    cmp = Comparison(tree.load_scale, tree.size)
    cmp.wrench("S", result["S"].wrench, fe_reaction(fe, "N0"))
    for node, a, b in joints:
        new_body = b if a == hub[node] else a
        held = np.zeros(6)
        for e, (i, j) in enumerate(tree.members):
            if node in (i, j) and member_body[e] == new_body:
                held += fe_end_wrench(fe, f"M{e}", f"N{node}")
        expected = held if b == new_body else -held  # the force on b from a
        cmp.wrench(f"J{node}_{a}_{b}", result[f"J{node}_{a}_{b}"].wrench, expected)
    return cmp


@pytest.mark.parametrize("seed", SEEDS)
def test_tree_frame_split_into_bodies(seed):
    """Every Fixed joint's wrench equals the FE end forces of the members it holds."""
    compare_tree_split(seed).check(DETERMINATE_RTOL)


# --------------------------------------------------------------------------- 3: planar


def random_polyline(rng, start, n_members) -> list[np.ndarray]:
    heading = rng.uniform(0, 2 * np.pi)
    points = [np.asarray(start, float)]
    for _ in range(n_members):
        heading += rng.uniform(-1.6, 1.6)
        step = rng.uniform(1.0, 3.0) * np.array([np.cos(heading), np.sin(heading), 0.0])
        points.append(points[-1] + step)
    return points


def compare_pin_roller_frame(seed: int) -> Comparison:
    """A planar polyline frame on a pin and an inclined roller."""
    rng = np.random.default_rng(seed)
    while True:
        points = random_polyline(rng, [0, 0, 0], int(rng.integers(1, 5)))
        r = int(rng.integers(1, len(points)))
        normal = planar_unit(rng)
        arm = points[r] - points[0]
        # the roller's line of action must pass well clear of the pin
        if abs(cross2(arm, normal)) > 0.3 * np.linalg.norm(arm):
            break
    frame = Frame(
        np.array(points),
        [(k, k + 1) if rng.random() < 0.5 else (k + 1, k) for k in range(len(points) - 1)],
    )
    frame.add_random_loads(rng, nodes=np.arange(len(points)), planar=True)

    m = em.Model(planar=True)
    m.body("frame")
    m.support("A", em.Pin(at=points[0][:2]), body="frame")
    m.support("B", em.Roller(at=points[r][:2], normal=normal[:2]), body="frame")
    one_body(m, frame, "frame", planar=True)
    result = solved(m)

    fe = fe_model()
    fe_frame(fe, frame)
    ground = fe_ground_link(fe, "roller", f"N{r}", normal)
    fe_planar(fe)
    fe_restrain(fe, "N0", "DX DY")
    fe_solve(fe)

    cmp = Comparison(frame.load_scale, frame.size)
    cmp.force("A", result["A"].force, fe_reaction(fe, "N0")[:3])
    roller = fe_reaction(fe, ground)
    cmp.force("B", result["B"].force, roller[:3])
    cmp.force("B.N", result["B"].scalars["N"], roller[:3] @ normal)
    return cmp


@pytest.mark.parametrize("seed", SEEDS)
def test_planar_frame_pin_and_roller(seed):
    """Pin reaction and roller force (vector and N) of a planar frame."""
    compare_pin_roller_frame(seed).check(DETERMINATE_RTOL)


def compare_three_hinged_frame(seed: int) -> Comparison:
    """Two polyline bodies on pins, hinged together at the crown."""
    rng = np.random.default_rng(seed)
    while True:
        span = rng.uniform(4, 10)
        A = np.zeros(3)
        B = np.array([span, rng.uniform(-2, 2), 0.0])
        C = np.array([span * rng.uniform(0.25, 0.75), rng.uniform(1.5, 5), 0.0])
        if abs(cross2(C - A, B - A)) > 0.3 * np.linalg.norm(C - A) * np.linalg.norm(B - A):
            break

    def leg(p, q):  # a kinked path from p to q
        n = int(rng.integers(1, 4))
        inner = [
            p + (q - p) * t + rng.uniform(-0.6, 0.6, 3) * [1, 1, 0]
            for t in np.sort(rng.uniform(0.15, 0.85, n - 1))
        ]
        return [p, *inner, q]

    left, right = leg(A, C), leg(C, B)
    points = np.array(left + right[1:])
    crown = len(left) - 1
    members = [(k, k + 1) if rng.random() < 0.5 else (k + 1, k) for k in range(len(points) - 1)]
    frame = Frame(points, members)
    frame.add_random_loads(rng, nodes=np.arange(len(points)), planar=True)
    # nodal loads at the crown: forces only (a couple there would need a body)
    frame.node_loads = [(k, f, np.zeros(3) if k == crown else c) for k, f, c in frame.node_loads]
    member_body = {e: "L" if e < crown else "R" for e in range(len(members))}
    node_body = {k: "L" if k <= crown else "R" for k in range(len(points))}

    m = em.Model(planar=True)
    m.body("L")
    m.body("R")
    m.support("A", em.Pin(at=A[:2]), body="L")
    m.support("B", em.Pin(at=B[:2]), body="R")
    pair = ("L", "R") if rng.random() < 0.5 else ("R", "L")
    m.joint("C", em.Pin(at=C[:2]), bodies=pair)
    em_frame_loads(m, frame, member_body, node_body, planar=True)
    result = solved(m)

    fe = fe_model()
    fe_frame(fe, frame)
    # the hinge: release in-plane bending at the crown end of the last left member
    i, _ = members[crown - 1]
    end = "i" if i == crown else "j"
    fe.def_releases(f"M{crown - 1}", **{f"Rz{end}": True})
    fe_planar(fe)
    fe_restrain(fe, "N0", "DX DY")
    fe_restrain(fe, f"N{len(points) - 1}", "DX DY")
    fe_solve(fe)

    # the right member at the crown receives exactly the force from the left body
    on_right = fe_end_wrench(fe, f"M{crown}", f"N{crown}")[:3]
    cmp = Comparison(frame.load_scale, frame.size)
    cmp.force("A", result["A"].force, fe_reaction(fe, "N0")[:3])
    cmp.force("B", result["B"].force, fe_reaction(fe, f"N{len(points) - 1}")[:3])
    cmp.force("C", result["C"].force, on_right if pair == ("L", "R") else -on_right)
    assert abs(result["C"].moment[2]) < 1e-9 * frame.load_scale * frame.size
    return cmp


@pytest.mark.parametrize("seed", SEEDS)
def test_three_hinged_frame(seed):
    """Both pin reactions and the hinge force of a three-hinged frame."""
    compare_three_hinged_frame(seed).check(DETERMINATE_RTOL)


# --------------------------------------------------------------------------- trusses


@dataclass
class Truss:
    points: np.ndarray
    bars: list[tuple[int, int]]
    # node, kind ("ball" | "pin" | "link" | "roller"), direction for link/roller
    supports: list[tuple[int, str, np.ndarray | None]]
    loads: list[tuple[int, np.ndarray]]

    @property
    def load_scale(self) -> float:
        return float(sum(np.linalg.norm(f) for _, f in self.loads))


def node_loads(rng, n_nodes: int, supported: set[int], planar: bool):
    """Forces at a random set of nodes, always including an unsupported one
    (loads only at supports would leave every bar unstressed)."""
    free = [k for k in range(n_nodes) if k not in supported]
    first = int(rng.choice(free))
    others = [k for k in range(n_nodes) if k != first]
    at = [first, *rng.choice(others, size=int(rng.integers(0, len(others) + 1)), replace=False)]
    return [
        (int(k), 1e3 * rng.uniform(0.2, 2) * (planar_unit(rng) if planar else unit_vector(rng)))
        for k in at
    ]


def random_planar_truss(seed: int) -> Truss:
    """A simple truss: a triangle, then nodes each hung on the two ends of an
    existing bar, on a pin and an inclined roller."""
    rng = np.random.default_rng(seed)
    p0 = np.zeros(3)
    p1 = rng.uniform(2, 4) * planar_unit(rng)
    mid, perp = (p0 + p1) / 2, np.cross([0, 0, 1], p1 - p0)
    p2 = mid + rng.uniform(0.4, 1.2) * perp + rng.uniform(-0.4, 0.4) * (p1 - p0)
    points, bars = [p0, p1, p2], [(0, 1), (1, 2), (0, 2)]
    n_nodes = int(rng.integers(4, 11))
    while len(points) < n_nodes:
        a, b = bars[int(rng.integers(len(bars)))]
        pa, pb = points[a], points[b]
        side = rng.choice([-1, 1])
        q = (pa + pb) / 2 + side * rng.uniform(0.5, 1.2) * np.cross([0, 0, 1], pb - pa)
        q += rng.uniform(-0.4, 0.4) * (pb - pa)
        if min(np.linalg.norm(q - p) for p in points) < 0.7:
            continue
        k = len(points)
        points.append(q)
        bars += [(a, k), (k, b)]
    points = np.array(points)
    while True:
        pin, roller = (int(x) for x in rng.choice(len(points), size=2, replace=False))
        normal = planar_unit(rng)
        arm = points[roller] - points[pin]
        if abs(cross2(arm, normal)) > 0.3 * np.linalg.norm(arm):
            break
    bars = [(i, j) if rng.random() < 0.5 else (j, i) for i, j in bars]
    loads = node_loads(rng, len(points), {pin, roller}, planar=True)
    return Truss(points, bars, [(pin, "pin", None), (roller, "roller", normal)], loads)


def random_space_truss(seed: int) -> Truss:
    """A determinate space truss: a triangle held by a ball (3 reactions) and
    three one-force supports (links and rollers), then nodes each held by three
    non-coplanar bars to existing nodes."""
    rng = np.random.default_rng(seed)
    while True:  # base triangle and supports that hold it as a rigid body
        tri = [rng.uniform(-1, 1, 3) * 2 for _ in range(3)]
        if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) < 2.0:
            continue
        dirs = [unit_vector(rng) for _ in range(3)]
        lines = [(tri[0], e) for e in np.eye(3)]
        lines += [(tri[1], dirs[0]), (tri[1], dirs[1]), (tri[2], dirs[2])]
        plucker = np.array([np.concatenate([d, np.cross(p, d)]) for p, d in lines])
        if np.linalg.cond(plucker) < 30:
            break
    points, bars = list(tri), [(0, 1), (1, 2), (0, 2)]
    n_nodes = int(rng.integers(4, 11))
    while len(points) < n_nodes:
        q = np.mean(points, axis=0) + rng.normal(size=3) * 2
        if min(np.linalg.norm(q - p) for p in points) < 0.8:
            continue
        ends = rng.choice(len(points), size=3, replace=False)
        u = np.array([(points[e] - q) / np.linalg.norm(points[e] - q) for e in ends])
        if abs(np.linalg.det(u)) < 0.3:  # the three bars must not be near coplanar
            continue
        k = len(points)
        points.append(q)
        bars += [(int(e), k) for e in ends]
    points = np.array(points)
    kinds = ["link" if rng.random() < 0.5 else "roller" for _ in range(3)]
    supports = [
        (0, "ball", None),
        (1, kinds[0], dirs[0]),
        (1, kinds[1], dirs[1]),
        (2, kinds[2], dirs[2]),
    ]
    bars = [(i, j) if rng.random() < 0.5 else (j, i) for i, j in bars]
    loads = node_loads(rng, len(points), {0, 1, 2}, planar=False)
    return Truss(points, bars, supports, loads)


def compare_truss(truss: Truss, planar: bool, seed: int) -> Comparison:
    """engmech: one particle per node, a Link joint per bar. PyNite: pin-ended
    bars; rollers and links as pin-ended links to fixed ground nodes."""
    rng = np.random.default_rng(2000 + seed)  # which end of each link is 'a'

    def v(x):
        return x[:2] if planar else x

    m = em.Model(planar=planar)
    for k in range(len(truss.points)):
        m.body(f"n{k}", particle=True)
    for e, (i, j) in enumerate(truss.bars):
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        ends = [v(truss.points[a]), v(truss.points[b])]
        m.joint(f"bar{e}", em.Link(ends=ends), bodies=(f"n{a}", f"n{b}"))
    for s, (k, kind, d) in enumerate(truss.supports):
        p = truss.points[k]
        if kind == "ball":
            joint = em.Ball(at=p)
        elif kind == "pin":
            joint = em.Pin(at=v(p))
        elif kind == "roller":
            joint = em.Roller(at=v(p), normal=v(d))
        else:  # a link to an anchor one unit along -d, where the FE ground link ends
            joint = em.Link(at=v(p), anchor=v(p - d))
        m.support(f"S{s}", joint, body=f"n{k}")
    for k, f in truss.loads:
        m.load(em.Force(v(f), at=v(truss.points[k])), body=f"n{k}")
    result = solved(m)

    fe = fe_model()
    for k, p in enumerate(truss.points):
        fe_node(fe, f"N{k}", p)
        fe_restrain(fe, f"N{k}", "RX RY RZ")  # bars carry no moments; nodes do not rotate
    for e, (i, j) in enumerate(truss.bars):
        fe_member(fe, f"B{e}", f"N{i}", f"N{j}", axial_only=True)
    ground = {}
    for s, (k, kind, d) in enumerate(truss.supports):
        if kind in ("ball", "pin"):
            fe_restrain(fe, f"N{k}", "DX DY DZ")
        else:
            ground[s] = fe_ground_link(fe, f"S{s}", f"N{k}", d)
    if planar:
        fe_planar(fe)
    for k, f in truss.loads:
        for axis, fv in zip("XYZ", f, strict=True):
            fe.add_node_load(f"N{k}", f"F{axis}", float(fv))
    fe_solve(fe)

    cmp = Comparison(truss.load_scale, 1.0)
    for e in range(len(truss.bars)):
        cmp.force(f"bar{e}.T", result[f"bar{e}"].scalars["T"], fe_tension(fe, f"B{e}"))
    for s, (k, kind, d) in enumerate(truss.supports):
        R = result[f"S{s}"]
        if kind in ("ball", "pin"):
            cmp.force(f"S{s}", R.force, fe_reaction(fe, f"N{k}")[:3])
            continue
        on_body = fe_reaction(fe, ground[s])[:3]  # force the link puts on the node
        cmp.force(f"S{s}", R.force, on_body)
        if kind == "roller":
            cmp.force(f"S{s}.N", R.scalars["N"], on_body @ d)
        else:
            cmp.force(f"S{s}.T", R.scalars["T"], fe_tension(fe, f"S{s}"))
    return cmp


def compare_planar_truss(seed: int) -> Comparison:
    return compare_truss(random_planar_truss(seed), planar=True, seed=seed)


def compare_space_truss(seed: int) -> Comparison:
    return compare_truss(random_space_truss(seed), planar=False, seed=seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_planar_truss(seed):
    """Bar tensions and reactions of a simple truss on a pin and an inclined roller."""
    compare_planar_truss(seed).check(DETERMINATE_RTOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_space_truss(seed):
    """Bar tensions and reactions of a determinate space truss on a ball, links and rollers."""
    compare_space_truss(seed).check(DETERMINATE_RTOL)


# --------------------------------------------------------------------------- 5: elastic supports
#
# engmech: one rigid body on supports with stiffness; redundant reactions shared by
# least work. PyNite: the same supports as springs (nodal spring supports, or axial
# spring elements to fixed ground nodes for inclined links), the body as a frame whose
# members are rho times stiffer than the supports (rho = E*I / (k_mean * size^3), the
# ratio of a member's bending stiffness to the mean support stiffness).


@dataclass
class ElasticCase:
    frame: Frame
    planar: bool
    em_model: em.Model
    compare: dict[str, tuple[str, list[int]]]  # engmech support -> FE node, Fx..Mz indices
    springs: list[tuple[int, str, float]] = field(default_factory=list)  # node, 'DX'.., k
    # elastic links: node, unit direction d, k; the ground end is at node - d
    links: list[tuple[int, np.ndarray, float]] = field(default_factory=list)
    restraints: list[tuple[int, str]] = field(default_factory=list)  # node, e.g. 'DX DY RZ'


def fe_elastic(case: ElasticCase, rho: float) -> FEModel3D:
    I = 1e-5
    k_mean = float(np.mean([k for *_, k in case.springs + case.links]))
    E = rho * k_mean * case.frame.size**3 / I
    A = 12 * I / case.frame.size**2  # axial as stiff as bending: keeps K well conditioned
    fe = fe_model(E=E, G=E / 2.6, A=A, Iy=I, Iz=I, J=2 * I)
    fe_frame(fe, case.frame)
    for node, dof, k in case.springs:
        fe.def_support_spring(f"N{node}", dof, k)
    for s, (node, d, k) in enumerate(case.links):
        # an axial spring element to a fixed ground node: its reaction there is the
        # force the link puts on the body
        fe_node(fe, f"L{s}_ground", case.frame.points[node] - d)
        fe.def_support(f"L{s}_ground", True, True, True, True, True, True)
        fe.add_spring(f"L{s}", f"L{s}_ground", f"N{node}", k)
    for node, dofs in case.restraints:
        fe_restrain(fe, f"N{node}", dofs)
    if case.planar:
        fe_planar(fe)
    fe_solve(fe)
    return fe


def compare_elastic(case: ElasticCase, rhos=RHOS) -> list[Comparison]:
    """engmech's least-work answer against the FE model at each stiffness ratio."""
    result = solved(case.em_model)
    out = []
    for rho in rhos:
        fe = fe_elastic(case, rho)
        cmp = Comparison(case.frame.load_scale, case.frame.size)
        for name, (node, comps) in case.compare.items():
            cmp.wrench(name, result[name].wrench[comps], fe_reaction(fe, node)[comps])
        out.append(cmp)
    return out


def elastic_beam(seed: int) -> ElasticCase:
    """A straight beam on 3-5 vertical springs of different stiffness and one
    rigid horizontal roller, with point, couple and line loads anywhere."""
    rng = np.random.default_rng(seed)
    L = rng.uniform(4, 10)
    while True:  # springs apart, and no very short member (it would spoil conditioning)
        xs = np.sort(rng.uniform(0, L, int(rng.integers(3, 6))))
        stations = np.unique(np.concatenate([[0.0, L], xs]))
        if np.min(np.diff(xs)) > 0.1 * L and np.min(np.diff(stations)) > 0.03 * L:
            break
    points = np.column_stack([stations, np.zeros((len(stations), 2))])
    members = [(k, k + 1) if rng.random() < 0.5 else (k + 1, k) for k in range(len(points) - 1)]
    frame = Frame(points, members)
    frame.add_random_loads(rng, nodes=np.arange(len(points)), planar=True)
    horizontal = int(rng.integers(len(points)))

    m = em.Model(planar=True)
    m.body("beam")
    springs, compare = [], {}
    for s, x in enumerate(xs):
        k = float(1e6 * rng.uniform(0.3, 3))
        node = int(np.searchsorted(stations, x))
        m.support(f"K{s}", em.Roller(at=[x, 0], normal="+y", stiffness=k), body="beam")
        springs.append((node, "DY", k))
        compare[f"K{s}"] = (f"N{node}", [1])
    m.support("H", em.Roller(at=points[horizontal][:2], normal="+x"), body="beam")
    compare["H"] = (f"N{horizontal}", [0])
    one_body(m, frame, "beam", planar=True)
    return ElasticCase(frame, True, m, compare, springs, restraints=[(horizontal, "DX")])


def elastic_plate(seed: int) -> ElasticCase:
    """A rigid plate on 4-7 vertical springs, held in its plane by one rigid
    support (Fx, Fy, Mz), under eccentric 3D loads. The FE plate is a star of
    stiff members from the in-plane support to every spring and load point."""
    rng = np.random.default_rng(seed)
    while True:
        n = int(rng.integers(4, 8))
        xy = rng.uniform(-2, 2, (n, 2))
        spread = np.linalg.cond(np.column_stack([np.ones(n), xy]))
        gaps = [np.linalg.norm(a - b) for i, a in enumerate(xy) for b in xy[i + 1 :]]
        if spread < 5 and min(gaps) > 0.5 and np.min(np.linalg.norm(xy, axis=1)) > 0.3:
            break
    loads_xy = rng.uniform(-3, 3, (int(rng.integers(1, 4)), 2))
    plan = np.vstack([[0, 0], xy, loads_xy])
    points = np.column_stack([plan, np.zeros(len(plan))])
    members = [(0, k) if rng.random() < 0.5 else (k, 0) for k in range(1, len(points))]
    frame = Frame(points, members)
    frame.add_random_loads(rng, nodes=np.arange(len(points)))

    m = em.Model()
    m.body("plate")
    springs, compare = [], {}
    for s in range(n):
        k = float(1e6 * rng.uniform(0.3, 3))
        m.support(f"K{s}", em.Roller(at=points[1 + s], normal="+z", stiffness=k), body="plate")
        springs.append((1 + s, "DZ", k))
        compare[f"K{s}"] = (f"N{1 + s}", [2])
    m.support("C", em.Custom(at=points[0], constrain=["Fx", "Fy", "Mz"]), body="plate")
    compare["C"] = ("N0", [0, 1, 5])
    one_body(m, frame, "plate")
    return ElasticCase(frame, False, m, compare, springs, restraints=[(0, "DX DY RZ")])


def elastic_bolt_group(seed: int) -> ElasticCase:
    """A plate held only by 3-6 bolts (pins with translational stiffness, some
    stiffer in one direction) under an eccentric in-plane load: the elastic
    method for bolt groups. The FE plate is a star of stiff members from the
    bolt group's centroid to every bolt and load point."""
    rng = np.random.default_rng(seed)
    while True:
        n = int(rng.integers(3, 7))
        xy = rng.uniform(-0.3, 0.3, (n, 2))
        gaps = [np.linalg.norm(a - b) for i, a in enumerate(xy) for b in xy[i + 1 :]]
        centre = xy.mean(axis=0)
        if min(gaps) > 0.08 and np.min(np.linalg.norm(xy - centre, axis=1)) > 0.03:
            break
    load_at = centre + 0.3 * planar_unit(rng)[:2] * rng.uniform(1, 3)
    plan = np.vstack([centre, xy, load_at])
    points = np.column_stack([plan, np.zeros(len(plan))])
    members = [(0, k) if rng.random() < 0.5 else (k, 0) for k in range(1, len(points))]
    frame = Frame(points, members)
    frame.node_loads.append((n + 1, 1e4 * planar_unit(rng), np.array([0, 0, 1e3 * rng.normal()])))
    frame.add_random_loads(rng, nodes=np.arange(len(points)), planar=True)

    m = em.Model(planar=True)
    m.body("plate")
    springs, compare = [], {}
    for s in range(n):
        kx = float(1e8 * rng.uniform(0.3, 3))
        ky = kx if rng.random() < 0.5 else float(kx * rng.uniform(0.2, 5))
        pin = em.Pin(at=xy[s], stiffness={"translational": [kx, ky, kx]})
        m.support(f"B{s}", pin, body="plate")
        springs += [(1 + s, "DX", kx), (1 + s, "DY", ky)]
        compare[f"B{s}"] = (f"N{1 + s}", [0, 1])
    one_body(m, frame, "plate", planar=True)
    return ElasticCase(frame, True, m, compare, springs)


def elastic_links(seed: int) -> ElasticCase:
    """A rigid block held only by 7-10 elastic links in random directions (a
    redundant hexapod), under random 3D loads. The FE block is a star of stiff
    members from its centre to every attachment and load point; each link is
    an axial spring element to a fixed ground node."""
    rng = np.random.default_rng(seed)
    while True:
        n_att = int(rng.integers(3, 6))
        att = rng.uniform(-1.5, 1.5, (n_att, 3))
        gaps = [np.linalg.norm(a - b) for i, a in enumerate(att) for b in att[i + 1 :]]
        if min(gaps) < 0.5 or np.min(np.linalg.norm(att, axis=1)) < 0.3:
            continue
        on = [*range(n_att), *rng.integers(n_att, size=int(rng.integers(7, 11)) - n_att)]
        dirs = [unit_vector(rng) for _ in on]
        pairs = list(zip(on, dirs, strict=True))
        lines = np.array([np.concatenate([d, np.cross(att[a], d)]) for a, d in pairs])
        if np.linalg.cond(lines) < 10:  # the links hold the block firmly in all six DOF
            break
    loads_at = rng.uniform(-2, 2, (int(rng.integers(1, 4)), 3))
    points = np.vstack([np.zeros(3), att, loads_at])
    members = [(0, k) if rng.random() < 0.5 else (k, 0) for k in range(1, len(points))]
    frame = Frame(points, members)
    frame.add_random_loads(rng, nodes=np.arange(len(points)))

    m = em.Model()
    m.body("block")
    links, compare = [], {}
    for s, (a, d) in enumerate(pairs):
        k = float(1e6 * rng.uniform(0.3, 3))
        p = points[1 + a]
        m.support(f"L{s}", em.Link(at=p, anchor=p - d, stiffness=k), body="block")
        links.append((1 + a, d, k))
        compare[f"L{s}"] = (f"L{s}_ground", [0, 1, 2])
    one_body(m, frame, "block")
    return ElasticCase(frame, False, m, compare, links=links)


ELASTIC_CASES = {
    "beam": elastic_beam,
    "plate": elastic_plate,
    "bolt_group": elastic_bolt_group,
    "links": elastic_links,
}


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", ELASTIC_CASES)
def test_elastic_supports_match_stiff_fe_model(kind, seed):
    """engmech's least-work reactions are the rigid-body limit of the FE model.

    The difference must fall in proportion to 1/rho (first-order convergence,
    the signature of member flexibility being the only thing that differs) and
    stay below C_MAX / rho at every ratio."""
    rhos = np.array(RHOS)
    diffs = np.array([cmp.worst[1] for cmp in compare_elastic(ELASTIC_CASES[kind](seed))])
    order = -np.polyfit(np.log10(rhos), np.log10(diffs), 1)[0]
    assert 0.95 < order < 1.05, f"convergence order {order:.2f}: {diffs}"
    assert np.all(diffs * rhos < C_MAX), diffs * rhos
