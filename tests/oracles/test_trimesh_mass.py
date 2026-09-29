"""Cross-check engmech's mass properties against trimesh.

trimesh integrates mass, centre of mass and the inertia tensor exactly over a
closed triangle mesh (divergence theorem), independently of engmech's closed
forms and parallel-axis bookkeeping. For reproducible random cases the same
solid is built twice: through engmech's public input (``em.Model`` bodies with
``shapes``, from Python and from a YAML model file in millimetres) and as a
trimesh mesh.

Conventions relied on (pinned by ``test_trimesh_inertia_convention``):

* ``mesh.mass_properties.inertia`` is the inertia tensor about the centre of
  mass in the mesh (global) axes, with off-diagonals ``-∫(x-cx)(y-cy) dm``:
  the same "true tensor" convention as engmech.
* ``trimesh.triangles.mass_properties(..., center_mass=0)`` integrates the
  tensor about the origin directly, without a parallel-axis step.

How the comparisons are judged:

* Boxes, and composites of boxes, are polyhedra: both sides are exact and must
  agree to round-off (``EXACT``).
* Curved shapes are compared against three meshes, each with half the mesh
  spacing h of the last. Tessellation error is second order in h, so the
  discrepancy must fall 4x per refinement (an engmech error would make it
  stall), must stay below a rigorous inscribed-polytope bound at the finest
  level, and must vanish after Richardson extrapolation, which leaves an O(h⁴)
  remainder. Nothing is rescaled to the exact volume.
* Through holes, pockets and bores are real boolean differences from
  manifold3d, called directly with double-precision meshes: trimesh's own
  manifold wrapper rounds vertices to float32, which caps agreement near 1e-6
  and hides the h² convergence. Closed internal cavities (spherical shells, the
  box cavity of the principal-axes cases) are meshed as the outer surface plus
  the cavity's inward-facing surface, the same boundary a boolean would give.
* engmech's slender rod is compared with a thin solid cylinder: they must differ
  by exactly the dropped radial terms, which are O((r/L)²).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pytest

import engmech as em

trimesh = pytest.importorskip("trimesh")

try:
    import manifold3d
except ImportError:  # pragma: no cover - part of the validation dependency group
    manifold3d = None

pytestmark = pytest.mark.oracle

needs_booleans = pytest.mark.skipif(
    manifold3d is None, reason="boolean differences need manifold3d (validation group)"
)

# Polyhedra: both sides exact. trimesh sums raw moments about the origin and then
# shifts them to the centre of mass, losing ~eps*(|cog|/size)^2 < 1e-12 here.
EXACT = 1e-10
SYMMETRIC = 1e-10  # a quantity that tessellation cannot change (fixed by symmetry)
POLYGON_SECTIONS = (128, 256, 512)  # sides of the polygons standing in for circles
ICOSPHERE_LEVELS = (4, 5, 6)  # subdivisions; each one halves the edge length
RATE = (3.6, 4.4)  # error ratio per halving of h of a second-order discretisation
CASES = 20

FRONTENDS = ("python", "yaml")
ORIENTATION_FORMS = ("axes", "axis_angle", "euler")
EULER_SEQUENCES = ("xyz", "zyx", "zxz", "yzy", "XYZ", "ZYX", "ZXZ", "XZX")
AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
LENGTH_FIELDS = {"size", "center", "radius", "length", "inner_radius", "height", "base_center"}


class Props(NamedTuple):
    mass: float
    cog: np.ndarray
    inertia: np.ndarray


@dataclass
class Part:
    spec: dict[str, Any]  # engmech shape input in SI, with its "type"
    mesh: Callable[[int], Any]  # tessellation level -> oracle mesh (ignored by polyhedra)


# --------------------------------------------------------------------------- oracle side


def mesh_props(mesh, density: float) -> Props:
    assert mesh.is_volume, "an oracle mesh must be a closed, consistently wound solid"
    mesh.density = density
    p = mesh.mass_properties
    return Props(float(p.mass), np.asarray(p.center_mass), np.asarray(p.inertia))


def mesh_inertia_about(mesh, density: float, point) -> np.ndarray:
    """Inertia tensor about ``point``, integrated directly (no parallel-axis step)."""
    shifted = mesh.triangles - np.asarray(point, float)
    p = trimesh.triangles.mass_properties(shifted, density=density, center_mass=np.zeros(3))
    return np.asarray(p.inertia)


def pose(rotation, origin) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = origin
    return T


def z_to(axis) -> np.ndarray:
    """A rotation taking +z to ``axis`` (trimesh's choice, not engmech's)."""
    return trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis_vector(axis))[:3, :3]


def box_mesh(size, center, rotation):
    return trimesh.creation.box(extents=size, transform=pose(rotation, center))


def cylinder_mesh(radius, length, center, axis, sections, inner_radius=0.0):
    T = pose(z_to(axis), center)
    if inner_radius:
        return trimesh.creation.annulus(
            r_min=inner_radius, r_max=radius, height=length, sections=sections, transform=T
        )
    return trimesh.creation.cylinder(radius=radius, height=length, sections=sections, transform=T)


def cone_mesh(radius, height, base_center, axis, sections):
    # trimesh's cone has its base on z = 0 and its apex at z = height
    T = pose(z_to(axis), base_center)
    return trimesh.creation.cone(radius=radius, height=height, sections=sections, transform=T)


def sphere_mesh(radius, center, level, inner_radius=0.0):
    ball = trimesh.creation.icosphere(subdivisions=level, radius=radius)
    if inner_radius:
        cavity = trimesh.creation.icosphere(subdivisions=level, radius=inner_radius)
        ball = with_cavity(ball, cavity)
    ball.apply_translation(center)
    return ball


def with_cavity(solid, cavity):
    """``solid`` less a ``cavity`` strictly inside it: the boundary is the outer
    surface plus the cavity's surface facing into the void, which is exactly what
    a boolean difference produces."""
    cavity = cavity.copy()
    cavity.invert()
    return trimesh.util.concatenate([solid, cavity])


def difference(solid, *cutters):
    """Boolean difference by manifold3d in double precision."""
    result = _manifold(solid)
    for cutter in cutters:
        result = result - _manifold(cutter)
    assert result.status() == manifold3d.Error.NoError
    out = result.to_mesh64()
    return trimesh.Trimesh(
        np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=False
    )


def _manifold(mesh):
    return manifold3d.Manifold(
        manifold3d.Mesh64(
            vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint64),
        )
    )


def union(parts: list[Part], level: int):
    """Disjoint parts: the union's boundary is the union of the boundaries."""
    return trimesh.util.concatenate([p.mesh(level) for p in parts])


def richardson(coarse: Props, fine: Props) -> Props:
    """Cancel the O(h²) error term of two results at spacings h and h/2."""
    return Props(*((4 * f - c) / 3 for c, f in zip(coarse, fine, strict=True)))


# --------------------------------------------------------------------------- engmech side


def engmech_props(specs: list[dict], frontend: str, tmp_path) -> em.MassProperties:
    if frontend == "python":
        model = em.Model("trimesh oracle")
        model.body("part", shapes=[_python_shape(s) for s in specs])
        return model.build().bodies["part"].mass
    # a model file in millimetres, loaded like any user's file
    lines = ["units: {length: mm}", "bodies:", "  part:", "    shapes:"]
    for spec in specs:
        spec = {k: (_mm(v) if k in LENGTH_FIELDS else v) for k, v in spec.items()}
        spec["density"] = f"{spec['density']!r} kg/m^3"
        lines.append(f"      - {json.dumps(spec)}")  # a YAML flow mapping (JSON is YAML 1.2)
    path = tmp_path / "part.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return em.load(path).build().bodies["part"].mass


def _python_shape(spec: dict):
    kinds = {"box": em.Box, "cylinder": em.Cylinder, "sphere": em.Sphere, "cone": em.Cone}
    return kinds[spec["type"]](**{k: v for k, v in spec.items() if k != "type"})


def _mm(value):
    return [1000.0 * v for v in value] if isinstance(value, list) else 1000.0 * value


# --------------------------------------------------------------------------- comparison


def errors(got, want: Props) -> np.ndarray:
    """Relative discrepancies: mass, cog (per radius of gyration), inertia (2-norm)."""
    size = np.linalg.norm(want.inertia, 2)
    gyration = math.sqrt(size / abs(want.mass))
    return np.array(
        [
            abs(got.mass - want.mass) / abs(want.mass),
            np.linalg.norm(got.cog - want.cog) / gyration,
            np.linalg.norm(got.inertia - want.inertia, 2) / size,
        ]
    )


def assert_exact(got, want: Props, tol: float = EXACT) -> None:
    e = errors(got, want)
    assert e.max() <= tol, f"relative mass/cog/inertia errors {e} exceed {tol:g}"


def assert_converges(got, series: list[Props], h: float, bounds=None) -> None:
    """``series``: oracle results at spacings 2h, h, h/2. The discrepancy must be
    second-order tessellation error, below ``bounds`` (mass, cog, inertia) at the
    finest level, and gone after Richardson extrapolation up to O(h⁴)."""
    err = np.array([errors(got, s) for s in series])
    for q, name in enumerate(("mass", "cog", "inertia")):
        e = err[:, q]
        if e[0] < SYMMETRIC:  # tessellation cannot change this quantity
            assert e.max() < SYMMETRIC, f"{name}: errors {e}"
            continue
        ratios = e[:-1] / e[1:]
        assert np.all((ratios > RATE[0]) & (ratios < RATE[1])), (
            f"{name}: errors {e} (ratios {ratios}) are not second-order tessellation error"
        )
        if bounds is not None:
            assert e[-1] <= bounds[q], f"{name}: error {e[-1]:g} above the bound {bounds[q]:g}"
    # the remainder is O(h⁴) with a coefficient well below 1 (1/480 for a polygon's area)
    extrapolated = errors(got, richardson(series[-2], series[-1]))
    assert extrapolated.max() <= h**4, (
        f"Richardson-extrapolated errors {extrapolated} exceed h^4 = {h**4:g}"
    )


# --------------------------------------------------------------------------- random input


def random_rotation(rng) -> np.ndarray:
    return trimesh.transformations.random_rotation_matrix(rand=rng.random(3))[:3, :3]


def random_direction(rng) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def random_orientation(rng, form: str) -> tuple[dict[str, Any], np.ndarray]:
    """An orientation in one of engmech's three input forms, and the rotation it
    denotes built independently (columns: the shape's local axes in global axes)."""
    if form == "axes":  # two exactly orthogonal local axes, of arbitrary lengths
        R = random_rotation(rng)
        pair = sorted(rng.choice(3, size=2, replace=False))
        return {"xyz"[i]: (R[:, i] * rng.uniform(0.2, 5.0)).tolist() for i in pair}, R
    if form == "axis_angle":
        axis = rng.normal(size=3) * rng.uniform(0.2, 5.0)
        angle = rng.uniform(-180.0, 180.0)  # bare angles are degrees
        R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)[:3, :3]
        value = angle if rng.random() < 0.5 else f"{math.radians(angle)!r} rad"
        return {"axis": axis.tolist(), "angle": value}, R
    seq = str(rng.choice(EULER_SEQUENCES))
    angles = rng.uniform(-180.0, 180.0, 3)
    steps = [
        trimesh.transformations.rotation_matrix(math.radians(a), AXIS_VECTORS[c.lower()])[:3, :3]
        for a, c in zip(angles, seq, strict=True)
    ]
    # lowercase is extrinsic (about the fixed axes: each step multiplies from the left),
    # uppercase intrinsic (about the rotated axes: each step multiplies from the right)
    R = steps[2] @ steps[1] @ steps[0] if seq.islower() else steps[0] @ steps[1] @ steps[2]
    return {"euler": angles.tolist(), "sequence": seq}, R


def random_axis(rng):
    """Mostly random directions of arbitrary length; sometimes a coordinate axis
    (engmech builds the frame of an axis along ±z differently)."""
    special = ("+z", "-z", "+x", [0.0, 0.0, -2.0], [0.0, 3.0, 0.0])
    if rng.random() < 0.25:
        return special[rng.integers(len(special))]
    return (random_direction(rng) * rng.uniform(0.2, 5.0)).tolist()


def axis_vector(axis) -> np.ndarray:
    if isinstance(axis, str):
        return (-1.0 if axis.startswith("-") else 1.0) * np.array(AXIS_VECTORS[axis[-1]])
    return np.asarray(axis, float) / np.linalg.norm(axis)


def grid_cells(rng, count: int) -> list[np.ndarray]:
    """Distinct cells of a 3x3x3 grid of unit spacing: balls of radius 0.45 about
    points within 0.04 of different cell centres cannot overlap."""
    cells = rng.choice(27, size=count, replace=False)
    return [np.array(np.unravel_index(c, (3, 3, 3)), float) - 1.0 for c in cells]


def box_part(size, center, orientation, R, density, subtract=False) -> Part:
    spec = {"type": "box", "size": list(map(float, size)), "center": list(map(float, center))}
    spec |= {"orientation": orientation, "density": density, "subtract": subtract}
    return Part(spec, lambda level: box_mesh(size, center, R))


def cylinder_part(radius, length, center, axis, density, inner=0.0, subtract=False) -> Part:
    spec = {"type": "cylinder", "radius": radius, "length": length}
    spec |= {"center": list(map(float, center)), "axis": axis, "inner_radius": inner}
    spec |= {"density": density, "subtract": subtract}
    return Part(spec, lambda n: cylinder_mesh(radius, length, center, axis, n, inner))


def random_box(rng, center, density, reach=0.45) -> Part:
    """A randomly sized and oriented box within ``reach`` of its centre."""
    size = rng.uniform(0.05, 0.6, 3)
    size *= min(1.0, 2 * reach / np.linalg.norm(size))
    orientation, R = random_orientation(rng, str(rng.choice(ORIENTATION_FORMS)))
    return box_part(size, center, orientation, R, density)


def random_cylinder(rng, center, density, reach=0.45) -> Part:
    """A random, sometimes hollow, cylinder within ``reach`` of its centre."""
    radius, length = rng.uniform(0.03, 0.35), rng.uniform(0.02, 0.8)
    scale = min(1.0, reach / math.hypot(radius, length / 2))
    radius, length = radius * scale, length * scale
    inner = radius * rng.uniform(0.2, 0.9) if rng.random() < 0.4 else 0.0
    return cylinder_part(radius, length, center, random_axis(rng), density, inner)


# --------------------------------------------------------------------------- 0. the oracle


def test_trimesh_inertia_convention():
    """A corner tetrahedron with unequal legs has known, distinct products of inertia.

    For vertices O, a·x, b·y, c·z: V = abc/6 and cog = (a, b, c)/4; about the cog
    ∫(x-cx)² dV = 3a²V/80 and ∫(x-cx)(y-cy) dV = -abV/80; about O ∫x² dV = a²V/10
    and ∫xy dV = abV/20.
    """
    a, b, c, rho = 1.3, 0.7, 2.1, 3.0
    tet = trimesh.Trimesh(
        [[0, 0, 0], [a, 0, 0], [0, b, 0], [0, 0, c]],
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        process=False,
    )
    m = rho * a * b * c / 6
    L = np.array([a, b, c])
    squares = np.diag((L @ L) - L * L)  # diag(b²+c², a²+c², a²+b²)
    offdiag = np.outer(L, L) - np.diag(L * L)
    central = m * (3 / 80 * squares + offdiag / 80)  # products of inertia +abV/80 > 0
    about_corner = m * (squares / 10 - offdiag / 20)

    R = trimesh.transformations.rotation_matrix(0.8, [1.0, -2.0, 0.5])[:3, :3]
    t = np.array([0.4, -1.2, 2.5])
    tet.apply_transform(pose(R, t))
    props = mesh_props(tet, rho)
    assert props.mass == pytest.approx(m, rel=1e-14)
    np.testing.assert_allclose(props.cog, R @ L / 4 + t, rtol=1e-14)
    np.testing.assert_allclose(props.inertia, R @ central @ R.T, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        mesh_inertia_about(tet, rho, t), R @ about_corner @ R.T, rtol=1e-12, atol=1e-15
    )


# --------------------------------------------------------------------------- 1. boxes


@pytest.mark.parametrize("frontend", FRONTENDS)
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("form", ORIENTATION_FORMS)
def test_oriented_box_is_exact(form, seed, frontend, tmp_path):
    rng = np.random.default_rng([1, ORIENTATION_FORMS.index(form), seed])
    size, center = rng.uniform(0.05, 2.0, 3), rng.uniform(-2.0, 2.0, 3)
    rho = rng.uniform(500.0, 9000.0)
    orientation, R = random_orientation(rng, form)
    part = box_part(size, center, orientation, R, rho)

    got = engmech_props([part.spec], frontend, tmp_path)
    assert_exact(got, mesh_props(part.mesh(0), rho))


# --------------------------------------------------------------------------- 2. curved shapes

CURVED_KINDS = ("cylinder", "tube", "cone", "sphere", "shell")


@dataclass
class Curved:
    spec: dict[str, Any]
    mesh: Callable[[int, bool], Any]  # (level, hollow) -> mesh; hollow=False: the outer solid
    center: np.ndarray
    radius: float
    icosphere: bool

    @property
    def levels(self) -> tuple[int, ...]:
        return ICOSPHERE_LEVELS if self.icosphere else POLYGON_SECTIONS

    @property
    def powers(self) -> tuple[int, int]:
        """How mass and inertia scale with the radius of the ball / cross-section."""
        return (3, 5) if self.icosphere else (2, 4)

    def spacing(self, level: int) -> float:
        """Mesh spacing relative to the radius."""
        if self.icosphere:
            return self.mesh(level, False).edges_unique_length.max() / self.radius
        return 2 * math.pi / level

    def inscribed(self, level: int) -> float:
        """Inradius / circumradius of the tessellated circle or sphere."""
        if self.icosphere:
            outer = self.mesh(level, False)
            v = outer.triangles[:, 0] - self.center
            return np.abs(np.einsum("ij,ij->i", outer.face_normals, v)).min() / self.radius
        return math.cos(math.pi / level)


def curved_case(kind: str, rng) -> Curved:
    rho = rng.uniform(500.0, 9000.0)
    center = rng.uniform(-1.0, 1.0, 3)
    radius = rng.uniform(0.02, 0.5)
    inner = radius * rng.uniform(0.2, 0.9) if kind in ("tube", "shell") else 0.0
    axis = random_axis(rng)
    c = center.tolist()
    if kind in ("cylinder", "tube"):
        length = rng.uniform(0.01, 1.0)
        spec = {"type": "cylinder", "radius": radius, "length": length, "center": c}
        spec |= {"axis": axis, "inner_radius": inner}

        def mesh(n, hollow):
            return cylinder_mesh(radius, length, center, axis, n, inner if hollow else 0.0)

    elif kind == "cone":
        height = rng.uniform(0.02, 1.0)
        spec = {"type": "cone", "radius": radius, "height": height, "base_center": c}
        spec |= {"axis": axis}

        def mesh(n, hollow):
            return cone_mesh(radius, height, center, axis, n)

    else:
        spec = {"type": "sphere", "radius": radius, "center": c, "inner_radius": inner}

        def mesh(n, hollow):
            return sphere_mesh(radius, center, n, inner if hollow else 0.0)

    spec["density"] = rho
    return Curved(spec, mesh, center, radius, icosphere=kind in ("sphere", "shell"))


@pytest.mark.parametrize("frontend", FRONTENDS)
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kind", CURVED_KINDS)
def test_curved_shape_converges_to_engmech(kind, seed, frontend, tmp_path):
    case = curved_case(kind, np.random.default_rng([2, CURVED_KINDS.index(kind), seed]))
    rho = case.spec["density"]
    got = engmech_props([case.spec], frontend, tmp_path)
    series = [mesh_props(case.mesh(n, True), rho) for n in case.levels]

    # The tessellated circle (sphere) lies between its inscribed and circumscribed
    # circles (spheres), radius ratio c. Mass and every quadratic form v·I·v about
    # the common centre grow with the solid, and shrinking the radius by c scales
    # them by at least c^p, so a solid's relative error is at most c^-p - 1. A
    # hollow shape's error is the difference of its outer and inner solids' errors,
    # so it is at most that times (outer solid) / (hollow shape).
    finest = case.levels[-1]
    outer = mesh_props(case.mesh(finest, False), rho)
    c = case.inscribed(finest)
    amplify = (
        outer.mass / series[-1].mass,
        np.linalg.norm(outer.inertia, 2) / np.linalg.norm(series[-1].inertia, 2),
    )
    p_mass, p_inertia = case.powers
    bounds = [(c**-p_mass - 1) * amplify[0], 0.0, (c**-p_inertia - 1) * amplify[1]]
    assert_converges(got, series, h=case.spacing(case.levels[-2]), bounds=bounds)


# --------------------------------------------------------------------------- 3. composites


@pytest.mark.parametrize("frontend", FRONTENDS)
@pytest.mark.parametrize("seed", range(CASES))
def test_union_of_boxes_is_exact(seed, frontend, tmp_path):
    rng = np.random.default_rng([3, seed])
    rho = rng.uniform(500.0, 9000.0)
    cells = grid_cells(rng, int(rng.integers(2, 6)))
    parts = [random_box(rng, cell + rng.uniform(-0.04, 0.04, 3), rho) for cell in cells]

    got = engmech_props([p.spec for p in parts], frontend, tmp_path)
    assert_exact(got, mesh_props(union(parts, 0), rho))


@pytest.mark.parametrize("frontend", FRONTENDS)
@pytest.mark.parametrize("seed", range(CASES))
def test_union_of_boxes_and_cylinders_converges(seed, frontend, tmp_path):
    rng = np.random.default_rng([4, seed])
    rho = rng.uniform(500.0, 9000.0)
    parts = []
    for i, cell in enumerate(grid_cells(rng, int(rng.integers(2, 6)))):
        center = cell + rng.uniform(-0.04, 0.04, 3)
        make = random_cylinder if i == 0 or rng.random() < 0.5 else random_box
        parts.append(make(rng, center, rho))

    got = engmech_props([p.spec for p in parts], frontend, tmp_path)
    series = [mesh_props(union(parts, n), rho) for n in POLYGON_SECTIONS]
    assert_converges(got, series, h=2 * math.pi / POLYGON_SECTIONS[-2])


def holed_plate(rng) -> list[Part]:
    """A randomly oriented plate less 2-3 different cut-outs (so at least one is
    round), each in its own slice along the plate's local x and fully inside the
    plate: a through hole along the plate normal, a blind rectangular pocket from
    the top face, or a blind cross-drilled bore in the plate's plane."""
    rho = rng.uniform(500.0, 9000.0)
    size = np.array([rng.uniform(0.2, 0.6), rng.uniform(0.1, 0.4), rng.uniform(0.02, 0.1)])
    half = size / 2
    center = rng.uniform(-1.0, 1.0, 3)
    orientation, R = random_orientation(rng, str(rng.choice(ORIENTATION_FORMS)))
    parts = [box_part(size, center, orientation, R, rho)]
    kinds = [str(k) for k in rng.permutation(["through", "pocket", "cross"])[: rng.integers(2, 4)]]
    width = size[0] / len(kinds)
    for i, kind in enumerate(kinds):
        x0 = -half[0] + (i + 0.5) * width  # middle of this cut-out's slice
        if kind == "pocket":
            pocket = size * [rng.uniform(0.3, 0.8), rng.uniform(0.3, 0.8), rng.uniform(0.3, 0.7)]
            pocket[0] = min(pocket[0], 0.8 * width)
            local = np.array(
                [
                    x0 + rng.uniform(-0.4, 0.4) * (width - pocket[0]),
                    rng.uniform(-0.4, 0.4) * (size[1] - pocket[1]),
                    half[2] - pocket[2] / 2,
                ]
            )
            _assert_inside(half, width, x0, local, pocket / 2)
            parts.append(box_part(pocket, center + R @ local, orientation, R, rho, subtract=True))
            continue
        if kind == "through":
            radius = rng.uniform(0.15, 0.4) * min(width, size[1])
            length = size[2]  # flush with both faces
            direction = np.array([0.0, 0.0, rng.choice([-1.0, 1.0])])
            local = np.array(
                [
                    x0 + rng.uniform(-0.4, 0.4) * (width - 2 * radius),
                    rng.uniform(-0.4, 0.4) * (size[1] - 2 * radius),
                    0.0,
                ]
            )
        else:
            phi = rng.uniform(0.0, 2 * math.pi)
            direction = np.array([math.cos(phi), math.sin(phi), 0.0])
            cos, sin = np.abs(direction[:2])
            radius = rng.uniform(0.5, 1.0) * min(0.4 * size[2], 0.2 * width, 0.2 * size[1])
            # a cylinder reaches (length/2)|a·e| + radius·sqrt(1 - (a·e)²) along e
            reach_x = (0.45 * width - radius * sin) / max(cos, 1e-12)
            reach_y = (0.9 * half[1] - radius * cos) / max(sin, 1e-12)
            length = 2 * rng.uniform(0.5, 1.0) * min(reach_x, reach_y)
            local = np.array([x0, 0.0, rng.uniform(-0.5, 0.5) * (half[2] - radius)])
        reach = np.abs(direction) * length / 2 + radius * np.sqrt(1 - direction**2)
        _assert_inside(half, width, x0, local, reach)
        axis = (R @ direction * rng.uniform(0.2, 5.0)).tolist()
        parts.append(cylinder_part(radius, length, center + R @ local, axis, rho, subtract=True))
    return parts


def _assert_inside(half, width, x0, local, reach):
    """Guard the generator: a cut-out reaching ``reach`` along the plate's local axes
    from ``local`` stays in the plate and in its own slice (so none overlap)."""
    slack = 1 + 1e-12
    assert np.all(np.abs(local) + reach <= half * slack)
    assert abs(local[0] - x0) + reach[0] <= width / 2 * slack


@needs_booleans
@pytest.mark.parametrize("frontend", FRONTENDS)
@pytest.mark.parametrize("seed", range(CASES))
def test_plate_with_holes_matches_boolean_difference(seed, frontend, tmp_path):
    rng = np.random.default_rng([5, seed])
    plate, *cutouts = holed_plate(rng)
    rho = plate.spec["density"]

    got = engmech_props([plate.spec] + [c.spec for c in cutouts], frontend, tmp_path)
    series = [
        mesh_props(difference(plate.mesh(n), *(c.mesh(n) for c in cutouts)), rho)
        for n in POLYGON_SECTIONS
    ]
    assert_converges(got, series, h=2 * math.pi / POLYGON_SECTIONS[-2])


# --------------------------------------------------------------------------- 4. principal axes


def asymmetric_composite(rng) -> tuple[list[Part], Any, float]:
    """2-4 disjoint, randomly oriented boxes, the first with a randomly oriented box
    cavity strictly inside it. Returns the parts, the oracle mesh and the density."""
    rho = rng.uniform(500.0, 9000.0)
    cells = grid_cells(rng, int(rng.integers(2, 5)))
    parts = [random_box(rng, cell + rng.uniform(-0.04, 0.04, 3), rho) for cell in cells]
    host = parts[0].spec
    inradius = min(host["size"]) / 2
    # the cavity's circumsphere (radius |size|/2) stays inside the host's insphere
    offset = random_direction(rng) * rng.uniform(0.0, 0.4) * inradius
    size = rng.uniform(0.2, 1.0, 3)
    size *= inradius / np.linalg.norm(size)  # |size|/2 = inradius/2
    orientation, R = random_orientation(rng, str(rng.choice(ORIENTATION_FORMS)))
    cavity = box_part(size, np.array(host["center"]) + offset, orientation, R, rho, subtract=True)
    mesh = with_cavity(union(parts, 0), cavity.mesh(0))
    return [*parts, cavity], mesh, rho


@pytest.mark.parametrize("seed", range(CASES))
def test_inertia_about_points_and_principal_axes(seed, tmp_path):
    rng = np.random.default_rng([6, seed])
    parts, mesh, rho = asymmetric_composite(rng)
    got = engmech_props([p.spec for p in parts], "python", tmp_path)
    want = mesh_props(mesh, rho)
    assert_exact(got, want)
    size = np.linalg.norm(want.inertia, 2)

    far = want.cog + 50 * random_direction(rng)
    for point in [np.zeros(3), *rng.uniform(-3.0, 3.0, (3, 3)), far]:
        direct = mesh_inertia_about(mesh, rho, point)
        err = np.linalg.norm(got.inertia_about(point) - direct, 2) / np.linalg.norm(direct, 2)
        assert err <= EXACT, f"inertia about {point}: relative error {err:g}"

    moments, axes = got.principal()
    tm_moments = mesh.principal_inertia_components
    tm_axes = mesh.principal_inertia_vectors.T  # trimesh returns the axes as rows
    np.testing.assert_allclose(moments, tm_moments, rtol=0, atol=EXACT * size)
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-14)
    assert np.linalg.det(axes) == pytest.approx(1.0)
    # engmech's axes diagonalise trimesh's tensor ...
    np.testing.assert_allclose(axes.T @ want.inertia @ axes, np.diag(moments), atol=EXACT * size)
    # ... and are trimesh's axes up to sign. An eigenvector moves by (perturbation /
    # gap to the other moments), so the tolerance scales with the gap.
    for i in range(3):
        gap = min(abs(tm_moments[i] - tm_moments[j]) for j in range(3) if j != i)
        assert gap > 1e-3 * size, "the random composite should have distinct moments"
        sin_angle = np.linalg.norm(np.cross(axes[:, i], tm_axes[:, i]))
        assert sin_angle <= EXACT * size / gap, f"principal axis {i} off by {sin_angle:g} rad"


# --------------------------------------------------------------------------- 5. slender rods

SLENDERNESS = (1 / 20, 1 / 40, 1 / 80, 1 / 160)  # radius / length


@pytest.mark.parametrize("seed", range(CASES))
def test_slender_rod_matches_thin_cylinder_to_second_order(seed):
    """engmech's rod drops a solid cylinder's radial terms: m r²/4 across the axis
    and m r²/2 about it. That is m r²/2 in the 2-norm against m(3r² + L²)/12: a
    relative error of 6(r/L)²/(1 + 3(r/L)²) <= 6(r/L)²."""
    rng = np.random.default_rng([7, seed])
    rho = rng.uniform(500.0, 9000.0)
    length = rng.uniform(0.5, 2.0)
    start = rng.uniform(-1.0, 1.0, 3)
    end = start + length * random_direction(rng)
    model_errors = []
    for slenderness in SLENDERNESS:
        radius = slenderness * length
        model = em.Model("rod")
        rod = em.Rod(start=start.tolist(), end=end.tolist(), density=rho * math.pi * radius**2)
        model.body("rod", shapes=[rod])
        got = model.build().bodies["rod"].mass
        # the thin cylinder, with its tessellation error extrapolated away (to ~1e-9)
        coarse, fine = (
            mesh_props(cylinder_mesh(radius, length, (start + end) / 2, end - start, n), rho)
            for n in POLYGON_SECTIONS[-2:]
        )
        e = errors(got, richardson(coarse, fine))
        assert e[0] <= 1e-8, f"rod mass differs from the cylinder's by {e[0]:g}"
        # trimesh takes first moments about the origin: a rod ~1 m away with a volume
        # of ~1e-5 m³ cancels ~eps·|x|³/V ~ 1e-11 of its cog
        assert e[1] <= 1e-9, f"rod cog differs from the cylinder's by {e[1]:g}"
        assert e[2] <= 6 * slenderness**2, f"r/L = {slenderness}: inertia error {e[2]:g}"
        model_errors.append(e[2])
    ratios = np.array(model_errors[:-1]) / model_errors[1:]
    assert np.all((ratios > 3.9) & (ratios < 4.1)), f"not second order in r/L: {ratios}"
