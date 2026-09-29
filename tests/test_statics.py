"""Statics verification: textbook cases, diagnostics, and invariance properties."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import engmech as em
from engmech.errors import InputError
from engmech.spatial import rotation_about_axis


def solve(model):
    results = model.solve()
    for case in results.cases.values():
        assert case.verified, f"equilibrium check failed: {case.max_residual}"
    return results


# --------------------------------------------------------------------------- textbook


def test_couple_is_a_free_vector():
    """Regression for the original bug: an applied couple away from the support
    must not create an extra r x M moment."""
    m = em.Model()
    m.support("A", em.Fixed(at=[0, 0, 0]))
    m.load(em.Force([0, 0, -100], at=[1, 0, 0]))
    m.load(em.Moment([0, -10, 0], at=[1, 0, 0]))
    A = solve(m).primary["A"]
    np.testing.assert_allclose(A.force, [0, 0, 100], atol=1e-12)
    np.testing.assert_allclose(A.moment, [0, -90, 0], atol=1e-12)


def test_nearly_coincident_points_do_not_break_scaling():
    """Found by hypothesis: a load 1e-38 m from the support once lost a couple."""
    m = em.Model()
    m.support("A", em.Fixed(at=[0, 0, 0]))
    m.load(em.Force([0, 0, 1], at=[0, 0, 1.175494351e-38]))
    m.load(em.Moment([0, 0, 1]))
    A = solve(m).primary["A"]
    np.testing.assert_allclose(A.wrench, [0, 0, -1, 0, 0, -1], atol=1e-12)


def test_small_force_next_to_a_huge_couple_survives():
    m = em.Model()
    m.support("A", em.Fixed(at=[0, 0, 0]))
    m.load(em.Force([0, 0, 1], at=[1, 0, 0]))
    m.load(em.Moment([0, 0, 1e11]))
    A = solve(m).primary["A"]
    assert A.force[2] == pytest.approx(-1)


def test_two_bodies_welded_in_series():
    """Regression for the original iterative assembly solver."""
    m = em.Model()
    m.body("rod")
    m.body("disc")
    m.support("ground", em.Fixed(at=[0, 0, 0]), body="rod")
    m.joint("rd", em.Fixed(at=[1, 0, 0]), bodies=("rod", "disc"))
    m.load(em.Force([-10, 0, -10], at=[2, 0, 0]), body="disc")
    m.load(em.Moment([23, 0, 0]), body="disc")
    r = solve(m).primary
    np.testing.assert_allclose(r["ground"].wrench, [10, 0, 10, -23, -20, 0], atol=1e-12)
    np.testing.assert_allclose(r["rd"].wrench, [10, 0, 10, -23, -10, 0], atol=1e-12)


def test_inclined_roller():
    """Beam on a pin and a roller on a 30° incline, 10 kN at midspan."""
    m = em.Model(planar=True, units="SI-kN")
    m.support("A", em.Pin(at=[0, 0]))
    m.support("B", em.Roller(at=[4, 0], normal=[-0.5, np.sqrt(3) / 2]))
    m.load(em.Force([0, -10], at=[2, 0]))
    r = solve(m).primary
    N = r["B"].scalars["N"]
    assert pytest.approx(5000) == N * 0.8660254037844386  # moments about A
    assert r["A"].force[0] == pytest.approx(0.5 * N)


def test_distributed_load_resultants():
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0]))
    m.support("B", em.Roller(at=[3, 0], normal="+y"))
    m.load(em.DistributedLoad([0, 0], [3, 0], {"start": 0, "end": 6}, "-y"))
    r = solve(m).primary
    # total 9 N at x = 2
    assert r["B"].scalars["N"] == pytest.approx(6)
    assert r["A"].force[1] == pytest.approx(3)


def test_projected_distributed_load():
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0]))
    m.support("B", em.Roller(at=[4, 3], normal="+y"))
    m.load(em.DistributedLoad([0, 0], [4, 3], 2, "-y", projected=True))
    r = solve(m).primary
    # 2 N/m over a horizontal projection of 4 m = 8 N, centred at x = 2
    assert r["A"].force[1] + r["B"].force[1] == pytest.approx(8)
    assert r["B"].scalars["N"] == pytest.approx(4)


def test_unknown_force_and_moment():
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0]))
    m.load(em.Force([0, -100], at=[2, 0]))
    m.load(em.UnknownLoad("P", at=[1, 0], direction="+y"))
    r = solve(m).primary
    assert r["P"].scalars["P"] == pytest.approx(200)
    m2 = em.Model(planar=True)
    m2.support("A", em.Pin(at=[0, 0]))
    m2.load(em.Force([0, -100], at=[2, 0]))
    m2.load(em.UnknownLoad("M", axis="+z"))
    assert solve(m2).primary["M"].scalars["M"] == pytest.approx(200)


def test_truss_with_particles_and_links():
    m = em.Model(planar=True)
    for name, xy in {"A": [0, 0], "B": [4, 0], "C": [2, 2]}.items():
        m.point(name, xy)
        m.body(name, particle=True)
    m.support("RA", em.Pin(at="A"), body="A")
    m.support("RB", em.Roller(at="B", normal="+y"), body="B")
    for a, b in ("AB", "AC", "BC"):
        m.joint(a + b, em.Link(ends=[a, b]), bodies=(a, b))
    m.load(em.Force([0, -10], at="C"), body="C")
    r = solve(m).primary
    assert r["AC"].scalars["T"] == pytest.approx(-5 * np.sqrt(2))
    assert r["AB"].scalars["T"] == pytest.approx(5)


def test_slider_and_actuator_in_3d():
    m = em.Model()
    m.support("S", em.Slider(at=[0, 0, 0], axis="+x", actuated=True))
    m.load(em.Force([7, 0, -3], at=[1, 0, 0]))
    r = solve(m).primary["S"]
    assert r.scalars["drive"] == pytest.approx(-7)
    np.testing.assert_allclose(r.force, [-7, 0, 3], atol=1e-12)
    np.testing.assert_allclose(r.moment, [0, -3, 0], atol=1e-12)


def test_custom_joint_in_rotated_frame():
    """Custom joints holding only their local x, along (1, 1, 0) and (1, -1, 0)."""
    m = em.Model()
    m.support("A", em.Ball(at=[0, 0, 0]))
    m.support(
        "B", em.Custom(at=[0, 0, 1], constrain=["Fx"], orientation={"x": [1, 1, 0], "z": [0, 0, 1]})
    )
    m.support(
        "C",
        em.Custom(at=[0, 0, 1], constrain=["Fx"], orientation={"x": [1, -1, 0], "z": [0, 0, 1]}),
    )
    m.load(em.Force([10, 0, 0], at=[0, 0, 2]))
    r = solve(m)
    # moments about A: the pair at z = 1 must supply Fx = -20; then A takes +10
    np.testing.assert_allclose(r.primary["B"].force, [-10, -10, 0], atol=1e-9)
    np.testing.assert_allclose(r.primary["C"].force, [-10, 10, 0], atol=1e-9)
    np.testing.assert_allclose(r.primary["A"].force, [10, 0, 0], atol=1e-9)
    assert r.analysis.degrees_of_freedom == 1  # spin about z is free but not loaded


def test_parallel_supports_are_a_mechanism():
    """Two custom joints that both end up holding (1, 1, 0) cannot resist My."""
    m = em.Model()
    m.support("A", em.Ball(at=[0, 0, 0]))
    m.support(
        "B", em.Custom(at=[0, 0, 1], constrain=["Fx"], orientation={"x": [1, 1, 0], "z": [0, 0, 1]})
    )
    m.support(
        "C",
        em.Custom(at=[0, 0, 1], constrain=["Fy"], orientation={"x": [1, -1, 0], "z": [0, 0, 1]}),
    )
    m.load(em.Force([10, 0, 0], at=[0, 0, 2]))
    assert m.solve().status == "unbalanced"


def test_universal_joint_transmits_torque_about_cross_axis():
    m = em.Model()
    m.support("U", em.Universal(at=[0, 0, 0], axes=["+x", "+y"]))
    m.load(em.Moment([0, 0, 5]))
    m.load(em.Force([0, 0, -1], at=[0, 0, 1]))
    r = solve(m).primary["U"]
    np.testing.assert_allclose(r.moment, [0, 0, -5], atol=1e-12)


def test_load_combinations_are_linear():
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0]))
    m.support("B", em.Roller(at=[5, 0], normal="+y"))
    m.load(em.Force([0, -10], at=[1, 0], case="D"))
    m.load(em.Force([0, -4], at=[4, 0], case="L"))
    m.combination("U", {"D": 1.2, "L": 1.6})
    r = solve(m)
    expected = 1.2 * r["D"]["A"].force + 1.6 * r["L"]["A"].force
    np.testing.assert_allclose(r["U"]["A"].force, expected)


# --------------------------------------------------------------------------- diagnostics


def test_indeterminate_components_are_flagged_not_guessed():
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0]))
    m.support("B", em.Pin(at=[4, 0]))
    m.load(em.Force([0, -100], at=[1, 0]))
    r = m.solve()
    case = r.primary
    assert case.status == "indeterminate"
    assert case["A"].component("Fx") is None
    assert case["B"].component("Fx") is None
    assert case["A"].component("Fy") == pytest.approx(75)
    assert case["B"].component("Fy") == pytest.approx(25)
    assert r.analysis.degree_of_indeterminacy == 1
    assert case.verified


def test_stiffness_resolves_indeterminacy():
    """Rigid bar on three equal springs: the elastic distribution."""
    m = em.Model(planar=True)
    for i, x in enumerate([0, 1, 2]):
        m.support(f"S{i}", em.Roller(at=[x, 0], normal="+y", stiffness=1000))
    m.support("H", em.Roller(at=[0, 0], normal="+x"))
    m.load(em.Force([0, -30], at=[0.5, 0]))
    r = solve(m).primary
    # rigid bar: deflection linear in x; least squares fit of loads
    # equilibrium: sum N = 30, sum N x = 15 -> N = a + b x with 3a + 3b = 30, 3a + 5b = 15
    b = -7.5
    a = 10 - b
    for i, x in enumerate([0, 1, 2]):
        assert r[f"S{i}"].scalars["N"] == pytest.approx(a + b * x)


def test_mechanism_is_detected_and_described():
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[1, 2]))
    m.load(em.Force([0, -10], at=[3, 2]))
    r = m.solve()
    assert r.status == "unbalanced"
    motion = r.primary.excited_modes[0][0]
    assert motion.kind == "rotation"
    np.testing.assert_allclose(motion.point[:2], [1, 2], atol=1e-9)
    assert not r.primary.verified


def test_balanced_mechanism_is_still_solved():
    m = em.Model(planar=True)
    m.support("A", em.Roller(at=[0, 0], normal="+y"))
    m.support("B", em.Roller(at=[4, 0], normal="+y"))
    m.load(em.Force([0, -100], at=[1, 0]))
    r = solve(m)
    assert r.status == "ok"
    assert r.analysis.degrees_of_freedom == 1
    assert "mechanism" in r.notes[0]
    assert r.primary["A"].scalars["N"] == pytest.approx(75)


def test_one_sided_supports_warn():
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0]))
    m.support("B", em.Contact(at=[2, 0], normal="+y"))
    m.load(em.Force([0, 10], at=[1, 0]))
    assert any("lift off" in w for w in solve(m).primary.warnings)

    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0]))
    m.support("C", em.Cable(at=[2, 0], anchor=[2, 2]))
    m.load(em.Force([0, 10], at=[1, 0]))
    assert any("slack" in w for w in solve(m).primary.warnings)


def test_planar_rejects_out_of_plane_input():
    m = em.Model(planar=True)
    m.support("A", em.Fixed(at=[0, 0]))
    m.load(em.Force([0, 0, 5], at=[1, 0, 0]))
    with pytest.raises(InputError, match="xy-plane"):
        m.solve()
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0], axis="+x"))
    with pytest.raises(InputError, match="z-axis"):
        m.solve()


def test_spatial_pin_needs_an_axis():
    m = em.Model()
    m.support("A", em.Pin(at=[0, 0, 0]))
    with pytest.raises(InputError, match="needs an 'axis'"):
        m.solve()


def test_particle_rejects_couples():
    m = em.Model()
    m.body("p", particle=True)
    m.support("A", em.Ball(at=[0, 0, 0]), body="p")
    m.load(em.Moment([0, 0, 1]), body="p")
    with pytest.raises(InputError, match="particle cannot take a couple"):
        m.solve()


def test_body_must_be_named_when_ambiguous():
    m = em.Model()
    m.body("a")
    m.body("b")
    m.load(em.Force([0, 0, 1], at=[0, 0, 0]))
    with pytest.raises(InputError, match="say which body"):
        m.solve()


# --------------------------------------------------------------------------- properties

finite = st.floats(-10, 10, allow_nan=False, allow_infinity=False)
vec3 = st.tuples(finite, finite, finite).map(np.array)


@settings(max_examples=80, deadline=None)
@given(
    forces=st.lists(st.tuples(vec3, vec3), min_size=1, max_size=5),
    couples=st.lists(vec3, max_size=2),
    support=vec3,
)
def test_fixed_support_cancels_the_resultant(forces, couples, support):
    m = em.Model()
    m.support("A", em.Fixed(at=support))
    for f, p in forces:
        m.load(em.Force(f, at=p))
    for c in couples:
        m.load(em.Moment(c))
    A = solve(m).primary["A"]
    F = sum(f for f, _ in forces)
    M = sum(np.cross(p - support, f) for f, p in forces) + sum(couples, np.zeros(3))
    scale = 1 + sum(np.linalg.norm(f) * (1 + np.linalg.norm(p)) for f, p in forces)
    np.testing.assert_allclose(A.force, -F, atol=1e-9 * scale)
    np.testing.assert_allclose(A.moment, -M, atol=1e-9 * scale)


@settings(max_examples=60, deadline=None)
@given(
    anchors=st.lists(vec3, min_size=6, max_size=6),
    attach=st.lists(vec3, min_size=6, max_size=6),
    load=vec3,
    at=vec3,
)
def test_six_links_match_an_independent_solve(anchors, attach, load, at):
    """A body held by six arbitrary links: compare with a hand-built 6x6 system."""
    dirs = []
    for a, p in zip(anchors, attach, strict=True):
        d = a - p
        assume(np.linalg.norm(d) > 0.5)
        dirs.append(d / np.linalg.norm(d))
    A = np.array([np.concatenate([d, np.cross(p, d)]) for d, p in zip(dirs, attach, strict=True)]).T
    assume(np.linalg.cond(A) < 1e6)
    b = -np.concatenate([load, np.cross(at, load)])
    expected = np.linalg.solve(A, b)

    m = em.Model()
    for i, (a, p) in enumerate(zip(anchors, attach, strict=True)):
        m.support(f"L{i}", em.Link(at=p, anchor=a))
    m.load(em.Force(load, at=at))
    r = solve(m).primary
    got = np.array([r[f"L{i}"].scalars["T"] for i in range(6)])
    np.testing.assert_allclose(got, expected, atol=1e-7 * (1 + np.abs(expected).max()))


def _planar_frame(scale_len: float, length_unit: str, force_unit: str, f_factor: float):
    m = em.Model(planar=True, units={"length": length_unit, "force": force_unit})
    m.body("left")
    m.body("right")
    m.support("A", em.Pin(at=[0, 0]), body="left")
    m.support("B", em.Pin(at=[6 * scale_len, 0]), body="right")
    m.joint("C", em.Pin(at=[3 * scale_len, 4 * scale_len]), bodies=("left", "right"))
    m.load(em.Force([0, -12 * f_factor], at=[1.5 * scale_len, 2 * scale_len]), body="left")
    m.load(em.Moment(5 * f_factor * scale_len), body="right")
    return m


def test_results_do_not_depend_on_units():
    a = solve(_planar_frame(1, "m", "kN", 1)).primary
    b = solve(_planar_frame(1000, "mm", "N", 1000)).primary
    for name in ("A", "B", "C"):
        np.testing.assert_allclose(a[name].wrench, b[name].wrench, rtol=1e-10, atol=1e-6)


def _two_body_structure(point, direction) -> em.Model:
    """A statically determinate two-body structure; ``point`` and ``direction``
    map the base geometry (identity, or a rigid motion of the whole model)."""
    m = em.Model()
    m.body("a")
    m.body("b")
    m.support("S", em.Ball(at=point([0, 0, 0])), body="a")
    m.support("L1", em.Link(at=point([2, 0, 0]), anchor=point([2, 0, 3])), body="a")
    m.support("L2", em.Link(at=point([2, 0, 0]), anchor=point([2, 3, 0])), body="a")
    m.support("L3", em.Link(at=point([0, 1, 0]), anchor=point([0, 1, 3])), body="a")
    m.joint("H", em.Pin(at=point([2, 1, 0]), axis=direction([0, 0, 1])), bodies=("a", "b"))
    m.support("R", em.Roller(at=point([4, 1, 0]), normal=direction([0, 1, 0])), body="b")
    m.load(em.Force(direction([1, -2, -5]), at=point([3, 1, 0.5])), body="b")
    m.load(em.Force(direction([0, 0, -3]), at=point([1, 0.5, 0])), body="a")
    m.load(em.Moment(direction([0.5, 0, 0])), body="a")
    return m


@settings(max_examples=40, deadline=None)
@given(axis=vec3, angle=st.floats(0, 6.3), shift=vec3)
def test_rigid_motion_of_the_whole_model(axis, angle, shift):
    """Rotating and translating the whole model rotates the reactions with it."""
    assume(np.linalg.norm(axis) > 0.1)
    R = rotation_about_axis(axis, angle)

    def same(v):
        return np.asarray(v, float)

    base = solve(_two_body_structure(same, same))
    assert base.analysis.degree_of_indeterminacy == 0
    assert base.analysis.degrees_of_freedom == 0
    moved = solve(_two_body_structure(lambda p: R @ same(p) + shift, lambda d: R @ same(d)))
    for name in ("S", "L1", "L2", "L3", "H", "R"):
        np.testing.assert_allclose(
            moved.primary[name].force, R @ base.primary[name].force, atol=1e-8
        )
        np.testing.assert_allclose(
            moved.primary[name].moment, R @ base.primary[name].moment, atol=1e-8
        )


@settings(max_examples=40, deadline=None)
@given(
    bolts=st.lists(st.tuples(finite, finite), min_size=3, max_size=8, unique=True),
    load=st.tuples(finite, finite),
    at=st.tuples(finite, finite),
)
def test_bolt_group_elastic_method(bolts, load, at):
    """Equal-stiffness bolts under a rigid plate reproduce the elastic method."""
    pts = np.array(bolts)
    assume(np.min(np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(len(pts)) * 9) > 0.2)
    c = pts.mean(axis=0)
    r = pts - c
    J = float(np.sum(r**2))
    assume(J > 0.5)
    P = np.array(load)
    e = np.array(at) - c
    M = e[0] * P[1] - e[1] * P[0]
    expected = -P / len(pts) - (M / J) * np.column_stack([-r[:, 1], r[:, 0]])

    m = em.Model(planar=True)
    for i, p in enumerate(pts):
        m.support(f"b{i}", em.Pin(at=p, stiffness={"translational": 1e6}))
    m.load(em.Force(P, at=at))
    res = solve(m).primary
    got = np.array([res[f"b{i}"].force[:2] for i in range(len(pts))])
    np.testing.assert_allclose(
        got, expected, atol=1e-8 * (1 + np.abs(P).sum() * (1 + np.abs(e).sum()))
    )
