"""Physics and error paths reached through the Python API (the file schema
already rejects most of these inputs before they get this far)."""

import re

import numpy as np
import pytest

import engmech as em
from engmech import mass as mp
from engmech.errors import InputError


def test_cylindrical_joint_carries_what_it_should():
    """A shaft in a long cylindrical bearing (free to slide along and turn about
    x) held axially by a link on the axis: the bearing takes the transverse
    force and the bending moments, the link the axial force; spin about x is
    free and unloaded."""
    m = em.Model()
    m.support("A", em.Cylindrical(at=[0, 0, 0], axis="+x"))
    m.support("L", em.Link(at=[1, 0, 0], anchor=[2, 0, 0]))
    m.load(em.Force([5, -12, 6], at=[0.5, 0, 0]))
    r = m.solve()
    A = r.primary["A"]
    np.testing.assert_allclose(A.force, [0, 12, -6], atol=1e-12)
    np.testing.assert_allclose(A.moment, [0, 3, 6], atol=1e-12)
    assert r.primary["L"].scalars["T"] == pytest.approx(-5)  # pushes back along -x
    assert not A.transmits[0]  # no axial force
    assert not A.transmits[3]  # no torque about the axis
    assert r.analysis.degrees_of_freedom == 1  # spin about x
    assert r.status == "ok"


def test_rotational_stiffness_shares_a_couple():
    """Two fixed supports at one point, each with a torsional spring about z:
    a couple is shared in proportion to rotational stiffness."""
    m = em.Model(planar=True)
    m.support("A", em.Fixed(at=[0, 0], stiffness={"rotational": 1000}))
    m.support("B", em.Fixed(at=[0, 0], stiffness={"rotational": 3000}))
    m.load(em.Moment(8))
    r = m.solve().primary
    assert r["A"].moment[2] == pytest.approx(-2)
    assert r["B"].moment[2] == pytest.approx(-6)


def test_motion_without_acceleration_or_pivot_is_pure_rotation_about_the_cog():
    m = em.Model()
    m.body(
        "disc",
        shapes=[em.Cylinder(mass=2, radius=0.1, center=[0, 0, 0])],
        motion=em.Motion(angular_velocity=[0, 0, 10], angular_acceleration=[0, 0, 4]),
    )
    m.support("A", em.Fixed(at=[0, 0, 0]))
    A = m.solve().primary["A"]
    np.testing.assert_allclose(A.force, 0, atol=1e-12)
    assert A.moment[2] == pytest.approx(2 * 0.01 / 2 * 4)


def _custom_body(removed_mass, removed_at, removed_inertia):
    m = em.Model()
    m.body(
        "b",
        shapes=[
            em.CustomMass(mass=10, cog=[0, 0, 0], inertia=[2, 2, 2]),
            em.CustomMass(
                mass=removed_mass,
                cog=[removed_at, 0, 0],
                inertia=[removed_inertia] * 3,
                subtract=True,
            ),
        ],
    )
    return m


def test_subtracted_custom_mass():
    props = _custom_body(1, 0.2, 0.01).build().bodies["b"].mass
    assert props.mass == pytest.approx(9)
    assert props.cog[0] == pytest.approx(-0.2 / 9)
    # Iyy about the new cog: 2 + 10(0.2/9)^2 - (0.01 + 1(0.2 - (-0.2/9))^2)
    assert props.inertia[1, 1] == pytest.approx(
        2 + 10 * (0.2 / 9) ** 2 - 0.01 - (0.2 + 0.2 / 9) ** 2
    )
    # removing more inertia than is there is caught as physically impossible
    with pytest.raises(InputError, match="positive semi-definite"):
        _custom_body(2, 1, 0.1).build()


def test_mass_edge_cases():
    assert mp.combine([]).mass == 0
    assert mp.combine([mp.point_mass(0, [0, 0, 0])]).mass == 0
    with pytest.raises(InputError, match="combined mass is zero"):
        mp.combine([mp.point_mass(1, [0, 0, 0]), mp.point_mass(1, [1, 0, 0], subtract=True)])
    assert np.isnan(mp.MassProperties(0.0).radii_of_gyration()).all()
    bad = mp.MassProperties(1.0, inertia=np.diag([-1.0, 1, 1]))
    assert any("positive semi-definite" in p for p in bad.check_physical())
    with pytest.raises(InputError, match="rotation matrix"):
        mp.box([1, 1, 1], [0, 0, 0], np.eye(3) * 2, mass=1)
    with pytest.raises(InputError, match="3 values or a 3x3"):
        mp.custom(1, [0, 0, 0], [1, 2])


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda m: m.support("A", em.Fixed()), "needs a location 'at'"),
        (lambda m: m.support("A", em.Fixed(at=[0, 0, 0], stiffness=5)), "stiffness must be"),
        (
            lambda m: m.support("A", em.Fixed(at=[0, 0, 0], stiffness={"translational": [1, 2]})),
            "one stiffness or three",
        ),
        (
            lambda m: m.support(
                "A", em.Fixed(at=[0, 0, 0], stiffness={"translational": "[1 N/m, 2, 3] N/m"})
            ),
            "units once",
        ),
        (lambda m: m.support("A", em.Universal(at=[0, 0, 0])), "needs 'axes"),
        (lambda m: m.support("A", em.Roller(at=[0, 0, 0])), "needs a 'normal'"),
        (
            lambda m: m.support(
                "A", em.Link(at=[0, 0, 0], anchor=[1, 0, 0], ends=[[0, 0, 0], [1, 0, 0]])
            ),
            "not both",
        ),
        (
            lambda m: (
                m.body("a"),
                m.body("b"),
                m.joint("J", em.Link(ends=[[0, 0, 0]]), bodies=("a", "b")),
            ),
            "'ends' must be",
        ),
        (lambda m: m.support("A", em.Link(at=[0, 0, 0])), "needs 'at'"),
        (lambda m: m.support("A", em.Custom(at=[0, 0, 0], constrain=["Fw"])), "constrain"),
        (
            lambda m: m.load(em.UnknownLoad("P", at=[0, 0, 0], direction="+x", axis="+z")),
            "give 'direction'",
        ),
        (
            lambda m: m.load(
                em.DistributedLoad([0, 0, 0], [1, 0, 0], {"start": 1, "mid": 2}, "-z")
            ),
            "intensity must be",
        ),
        (lambda m: m.load(em.DistributedLoad([0, 0, 0], [0, 0, 0], 1, "-z")), "must differ"),
        (lambda m: m.body("x", shapes=[em.PointMass(density=1, at=[0, 0, 0])]), "not 'density'"),
        (lambda m: m.body("x", shapes=[em.PointMass(mass=1)]), "needs 'mass' and 'at'"),
        (lambda m: m.body("x", shapes=[em.Rod(mass=1, start=[0, 0, 0])]), "needs 'start'"),
        (
            lambda m: m.body("x", shapes=[em.Rod(start=[0, 0, 0], end=[1, 0, 0])]),
            "mass or linear_density",
        ),
        (
            lambda m: m.body("x", shapes=[em.Rod(mass=-1, start=[0, 0, 0], end=[1, 0, 0])]),
            "must not be negative",
        ),
        (lambda m: m.body("x", shapes=[em.Box(mass=1)]), "needs 'size'"),
        (lambda m: m.body("x", shapes=[em.Cylinder(mass=1)]), "needs 'radius'"),
        (lambda m: m.body("x", shapes=[em.Sphere(mass=1)]), "needs 'radius'"),
        (lambda m: m.body("x", shapes=[em.Cone(mass=1)]), "needs 'radius'"),
        (lambda m: m.body("x", shapes=[em.CustomMass(density=1, cog=[0, 0, 0])]), "not 'density'"),
        (lambda m: m.body("x", shapes=[em.CustomMass(mass=1)]), "need 'mass' and 'cog'"),
        (
            lambda m: m.body(
                "x", shapes=[em.CustomMass(mass=1, cog=[0, 0, 0], inertia=[[1, 0], [0, 1]])]
            ),
            "3x3",
        ),
        (
            lambda m: m.body(
                "x", motion=em.Motion(angular_velocity={"speed": 1}), mass=1, cog=[0, 0, 0]
            ),
            "{magnitude, axis}",
        ),
        (
            lambda m: m.body("x", motion=em.Motion(angular_velocity=[1, 2]), mass=1, cog=[0, 0, 0]),
            "3 components",
        ),
        (
            lambda m: m.body("x", motion=em.Motion(angular_velocity=3), mass=1, cog=[0, 0, 0]),
            "3-vector",
        ),
    ],
)
def test_api_errors(build, message):
    """Mistakes are caught when the model is built (the model has one body, so
    supports and loads need not name it; joints get their own bodies)."""
    with pytest.raises(InputError, match=re.escape(message)):
        _build_and_solve(build)


def _build_and_solve(build):
    m = em.Model()
    build(m)
    return m.solve()


def test_unknown_force_needs_a_direction_and_planar_couple_axis():
    m = em.Model(planar=True)
    m.support("A", em.Fixed(at=[0, 0]))
    m.load(em.UnknownLoad("M", axis="+x"))
    with pytest.raises(InputError, match="axis must be z"):
        m.solve()
    from engmech.inputs import Resolver
    from engmech.joints import UnknownForce

    with pytest.raises(InputError, match="needs a 'direction'"):
        UnknownForce(at=[0, 0, 0]).resolve(Resolver(), "P")
