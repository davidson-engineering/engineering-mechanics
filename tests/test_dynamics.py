"""Inverse dynamics (Newton-Euler through d'Alembert loads)."""

import numpy as np
import pytest

import engmech as em

G = 9.80665


def pendulum(alpha, omega=0.0, actuated=True, L=1.2, m=3.0):
    model = em.Model(planar=True, gravity="-y")
    model.body(
        "rod",
        shapes=[em.Rod(mass=m, start=[0, 0], end=[L, 0])],
        motion=em.Motion(
            angular_velocity=f"{omega} rad/s", angular_acceleration=f"{alpha} rad/s^2", pivot=[0, 0]
        ),
    )
    model.support("O", em.Pin(at=[0, 0], actuated=actuated))
    return model.solve()


def test_released_rod_needs_no_torque_at_the_free_fall_acceleration():
    """A uniform rod released from horizontal: alpha = -3g/2L, pivot force mg/4."""
    L, m = 1.2, 3.0
    r = pendulum(-3 * G / (2 * L), L=L, m=m)
    O = r.primary["O"]
    assert O.scalars["drive"] == pytest.approx(0, abs=1e-12)
    assert O.force[1] == pytest.approx(m * G / 4)
    assert O.force[0] == pytest.approx(0, abs=1e-12)


def test_unactuated_pendulum_is_a_mechanism_in_dynamic_balance():
    L = 1.2
    r = pendulum(-3 * G / (2 * L), actuated=False, L=L)
    assert r.status == "ok"
    assert r.analysis.degrees_of_freedom == 1
    wrong = pendulum(0.0, actuated=False, L=L)
    assert wrong.status == "unbalanced"


def test_centripetal_pivot_force():
    """Rod spinning at constant omega about its end: pivot pulls m w^2 L/2 inward."""
    model = em.Model(planar=True)
    model.body(
        "rod",
        shapes=[em.Rod(mass=2, start=[0, 0], end=[1, 0])],
        motion=em.Motion(angular_velocity="10 rad/s", pivot=[0, 0]),
    )
    model.support("O", em.Pin(at=[0, 0], actuated=True))
    O = model.solve().primary["O"]
    np.testing.assert_allclose(O.force, [-2 * 100 * 0.5, 0, 0], atol=1e-9)
    assert O.scalars["drive"] == pytest.approx(0, abs=1e-12)


def test_gyroscopic_moment():
    """A disc spinning about x on a shaft held in a fixed joint while the shaft
    precesses about z: the joint must supply I_s * spin * Omega about y."""
    m, r, spin, Om = 2.0, 0.1, 150.0, 1.5
    model = em.Model()
    model.body(
        "rotor",
        shapes=[em.Cylinder(mass=m, radius=r, length=0, center=[0.15, 0, 0], axis="+x")],
        motion=em.Motion(
            angular_velocity=[spin, 0, Om], angular_acceleration=[0, Om * spin, 0], pivot=[0, 0, 0]
        ),
    )
    model.support("O", em.Fixed(at=[0, 0, 0]))
    O = model.solve().primary["O"]
    Is = m * r * r / 2
    np.testing.assert_allclose(O.force, [-m * Om**2 * 0.15, 0, 0], atol=1e-9)
    assert O.moment[1] == pytest.approx(Is * spin * Om)


def test_free_body_newton_second_law():
    """An unsupported body: consistent motion balances, inconsistent does not."""

    def free(acc):
        model = em.Model()
        model.body(
            "block", mass=4, cog=[0, 0, 0], inertia=[1, 1, 1], motion=em.Motion(acceleration=acc)
        )
        model.load(em.Force([8, 0, 0], at=[0, 0, 0]))
        return model.solve()

    assert free([2, 0, 0]).status == "ok"
    assert free([3, 0, 0]).status == "unbalanced"


def test_inconsistent_motion_at_a_pin_is_flagged():
    model = em.Model(planar=True)
    model.body(
        "a",
        shapes=[em.Rod(mass=1, start=[0, 0], end=[1, 0])],
        motion=em.Motion(angular_velocity="2 rad/s", pivot=[0, 0]),
    )
    model.body("b", shapes=[em.Rod(mass=1, start=[1, 0], end=[2, 0])])  # at rest: wrong
    model.support("O", em.Pin(at=[0, 0], actuated=True), body="a")
    model.joint("J", em.Pin(at=[1, 0]), bodies=("a", "b"))
    model.support("R", em.Roller(at=[2, 0], normal="+y"), body="b")
    r = model.solve()
    assert any("different accelerations" in n for n in r.notes)


def test_consistent_two_link_motion_is_not_flagged():
    """Two links rotating together as one rigid piece."""
    w, al = 2.0, 3.0
    model = em.Model(planar=True)
    for name, a, b in (("a", 0, 1), ("b", 1, 2)):
        model.body(
            name,
            shapes=[em.Rod(mass=1, start=[a, 0], end=[b, 0])],
            motion=em.Motion(angular_velocity=w, angular_acceleration=al, pivot=[0, 0]),
        )
    model.support("O", em.Pin(at=[0, 0], actuated=True), body="a")
    model.joint("J", em.Fixed(at=[1, 0]), bodies=("a", "b"))
    r = model.solve()
    assert not any("accelerations" in n for n in r.notes)
    # torque = I_O alpha for the whole 2 m rod of 2 kg
    assert r.primary["O"].scalars["drive"] == pytest.approx(2 * 4 / 3 * al)
