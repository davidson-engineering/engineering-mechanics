"""Regressions for bugs found in review, each checked against a hand calculation."""

import math

import numpy as np
import pytest
from click.testing import CliRunner

import engmech as em
from engmech.cli import main
from engmech.errors import InputError
from engmech.io.loader import ModelFileError, loads_model
from engmech.units import Context, UnitSystem, evaluate_parameters


def ctx(params=None, units=None):
    u = UnitSystem.from_spec(units)
    return Context(u, evaluate_parameters(params or {}, units=u))


def test_bare_number_added_to_an_angle_uses_the_angle_unit():
    c = ctx({"theta": "30 deg"})
    assert math.degrees(c.scalar("90 - theta", "angle")) == pytest.approx(60)
    m = em.Model(planar=True, parameters={"theta": "30 deg"})
    m.support("A", em.Fixed(at=[0, 0]))
    m.load(em.Force({"magnitude": "10 N", "angle": "90 - theta"}, at=[0, 0]))
    np.testing.assert_allclose(m.solve().primary["A"].force, [-5, -10 * np.sqrt(3) / 2, 0])


def test_bare_number_in_a_sum_prefers_the_kind_being_parsed():
    c = ctx({"k0": "2 kN/mm"}, {"system": "SI", "stiffness": "kN/mm"})
    assert c.scalar("k0 + 1", "stiffness") == pytest.approx(3e6)


def test_hertz_is_not_accepted_as_an_angular_velocity():
    with pytest.raises(InputError, match="Hz is ambiguous"):
        ctx().scalar("5 Hz", "angular_velocity")
    assert ctx().scalar("300 rpm", "angular_velocity") == pytest.approx(10 * math.pi)


def test_parenthesised_scalars_are_not_vectors():
    c = ctx({"P": "2 kN", "Q": "3 kN", "a": "2 m"})
    assert c.scalar("(P + Q)*a", "moment") == pytest.approx(10_000)
    m = em.Model(planar=True, parameters={"t0": "30 deg"})
    m.support("A", em.Fixed(at=[0, 0]))
    m.load(em.Force({"magnitude": 1, "direction": "(t0 + 15 deg)"}, at=[0, 0]))
    F = -m.solve().primary["A"].force
    assert math.degrees(math.atan2(F[1], F[0])) == pytest.approx(45)


@pytest.mark.parametrize("text", ["asin(2)", "sqrt(-1 m^2)", "(-8)**0.5"])
def test_math_domain_errors_are_input_errors(text):
    with pytest.raises(InputError, match=r"undefined|negative"):
        ctx().quantity(text)


def test_planar_slider_motion_is_not_flagged():
    m = em.Model(planar=True)
    m.body("block", mass=2, cog=[0, 0], motion=em.Motion(acceleration="[5, 0] m/s^2"))
    m.support("S", em.Slider(at=[0, 0], axis="+x"))
    m.load(em.Force([10, 0], at=[0, 0]))
    r = m.solve()
    assert r.status == "ok"
    assert not any("accelerations" in n for n in r.notes)


def test_planar_axis_forms_must_be_about_z():
    m = em.Model(planar=True)
    m.support("A", em.Fixed(at=[0, 0]))
    m.load(em.Moment({"magnitude": "5 N*m", "axis": [1, 0, 1]}))
    with pytest.raises(InputError, match="z-axis"):
        m.solve()
    m = em.Model(planar=True)
    m.body(
        "bar",
        mass=2,
        cog=[1, 0],
        motion=em.Motion(angular_velocity={"magnitude": "3 rad/s", "axis": [0, 1]}, pivot=[0, 0]),
    )
    m.support("A", em.Pin(at=[0, 0]))
    with pytest.raises(InputError, match="about z"):
        m.solve()


def test_disc_by_density_needs_a_thickness():
    m = em.Model()
    m.body("wheel", shapes=[em.Cylinder(radius="0.1 m", center=[0, 0, 0], density="7850 kg/m^3")])
    with pytest.raises(InputError, match="no volume"):
        m.build()


def test_particles_reject_distributed_loads():
    m = em.Model(planar=True)
    m.body("n", particle=True)
    m.support("A", em.Pin(at=[2, 0]), body="n")
    m.load(em.DistributedLoad([2, 0], [3, 0], 1, "-y"), body="n")
    with pytest.raises(InputError, match="distributed load"):
        m.solve()


def test_stiffness_components_with_arithmetic_take_the_bracket_unit():
    text = """\
analysis: planar
units: SI-mm
supports:
  A: {type: pin, at: [0, 0], stiffness: {translational: "[2e6 - 5e5, 1e6, 1e6] N/m"}}
  B: {type: pin, at: [1000, 0], stiffness: {translational: "[1.5e6, 1e6, 1e6] N/m"}}
loads:
  - {force: [10, 0], at: [500, 0]}
"""
    r = loads_model(text).solve().primary
    # equal x-stiffness at A and B: the 10 N horizontal load splits evenly
    assert r["A"].force[0] == pytest.approx(-5)
    assert r["B"].force[0] == pytest.approx(-5)


def test_sweep_endpoint_without_unit_takes_the_other_ends(tmp_path):
    path = tmp_path / "b.yaml"
    path.write_text(
        "analysis: planar\nunits: SI-kN\nparameters: {P: 10 kN}\n"
        "supports:\n  A: {type: pin, at: [0, 0]}\n  B: {type: roller, at: [4, 0], normal: +y}\n"
        "loads:\n  - {force: [0, -P], at: [2, 0]}\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        main, ["sweep", str(path), "--param", "P=0:20 kN:3", "--output", "B.N"]
    )
    assert result.exit_code == 0, result.output
    assert "10" in result.output
    bad = CliRunner().invoke(main, ["sweep", str(path), "--param", "P=0 m:20 kN:3"])
    assert bad.exit_code == 1
    assert "Traceback" not in bad.output


def test_model_file_errors_are_input_errors():
    assert issubclass(ModelFileError, InputError)
