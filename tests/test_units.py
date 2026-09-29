import math

import numpy as np
import pytest

from engmech.errors import InputError
from engmech.units import Context, UnitSystem, evaluate, evaluate_parameters, format_number


def si(text, kind, params=None, units=None):
    ctx = Context(UnitSystem.from_spec(units), evaluate_parameters(params or {}))
    return ctx.scalar(text, kind)


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        ("10 kN", "force", 10_000),
        ("10kN", "force", 10_000),
        ("2.5 kN*m", "moment", 2500),
        ("2 N m", "moment", 2),
        ("2 N·m", "moment", 2),
        ("250 mm", "length", 0.25),
        ("3 ft + 4 in", "length", 3 * 0.3048 + 4 * 0.0254),
        ("45 deg", "angle", math.pi / 4),
        ("45°", "angle", math.pi / 4),
        ("0.5 rad", "angle", 0.5),
        ("300 rpm", "angular_velocity", 300 * 2 * math.pi / 60),
        ("9.81 m/s^2", "acceleration", 9.81),
        ("7850 kg/m^3", "density", 7850),
        ("1 lbf", "force", 4.4482216152605),
        ("1 lb", "mass", 0.45359237),
        ("sin(30 deg) * 10 N", "force", 5.0),
        ("atan2(1 m, 1 m)", "angle", math.pi / 4),
        ("sqrt(9 m^2)", "length", 3.0),
        ("-10 kN", "force", -10_000),
        ("1e3 N", "force", 1000),
    ],
)
def test_scalar_parsing(text, kind, expected):
    assert si(text, kind) == pytest.approx(expected, rel=1e-12)


def test_bare_numbers_use_the_unit_system():
    assert si(5, "length", units={"length": "mm"}) == pytest.approx(0.005)
    assert si("5", "force", units="SI-kN") == pytest.approx(5000)
    assert si(5, "moment", units="SI-kN") == pytest.approx(5000)  # kN*m
    assert si(90, "angle") == pytest.approx(math.pi / 2)  # degrees by default
    assert si(2, "inertia", units="SI-mm") == pytest.approx(2e-6)  # kg*mm^2


def test_wrong_dimension_is_explained():
    with pytest.raises(InputError, match=r"'10 mm' is not a force.*units like N"):
        si("10 mm", "force")


def test_trig_requires_explicit_angle_units():
    with pytest.raises(InputError, match="needs an angle with units"):
        evaluate("sin(30)")


def test_unknown_name():
    with pytest.raises(InputError, match="unknown name 'Lx'"):
        evaluate("Lx/2")


def test_parameters_and_units_do_not_collide():
    p = evaluate_parameters({"m": "2 kg", "L": "1.2 m", "g": "9.80665 m/s^2"})
    assert evaluate("2*m", p).to("kg").magnitude == pytest.approx(4)
    assert evaluate("9.81 m/s^2", p).to("m/s^2").magnitude == pytest.approx(9.81)
    assert evaluate("m * g", p).to("N").magnitude == pytest.approx(19.6133)
    assert evaluate("L/2", p).to("m").magnitude == pytest.approx(0.6)
    assert evaluate("5 kN*m", p).to("N*m").magnitude == pytest.approx(5000)
    # a parameter defined after g does not change g
    assert p["g"].to("m/s^2").magnitude == pytest.approx(9.80665)


def test_parameters_reference_earlier_ones_and_overrides():
    p = evaluate_parameters({"a": "2 m", "b": "3*a"}, overrides={"a": "1 m"})
    assert p["b"].to("m").magnitude == pytest.approx(3)
    with pytest.raises(InputError, match="not defined under 'parameters'"):
        evaluate_parameters({"a": 1}, overrides={"c": 2})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([0, -10, 0], [0, -10, 0]),
        ("[0, -10, 0] kN", [0, -10_000, 0]),
        (["0", "-10 kN", "0"], [0, -10_000, 0]),
        ("0, 5 kN, 0", [0, 5000, 0]),
    ],
)
def test_vectors(value, expected):
    ctx = Context()
    np.testing.assert_allclose(ctx.vector(value, "force"), expected)


def test_parenthesised_vector_with_unit():
    np.testing.assert_allclose(Context().vector("(1, 2, 3) mm", "length"), [0.001, 0.002, 0.003])


def test_mixing_bare_and_unit_components_is_rejected():
    with pytest.raises(InputError, match="mixes bare numbers"):
        Context().vector([3, "-10 kN", 0], "force")


def test_unit_after_brackets_and_on_items_is_rejected():
    with pytest.raises(InputError, match="not both"):
        Context().vector("[1 kN, 2, 3] kN", "force")


def test_unit_systems():
    us = UnitSystem.from_spec("US-in")
    assert us.label("moment") == "lbf·in" or us.label("moment") == "lbf⋅in"
    assert us.factor("force") == pytest.approx(1 / 4.4482216152605)
    custom = UnitSystem.from_spec({"system": "SI-kN", "length": "mm", "moment": "kN*m"})
    assert custom.factor("length") == pytest.approx(1000)
    assert custom.factor("moment") == pytest.approx(1e-3)
    with pytest.raises(InputError, match="not a force unit"):
        UnitSystem(force="mm")
    with pytest.raises(InputError, match="unknown unit system"):
        UnitSystem.from_spec("imperial")


@pytest.mark.parametrize(
    ("x", "text"),
    [
        (0, "0"),
        (1e-15, "0"),
        (28.0, "28"),
        (1333.33333, "1,333"),
        (-0.75, "-0.75"),
        (0.0012346, "0.001235"),
        (12345.678, "12,346"),
        (1.5e9, "1.500e+09"),
    ],
)
def test_format_number(x, text):
    assert format_number(x) == text
