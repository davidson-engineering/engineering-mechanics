"""Every documented input form gives the same physics as its plain form, and
every documented input mistake is rejected with a clear message.

These tests are the evidence that the ways of writing a value in
docs/input-format.md are interpreted as documented.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import engmech as em
from engmech import mass as mp
from engmech.errors import InputError
from engmech.inputs import Resolver
from engmech.io.loader import ModelFileError, loads_model
from engmech.units import Context, UnitSystem, evaluate_parameters


def resolver(planar=False, points=None, params=None, units=None):
    u = UnitSystem.from_spec(units)
    ctx = Context(u, evaluate_parameters(params or {}, units=u))
    pts = {k: np.array(v, float) for k, v in (points or {}).items()}
    return Resolver(ctx, pts, planar)


# --------------------------------------------------------------------------- equivalences


@pytest.mark.parametrize(
    "spec",
    [
        [3, -4],
        "[3, -4] kN",
        ["3 kN", "-4 kN"],
        {"magnitude": 5, "direction": [3, -4]},
        {"magnitude": "5 kN", "direction": "[0.3 m, -0.4 m]"},
        {"magnitude": 5, "angle": -53.13010235415598},
        {"magnitude": 5, "angle": "atan2(-4, 3)"},
        {"magnitude": 5, "toward": "Q"},
        {"magnitude": 5, "along": ["O", "Q"]},
    ],
)
def test_force_forms_are_equivalent(spec):
    r = resolver(planar=True, points={"O": [0, 0, 0], "Q": [3, -4, 0]}, units="SI-kN")
    f = r.force(spec, np.zeros(3))
    np.testing.assert_allclose(f, [3000, -4000, 0], atol=1e-9)


@pytest.mark.parametrize(
    ("spec", "planar"),
    [
        (5, True),
        ("5 N*m", True),
        ([0, 0, 5], False),
        ("[0, 0, 5] N*m", False),
        ({"magnitude": 5, "axis": "+z"}, False),
        ({"magnitude": "5 N m", "axis": [0, 0, 2]}, False),
        ({"magnitude": 5, "axis": "z"}, True),
    ],
)
def test_moment_forms_are_equivalent(spec, planar):
    np.testing.assert_allclose(resolver(planar=planar).moment(spec), [0, 0, 5])


@pytest.mark.parametrize(
    "spec",
    ["+y", "y", [0, 1, 0], [0, 7, 0], "[0, 3, 0] mm", "90 deg", "pi/2 rad", {"x": None}],
)
def test_direction_forms(spec):
    r = resolver(planar=True)
    if isinstance(spec, dict):
        spec = [0, 1]  # a 2-component planar vector
    np.testing.assert_allclose(r.direction(spec), [0, 1, 0], atol=1e-15)


@pytest.mark.parametrize("spec", ["+z", "z", [0, 0, 1], [0, 0, 5], "[0, 0, 2] m"])
def test_axis_forms(spec):
    np.testing.assert_allclose(resolver().axis(spec), [0, 0, 1])


@pytest.mark.parametrize(
    "spec",
    ["B", [2, 1], [2, 1, 0], "[2000, 1000] mm", ["2 m", "1000 mm"], "(2, 1) m", "[L, L/2]"],
)
def test_position_forms(spec):
    r = resolver(planar=True, points={"B": [2, 1, 0]}, params={"L": "2 m"})
    np.testing.assert_allclose(r.position(spec), [2, 1, 0])


ROT = Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix()


@pytest.mark.parametrize(
    "spec",
    [
        {"x": ROT[:, 0].tolist(), "z": ROT[:, 2].tolist()},
        {"z": ROT[:, 2].tolist(), "y": ROT[:, 1].tolist()},
        {"x": ROT[:, 0].tolist(), "y": ROT[:, 1].tolist()},
        {"euler": [20, -35, 50], "sequence": "xyz"},
        {"euler": "[20, -35, 50] deg", "sequence": "xyz"},
        {"euler": [50, -35, 20], "sequence": "ZYX"},  # intrinsic, reversed order
        {
            "axis": Rotation.from_matrix(ROT).as_rotvec().tolist(),
            "angle": f"{np.linalg.norm(Rotation.from_matrix(ROT).as_rotvec())} rad",
        },
    ],
)
def test_orientation_forms_are_equivalent(spec):
    np.testing.assert_allclose(resolver().orientation(spec), ROT, atol=1e-12)


def test_orientation_rotates_a_box_as_expected():
    box = em.Box(
        mass=2,
        size=[0.3, 0.2, 0.1],
        center=[0, 0, 0],
        orientation={"euler": [20, -35, 50], "sequence": "xyz"},
    )
    props = box.resolve(resolver(), "box")
    local = mp.box([0.3, 0.2, 0.1], [0, 0, 0], mass=2)
    np.testing.assert_allclose(props.inertia, ROT @ local.inertia @ ROT.T, atol=1e-15)


def _mass(shapes, planar=False):
    m = em.Model(planar=planar)
    m.body("b", shapes=shapes)
    return m.build().bodies["b"].mass


def test_mass_forms_are_equivalent():
    rho = 7850.0
    by_density = _mass([em.Rod(density="2 kg/m", start=[0, 0, 0], end=[3, 0, 0])])
    by_mass = _mass([em.Rod(mass=6, start=[0, 0, 0], end=[3, 0, 0])])
    np.testing.assert_allclose(by_density.inertia, by_mass.inertia)

    tube = _mass(
        [
            em.Cylinder(
                density=rho, radius=0.05, inner_radius=0.03, length=0.2, center=[0, 0, 0], axis="+x"
            )
        ]
    )
    diff = _mass(
        [
            em.Cylinder(density=rho, radius=0.05, length=0.2, center=[0, 0, 0], axis="+x"),
            em.Cylinder(
                density=rho, radius=0.03, length=0.2, center=[0, 0, 0], axis="+x", subtract=True
            ),
        ]
    )
    assert tube.mass == pytest.approx(diff.mass)
    np.testing.assert_allclose(tube.inertia, diff.inertia, atol=1e-15)

    shell = _mass([em.Sphere(mass=3, radius=0.1, inner_radius=0.09, center=[1, 0, 0])])
    assert shell.inertia[0, 0] == pytest.approx(2 / 5 * 3 * (0.1**5 - 0.09**5) / (0.1**3 - 0.09**3))
    cone = _mass([em.Cone(mass=3, radius=0.1, height=0.4, base_center=[0, 0, 0], axis="-x")])
    np.testing.assert_allclose(cone.cog, [-0.1, 0, 0])

    # CAD values given about another point and in rotated axes
    rod = mp.rod([0, 0, 0], [1, 0, 0], mass=3)
    I_end = rod.inertia_about([0, 0, 0])
    cad = _mass([em.CustomMass(mass=3, cog=[0.5, 0, 0], inertia=I_end.tolist(), about=[0, 0, 0])])
    np.testing.assert_allclose(cad.inertia, rod.inertia, atol=1e-15)
    rotated = _mass(
        [
            em.CustomMass(
                mass=1,
                cog=[0, 0, 0],
                inertia=[1, 2, 3],
                orientation={"axis": "+z", "angle": "90 deg"},
            )
        ]
    )
    np.testing.assert_allclose(np.diag(rotated.inertia), [2, 1, 3], atol=1e-12)

    shorthand = em.Model()
    shorthand.body("b", mass=4, cog=[1, 2, 3], inertia=[1, 1, 1])
    props = shorthand.build().bodies["b"].mass
    assert props.mass == 4
    np.testing.assert_allclose(props.cog, [1, 2, 3])


@pytest.mark.parametrize(
    "stiffness",
    [
        {"translational": 1000},
        {"translational": "1 kN/m"},
        {"translational": [1000, 1000, 1000]},
        {"translational": "[1, 1, 1] N/mm"},
        {"translational": ["1 N/mm", "1000", "1 kN/m"]},
    ],
)
def test_stiffness_forms_are_equivalent(stiffness):
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0], stiffness=stiffness))
    m.support("B", em.Pin(at=[1, 0], stiffness={"translational": 1000}))
    m.load(em.Force([10, 0], at=[0.5, 0]))
    r = m.solve().primary
    assert r["A"].force[0] == pytest.approx(-5)


def test_rigid_stiffness_components():
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0], stiffness={"translational": ["rigid", 1000, 1000]}))
    m.support("B", em.Pin(at=[1, 0], stiffness={"translational": 1000}))
    m.load(em.Force([10, 0], at=[0.5, 0]))
    assert m.solve().primary["A"].force[0] == pytest.approx(-10)  # the rigid one takes it all


def test_motion_forms_are_equivalent():
    def solve(motion):
        m = em.Model(planar=True)
        m.body("bar", shapes=[em.Rod(mass=2, start=[0, 0], end=[2, 0])], motion=motion)
        m.support("O", em.Pin(at=[0, 0], actuated=True))
        return m.solve().primary["O"].wrench

    w, al = 3.0, 4.0
    a_cog = [-(w**2) * 1.0, al * 1.0]
    base = solve(em.Motion(angular_velocity=w, angular_acceleration=al, pivot=[0, 0]))
    for motion in (
        em.Motion(
            angular_velocity=f"{w} rad/s", angular_acceleration=f"{al} rad/s^2", acceleration=a_cog
        ),
        em.Motion(
            angular_velocity={"magnitude": w, "axis": "+z"},
            angular_acceleration={"magnitude": al, "axis": "z"},
            pivot=[0, 0],
            pivot_acceleration=[0, 0],
        ),
        em.Motion(
            angular_velocity=f"{w * 60 / (2 * np.pi)} rpm", angular_acceleration=al, pivot=[0, 0]
        ),
    ):
        np.testing.assert_allclose(solve(motion), base, atol=1e-12)


@pytest.mark.parametrize(
    "gravity",
    [
        "-y",
        [0, -9.80665],
        "[0, -9.80665] m/s^2",
        {"direction": "-y"},
        {"direction": [0, -2], "magnitude": "9.80665 m/s^2"},
    ],
)
def test_gravity_forms_are_equivalent(gravity):
    m = em.Model(planar=True, gravity=gravity)
    m.body("b", mass=2, cog=[0, 0])
    m.support("A", em.Fixed(at=[0, 0]))
    assert m.solve().primary["A"].force[1] == pytest.approx(2 * 9.80665)


@pytest.mark.parametrize(
    "units",
    [
        "SI-mm",
        {"length": "mm"},
        {"system": "SI", "length": "mm"},
        {"system": "SI-kN", "length": "mm", "force": "N"},
    ],
)
def test_unit_system_forms_are_equivalent(units):
    m = em.Model(planar=True, units=units)
    m.support("A", em.Pin(at=[0, 0]))
    m.support("B", em.Roller(at=[1000, 0], normal="+y"))
    m.load(em.Force([0, -10], at=[250, 0]))
    assert m.solve().primary["B"].scalars["N"] == pytest.approx(2.5)


# --------------------------------------------------------------------------- mistakes

MISTAKES = [
    # (planar, snippet under supports/loads, expected message fragment)
    (True, "supports: {A: {type: pin, at: Q}}", "unknown point 'Q'"),
    (True, "supports: {A: {type: roller, at: [0, 0], normal: [0, 0]}}", "non-zero"),
    (True, "supports: {A: {type: slider, at: [0, 0], axis: z}}", "xy-plane"),
    (True, "supports: {A: {type: roller, at: [0, 0], normal: [0, 0, 1]}}", "xy-plane"),
    (
        True,
        "supports: {A: {type: fixed, at: [0, 0]}}\nloads: [{moment: [1, 0, 0]}]",
        "about the z-axis",
    ),
    (
        False,
        "supports: {A: {type: fixed, at: [0, 0, 0]}}\nloads: [{moment: 5}]",
        "only allowed in planar",
    ),
    (
        False,
        "supports: {A: {type: fixed, at: [0, 0, 0]}}\nloads: [{moment: [1, 2]}]",
        "3 components",
    ),
    (
        True,
        "supports: {A: {type: fixed, at: [0, 0]}}\n"
        "loads: [{force: {magnitude: 1, toward: [0, 0]}, at: [0, 0]}]",
        "points coincide",
    ),
    (
        True,
        "supports: {A: {type: fixed, at: [0, 0]}}\n"
        "loads: [{force: {magnitude: 1, along: [[0, 0]]}, at: [0, 0]}]",
        "from_point",
    ),
    (
        True,
        "supports: {A: {type: fixed, at: [0, 0]}}\n"
        "loads: [{force: {magnitude: 1, angle: 1, direction: +x}, at: [0, 0]}]",
        "exactly one of",
    ),
    (
        True,
        "supports: {A: {type: fixed, at: [0, 0]}}\nloads: [{force: {direction: +x}, at: [0, 0]}]",
        "needs a magnitude",
    ),
    (
        False,
        "supports: {A: {type: fixed, at: [0, 0, 0], orientation: [1, 2]}}",
        "expected a mapping",
    ),
    (
        False,
        "supports: {A: {type: fixed, at: [0, 0, 0], orientation: {euler: [1, 2, 3], "
        "sequence: xyq}}}",
        "orientation",
    ),
    (
        False,
        "supports: {A: {type: fixed, at: [0, 0, 0], orientation: {x: +x, z: +x}}}",
        "must not be parallel",
    ),
    (
        False,
        "supports: {A: {type: fixed, at: [0, 0, 0], orientation: {spin: 3}}}",
        "expected two axes",
    ),
    (
        False,
        "supports: {A: {type: universal, at: [0, 0, 0], axes: [+x, [1, 1, 0]]}}",
        "perpendicular",
    ),
    (False, "supports: {A: {type: link, at: [0, 0, 0], anchor: [0, 0, 0]}}", "different points"),
    (False, "supports: {A: {type: custom, at: [0, 0, 0], constrain: [Fx, Fx]}}", "twice"),
    (
        False,
        "supports: {A: {type: pin, at: [0, 0, 0], axis: +z, stiffness: {translational: -1}}}",
        "positive",
    ),
    (
        False,
        "supports: {A: {type: pin, at: [0, 0, 0], axis: +z, stiffness: {linear: 1}}}",
        "unexpected linear",
    ),
    (False, "supports: {A: {type: roller, at: [0, 0, 0], normal: +z, stiffness: 0}}", "positive"),
    (
        False,
        "bodies: {b: {shapes: [{type: box, size: [1, 1, 0], center: [0, 0, 0], density: 1}]}}",
        "size must be positive",
    ),
    (
        False,
        "bodies: {b: {shapes: [{type: box, size: [1, 1, 1], center: [0, 0, 0]}]}}",
        "exactly one of mass or density",
    ),
    (
        False,
        "bodies: {b: {shapes: [{type: cylinder, radius: 1, inner_radius: 2, "
        "center: [0, 0, 0], mass: 1}]}}",
        "inner_radius",
    ),
    (
        False,
        "bodies: {b: {shapes: [{type: sphere, radius: -1, center: [0, 0, 0], mass: 1}]}}",
        "radius > inner_radius",
    ),
    (
        False,
        "bodies: {b: {shapes: [{type: cone, radius: 1, height: 0, base_center: "
        "[0, 0, 0], mass: 1}]}}",
        "positive",
    ),
    (
        False,
        "bodies: {b: {shapes: [{type: rod, start: [0, 0, 0], end: [0, 0, 0], mass: 1}]}}",
        "must differ",
    ),
    (
        False,
        "bodies: {b: {shapes: [{type: point, at: [0, 0, 0], mass: -1}]}}",
        "must not be negative",
    ),
    (
        False,
        "bodies: {b: {shapes: [{type: box, size: [1, 1, 1], center: [0, 0, 0], "
        "mass: 1, subtract: true}]}}",
        "mass must be positive",
    ),
    (
        False,
        "bodies: {b: {shapes: [{type: custom, mass: 1, cog: [0, 0, 0], "
        "inertia: [[1, 2, 0], [0, 1, 0], [0, 0, 1]]}]}}",
        "symmetric",
    ),
    (
        False,
        "bodies: {b: {shapes: [{type: custom, mass: 1, cog: [0, 0, 0], inertia: [1, 1, 5]}]}}",
        "triangle inequality",
    ),
    (False, "bodies: {b: {mass: 1}}", "needs a 'cog'"),
    (False, "bodies: {b: {cog: [0, 0, 0]}}", "need a 'mass'"),
    (False, "bodies: {b: {motion: {angular_velocity: [0, 0, 1]}}}", "needs mass properties"),
    (
        False,
        "bodies: {b: {mass: 1, cog: [0, 0, 0], motion: {acceleration: [1, 0, 0], "
        "pivot: [0, 0, 0]}}}",
        "not both",
    ),
    (
        False,
        "bodies: {b: {mass: 1, cog: [0, 0, 0], motion: {pivot_acceleration: [1, 0, 0]}}}",
        "needs a 'pivot'",
    ),
    (False, "bodies: {ground: {}}", "reserved"),
    (
        False,
        "points: {A: [0, 0, 0]}\nloads: [{force: [0, 0, 1], at: A, body: ghost}]",
        "unknown body 'ghost'",
    ),
    (False, "gravity: [0, 0, 0]", "non-zero"),
    (False, "units: {length: N}", "not a length unit"),
    (False, "units: {speed: m/s}", "unknown quantity"),
    (
        False,
        "parameters: {P: 1 kN}\nloads: [{force: [0, 0, P], at: [0, 0, 0], "
        "case: a}]\ncombinations: {a: {a: 1}}",
        "same name as a load case",
    ),
    (
        False,
        "loads: [{force: [0, 0, 1], at: [0, 0, 0]}]\ncombinations: {U: {live: 1}}",
        "unknown load case",
    ),
    (
        False,
        "supports: {A: {type: fixed, at: [0, 0, 0]}}\nchecks: [{target: A.Fq, max: 1}]",
        "unknown component",
    ),
    (
        False,
        "supports: {A: {type: fixed, at: [0, 0, 0]}}\nchecks: [{target: A.Fx}]",
        "needs 'expect'",
    ),
    (
        False,
        "supports: {A: {type: fixed, at: [0, 0, 0]}}\nchecks: [{target: A.Fx, max: 1, case: nope}]",
        "unknown case",
    ),
]


@pytest.mark.parametrize(("planar", "snippet", "message"), MISTAKES)
def test_mistakes_are_rejected_clearly(planar, snippet, message):
    text = ("analysis: planar\n" if planar else "") + snippet + "\n"
    with pytest.raises((InputError, ModelFileError)) as info:
        loads_model(text, "m.yaml").solve()
    assert message in str(info.value), str(info.value)
    assert "m.yaml" in str(info.value) or not isinstance(info.value, ModelFileError)
