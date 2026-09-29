"""Numerical robustness and input fuzzing."""

import json

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import engmech as em
from engmech.errors import InputError
from engmech.io.loader import loads_model


@pytest.mark.parametrize("length", [1e-6, 1e-3, 1.0, 1e3, 1e5])
@pytest.mark.parametrize("load", [1e-6, 1.0, 1e6, 1e9])
def test_extreme_scales(length, load):
    """Micrometres to 100 km, micronewtons to giganewtons: B = 0.3 P - 0.1 P."""
    m = em.Model(planar=True)
    m.support("A", em.Pin(at=[0, 0]))
    m.support("B", em.Roller(at=[length, 0], normal="+y"))
    m.load(em.Force([0, -load], at=[0.3 * length, 0]))
    m.load(em.Moment(0.1 * load * length))
    r = m.solve()
    assert r.status == "ok"
    assert r.primary.verified
    assert r.primary["B"].scalars["N"] == pytest.approx(0.2 * load, rel=1e-12)


def test_large_model():
    """100 bodies welded in a chain (600 equations) solve exactly."""
    n = 100
    m = em.Model()
    for i in range(n):
        m.body(f"b{i}")
    m.support("G", em.Fixed(at=[0, 0, 0]), body="b0")
    for i in range(1, n):
        m.joint(f"J{i}", em.Fixed(at=[i, 0, 0]), bodies=(f"b{i - 1}", f"b{i}"))
    m.load(em.Force([0, 0, -1], at=[n, 0, 0]), body=f"b{n - 1}")
    r = m.solve()
    assert r.primary["G"].moment[1] == pytest.approx(-n)
    assert r.primary[f"J{n // 2}"].moment[1] == pytest.approx(-(n - n // 2))


def test_ill_conditioning_is_reported():
    def model(eps):
        m = em.Model(planar=True)
        m.support("A", em.Link(at=[0, 0], anchor=[-1, 0]))
        m.support("B", em.Link(at=[1, 0], anchor=[1, -1]))
        m.support("C", em.Link(at=[1, eps], anchor=[2, eps]))
        m.load(em.Force([0, -10], at=[0.5, 0]))
        return m.solve()

    assert model(1.0).sensitivity is None
    r = model(1e-4)
    assert r.sensitivity is not None
    assert "barely resist" in r.sensitivity
    # still exact: the reaction of the nearly-collinear pair is 10 * 0.5 / 1e-4
    assert abs(r.primary["A"].scalars["T"]) == pytest.approx(5e4, rel=1e-9)
    assert r.primary.verified


# --------------------------------------------------------------------------- fuzzing

VALUES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-5, 5),
    st.floats(-1e3, 1e3, allow_nan=False),
    st.sampled_from(
        [
            "10 kN",
            "-5",
            "+y",
            "-z",
            "x",
            "A",
            "B",
            "[0, 1]",
            "[1, 2, 3] m",
            "(1, 2) mm",
            "30 deg",
            "sin(30 deg)",
            "P*2",
            "1/0",
            "sqrt(-1)",
            "asin(2)",
            "[",
            "kN kN",
            "10 Hz",
            "rigid",
            "3 m + 2",
            "",
            "   ",
            "in",
            "1e400",
            "nan",
            "inf",
        ]
    ),
    st.lists(st.one_of(st.integers(-3, 3), st.sampled_from(["1 m", "2 kN", "A"])), max_size=4),
)
FIELDS = [
    "type",
    "at",
    "axis",
    "normal",
    "anchor",
    "ends",
    "body",
    "bodies",
    "stiffness",
    "actuated",
    "thrust",
    "constrain",
    "orientation",
    "axes",
]
JOINT_TYPES = [
    "fixed",
    "pin",
    "ball",
    "bearing",
    "slider",
    "cylindrical",
    "universal",
    "roller",
    "contact",
    "link",
    "cable",
    "custom",
    "nonsense",
]
joint = st.fixed_dictionaries(
    {"type": st.sampled_from(JOINT_TYPES)},
    optional={f: VALUES for f in FIELDS if f != "type"},
)
load = st.dictionaries(
    st.sampled_from(
        [
            "force",
            "moment",
            "at",
            "name",
            "case",
            "body",
            "unknown",
            "direction",
            "axis",
            "distributed",
        ]
    ),
    st.one_of(
        VALUES,
        st.fixed_dictionaries(
            {},
            optional={
                "magnitude": VALUES,
                "direction": VALUES,
                "angle": VALUES,
                "start": VALUES,
                "end": VALUES,
                "intensity": VALUES,
                "toward": VALUES,
            },
        ),
    ),
    max_size=5,
)
document = st.fixed_dictionaries(
    {},
    optional={
        "analysis": st.sampled_from(["planar", "spatial", "3d"]),
        "units": st.one_of(
            st.sampled_from(["SI", "SI-mm", "US-in", "metric"]),
            st.fixed_dictionaries({}, optional={"length": VALUES, "force": VALUES}),
        ),
        "parameters": st.dictionaries(
            st.sampled_from(["P", "L", "m", "in", "x y"]), VALUES, max_size=3
        ),
        "points": st.dictionaries(st.sampled_from(["A", "B", "C", "1A"]), VALUES, max_size=3),
        "gravity": VALUES,
        "bodies": st.dictionaries(
            st.sampled_from(["b", "c", "ground"]),
            st.one_of(
                st.none(),
                st.fixed_dictionaries(
                    {},
                    optional={
                        "mass": VALUES,
                        "cog": VALUES,
                        "particle": st.booleans(),
                        "shapes": st.lists(
                            st.fixed_dictionaries(
                                {
                                    "type": st.sampled_from(
                                        [
                                            "box",
                                            "rod",
                                            "cylinder",
                                            "sphere",
                                            "cone",
                                            "point",
                                            "custom",
                                            "blob",
                                        ]
                                    )
                                },
                                optional={
                                    "mass": VALUES,
                                    "density": VALUES,
                                    "size": VALUES,
                                    "center": VALUES,
                                    "radius": VALUES,
                                    "start": VALUES,
                                    "end": VALUES,
                                    "at": VALUES,
                                },
                            ),
                            max_size=2,
                        ),
                        "motion": st.fixed_dictionaries(
                            {},
                            optional={
                                "angular_velocity": VALUES,
                                "acceleration": VALUES,
                                "pivot": VALUES,
                            },
                        ),
                    },
                ),
            ),
            max_size=2,
        ),
        "supports": st.dictionaries(st.sampled_from(["A", "B", "C"]), joint, max_size=3),
        "joints": st.dictionaries(st.sampled_from(["J", "K"]), joint, max_size=2),
        "loads": st.lists(load, max_size=3),
        "combinations": st.dictionaries(
            st.sampled_from(["U"]),
            st.dictionaries(st.sampled_from(["default", "dead"]), VALUES, max_size=2),
            max_size=1,
        ),
        "checks": st.lists(
            st.fixed_dictionaries(
                {"target": st.sampled_from(["A.Fy", "B.N", "b.mass", "P", "nope"])},
                optional={"expect": VALUES, "max": VALUES},
            ),
            max_size=2,
        ),
    },
)


@settings(max_examples=400, deadline=None, suppress_health_check=list(HealthCheck))
@given(document)
def test_loader_never_crashes(doc):
    """Arbitrary model files fail only with clear input errors, never with a crash."""
    try:
        results = loads_model(json.dumps(doc, allow_nan=True)).solve()
    except InputError:
        return
    for case in results.cases.values():
        assert np.all(np.isfinite(case.solution.values))


coord = st.sampled_from([0.0, 0.0, 1.0, -1.0, 2.5, 1e-7, 3.0])
point3 = st.lists(coord, min_size=3, max_size=3)
vec3 = st.lists(st.sampled_from([0.0, 1.0, -1.0, 0.5, 1e-9, 7.0]), min_size=3, max_size=3)


@st.composite
def valid_models(draw):
    """Structurally valid spatial models with random, often degenerate geometry."""
    n_bodies = draw(st.integers(1, 3))
    bodies = [f"b{i}" for i in range(n_bodies)]
    doc: dict = {"bodies": {}}
    for b in bodies:
        spec = {}
        if draw(st.booleans()):
            spec["shapes"] = [
                {
                    "type": "box",
                    "mass": draw(st.sampled_from([1, 5])),
                    "size": [1, 1, 1],
                    "center": draw(point3),
                }
            ]
        doc["bodies"][b] = spec
    doc["gravity"] = draw(st.sampled_from([None, "-z"]))
    supports = {}
    for i in range(draw(st.integers(0, 4))):
        kind = draw(st.sampled_from(["fixed", "ball", "pin", "roller", "link", "bearing"]))
        s = {"type": kind, "body": draw(st.sampled_from(bodies))}
        if kind == "link":
            s.update(at=draw(point3), anchor=draw(point3))
        else:
            s["at"] = draw(point3)
        if kind in ("pin", "bearing"):
            s["axis"] = draw(vec3)
        if kind == "roller":
            s["normal"] = draw(vec3)
        if draw(st.integers(0, 4)) == 0 and kind != "link":
            s["stiffness"] = {"translational": 1000} if kind != "roller" else 1000
        supports[f"S{i}"] = s
    doc["supports"] = supports
    joints = {}
    for i in range(1, n_bodies):
        kind = draw(st.sampled_from(["fixed", "pin", "ball"]))
        j = {"type": kind, "bodies": [bodies[i - 1], bodies[i]], "at": draw(point3)}
        if kind == "pin":
            j["axis"] = draw(vec3)
        joints[f"J{i}"] = j
    doc["joints"] = joints
    doc["loads"] = [
        {"force": draw(vec3), "at": draw(point3), "body": draw(st.sampled_from(bodies))}
        for _ in range(draw(st.integers(0, 3)))
    ] + [{"moment": draw(vec3), "body": bodies[0]} for _ in range(draw(st.integers(0, 1)))]
    if doc["gravity"] is None:
        del doc["gravity"]
    return doc


@settings(max_examples=500, deadline=None, suppress_health_check=list(HealthCheck))
@given(valid_models())
def test_valid_models_solve_or_explain(doc):
    """Any valid model either raises a clear input error (e.g. a zero-length
    link) or solves; and whenever it is not reported as unbalanced, the
    independent equilibrium check must close."""
    try:
        results = loads_model(json.dumps(doc)).solve()
    except InputError:
        return
    for case in results.cases.values():
        assert np.all(np.isfinite(case.solution.values))
        if case.status != "unbalanced":
            assert case.verified, (case.status, case.max_residual, doc)
