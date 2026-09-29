import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engmech import mass as mp
from engmech.errors import InputError
from engmech.spatial import frame_from_axes, rotation_about_axis


def test_rod_about_end_is_ml2_over_3():
    rod = mp.rod([0, 0, 0], [2, 0, 0], mass=3)
    np.testing.assert_allclose(rod.cog, [1, 0, 0])
    I_end = rod.inertia_about([0, 0, 0])
    assert I_end[1, 1] == pytest.approx(3 * 4 / 3)
    assert I_end[2, 2] == pytest.approx(3 * 4 / 3)
    assert I_end[0, 0] == pytest.approx(0)


def test_box_density_and_formula():
    b = mp.box([0.2, 0.1, 0.05], [0, 0, 0], density=7850)
    m = 7850 * 0.2 * 0.1 * 0.05
    assert b.mass == pytest.approx(m)
    np.testing.assert_allclose(
        np.diag(b.inertia),
        [m / 12 * (0.1**2 + 0.05**2), m / 12 * (0.2**2 + 0.05**2), m / 12 * (0.2**2 + 0.1**2)],
    )


def test_box_equals_sum_of_its_eight_octants():
    size = np.array([0.3, 0.2, 0.1])
    whole = mp.box(size, [1, 2, 3], mass=8.0)
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                center = np.array([1, 2, 3]) + size / 4 * [sx, sy, sz]
                parts.append(mp.box(size / 2, center, mass=1.0))
    composite = mp.combine(parts)
    assert composite.mass == pytest.approx(8.0)
    np.testing.assert_allclose(composite.cog, whole.cog)
    np.testing.assert_allclose(composite.inertia, whole.inertia, atol=1e-14)


def test_hollow_cylinder_by_subtraction():
    rho, ro, ri, h = 2700.0, 0.05, 0.03, 0.2
    solid = mp.cylinder(ro, h, [0, 0, 0], density=rho)
    core = mp.cylinder(ri, h, [0, 0, 0], density=rho, subtract=True)
    tube = mp.cylinder(ro, h, [0, 0, 0], inner_radius=ri, density=rho)
    diff = solid + core
    assert diff.mass == pytest.approx(tube.mass)
    np.testing.assert_allclose(diff.inertia, tube.inertia, atol=1e-15)


def test_plate_with_off_centre_hole_shifts_cog():
    plate = mp.box([0.4, 0.2, 0.01], [0, 0, 0], density=7850)
    hole = mp.cylinder(0.05, 0.01, [0.1, 0, 0], density=7850, subtract=True)
    part = plate + hole
    m_hole = 7850 * np.pi * 0.05**2 * 0.01
    expected_x = -m_hole * 0.1 / (plate.mass - m_hole)
    assert part.cog[0] == pytest.approx(expected_x)


def test_cylinder_default_is_solid_and_disc_limit():
    c = mp.cylinder(0.1, 0.0, [0, 0, 0], axis=[0, 0, 1], mass=2)
    np.testing.assert_allclose(np.diag(c.inertia), [2 * 0.01 / 4, 2 * 0.01 / 4, 2 * 0.01 / 2])


def test_sphere_and_shell():
    s = mp.sphere(0.1, [0, 0, 0], mass=5)
    assert s.inertia[0, 0] == pytest.approx(2 / 5 * 5 * 0.01)
    shell = mp.sphere(0.1, [0, 0, 0], inner_radius=0.0999999, mass=5)
    assert shell.inertia[0, 0] == pytest.approx(2 / 3 * 5 * 0.01, rel=1e-5)


def test_cone_centroid_and_inertia():
    c = mp.cone(0.1, 0.4, [0, 0, 0], axis=[0, 0, 1], mass=3)
    np.testing.assert_allclose(c.cog, [0, 0, 0.1])
    assert c.inertia[2, 2] == pytest.approx(3 / 10 * 3 * 0.01)
    assert c.inertia[0, 0] == pytest.approx(3 / 20 * 3 * 0.01 + 3 / 80 * 3 * 0.16)


def test_rotated_box_matches_rotation_of_tensor():
    R = rotation_about_axis([1, 2, 3], 0.7)
    local = mp.box([0.3, 0.2, 0.1], [0, 0, 0], mass=2)
    rotated = mp.box([0.3, 0.2, 0.1], [0, 0, 0], R, mass=2)
    np.testing.assert_allclose(rotated.inertia, R @ local.inertia @ R.T)
    moments, _ = rotated.principal()
    np.testing.assert_allclose(moments, np.sort(np.diag(local.inertia)))


def test_principal_axes_recover_box_orientation():
    R = frame_from_axes(z=[1, 1, 0], x=[0, 0, 1])
    b = mp.box([0.1, 0.2, 0.3], [0, 0, 0], R, mass=1)
    _, axes = b.principal()
    # the smallest moment is about the longest edge (local z)
    assert abs(axes[:, 0] @ R[:, 2]) == pytest.approx(1)


def test_custom_inertia_about_another_point():
    rod = mp.rod([0, 0, 0], [1, 0, 0], mass=3)
    I_end = rod.inertia_about([0, 0, 0])
    back = mp.custom(3, [0.5, 0, 0], I_end, about=[0, 0, 0])
    np.testing.assert_allclose(back.inertia, rod.inertia, atol=1e-15)


def test_physical_checks():
    bad = mp.custom(1, [0, 0, 0], [1, 1, 5])
    assert any("triangle" in p for p in bad.check_physical())
    assert mp.MassProperties(-1.0).check_physical()
    with pytest.raises(InputError, match="exactly one of mass or density"):
        mp.box([1, 1, 1], [0, 0, 0])
    with pytest.raises(InputError, match="exactly one of mass or density"):
        mp.box([1, 1, 1], [0, 0, 0], mass=1, density=1)


vec = st.lists(st.floats(-5, 5, allow_nan=False), min_size=3, max_size=3)


@settings(max_examples=60, deadline=None)
@given(
    parts=st.lists(
        st.tuples(st.floats(0.1, 10), vec, st.lists(st.floats(0.01, 2), min_size=3, max_size=3)),
        min_size=1,
        max_size=5,
    ),
    point=vec,
)
def test_parallel_axis_consistency(parts, point):
    """Inertia of a composite about any point equals the sum of the parts' inertias
    about that point (the defining property of the parallel-axis theorem)."""
    bodies = [mp.box(size, cog, mass=m) for m, cog, size in parts]
    total = mp.combine(bodies)
    direct = sum(b.inertia_about(point) for b in bodies)
    np.testing.assert_allclose(total.inertia_about(point), direct, rtol=1e-9, atol=1e-9)
    assert not total.check_physical()
