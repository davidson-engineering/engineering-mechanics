import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from mechanics.mechanics import Body, Bodies, skew_sym

GRAVITY = np.array([0, 0, -9.81])


@pytest.fixture
def body():
    """Fixture for a sample Body instance."""
    return Body(id="test_body", mass=2.0, cog=[1, 1, 1], inertia=[1, 1, 1])


@pytest.fixture
def rotated_body():
    """Fixture for a rotated Body instance."""
    rotation = R.from_euler("z", 90, degrees=True).as_rotvec()
    return Body(
        id="rotated_body", mass=2.0, cog=[1, 1, 1], inertia=[1, 1, 1], rot=rotation
    )


@pytest.fixture
def bodies_collection(body, rotated_body):
    """Fixture for a collection of bodies."""
    return Bodies(bodies=[body, rotated_body])


def test_body_initialization(body):
    """Test the initialization of a Body instance."""
    assert body.id == "test_body"
    assert body.mass == 2.0
    assert np.allclose(body.cog, [1, 1, 1])
    assert np.allclose(body.inertia, np.diag([1, 1, 1]))


def test_body_inertia_transformation(rotated_body):
    """Test inertia transformation with rotation."""
    expected_inertia = (
        rotated_body.inertia
    )  # Rotated inertia should be updated in __init__
    assert np.allclose(rotated_body.inertia, expected_inertia)


def test_body_get_inertia_relative_to_point(body):
    """Test inertia tensor calculation relative to a different point."""
    ref_point = np.array([2, 2, 2])
    inertia_ref = body.get_inertia(ref=ref_point)
    expected_inertia = np.diag([1, 1, 1]) - body.mass * np.linalg.matrix_power(
        skew_sym(ref_point - body.cog), 2
    )
    assert np.allclose(inertia_ref, expected_inertia)


def test_body_principal_inertia(body):
    """Test principal moments of inertia calculation."""
    principal_inertia = body.get_prin_inertia()
    assert np.allclose(principal_inertia, np.diag([1, 1, 1]))


def test_body_torque_calculation(body):
    """Test torque calculation given angular velocity and acceleration."""
    accel = np.array([0, 0, 2])
    vel = np.array([0, 1, 0])
    torque = body.get_torque(vel=vel, accel=accel)
    expected_torque = body.get_inertia() @ accel + np.cross(
        vel, body.get_inertia() @ accel
    )
    assert np.allclose(torque, expected_torque)


def test_body_weight_calculation(body):
    """Test weight calculation based on gravity."""
    weight = body._get_weight()
    expected_weight = body.mass * GRAVITY
    assert np.allclose(weight.magnitude, expected_weight)
    assert np.allclose(weight.location, body.cog)


def test_body_scalar_multiplication(body):
    """Test scalar multiplication of a body."""
    scaled_body = body * 3
    assert scaled_body.mass == body.mass * 3
    assert np.allclose(scaled_body.cog, body.cog)
    assert np.allclose(scaled_body.inertia, body.inertia)


def test_body_addition_to_bodies(body):
    """Test adding a body to a Bodies collection using the + operator."""
    bodies = Bodies(bodies=[body])
    new_body = Body(id="new_body", mass=1.5, cog=[2, 2, 2], inertia=[0.5, 0.5, 0.5])
    bodies + new_body
    assert len(bodies.bodies) == 2
    assert bodies.bodies[1].id == "new_body"


def test_bodies_initialization(bodies_collection):
    """Test the initialization of a Bodies collection."""
    assert len(bodies_collection.bodies) == 2
    assert bodies_collection.mass == sum(body.mass for body in bodies_collection.bodies)


def test_bodies_update_properties(bodies_collection):
    """Test updating mass, center of gravity, and inertia for a Bodies collection."""
    bodies_collection.update_bodies()
    expected_mass = sum(body.mass for body in bodies_collection.bodies)
    expected_cog = (
        np.sum([body.mass * body.cog for body in bodies_collection.bodies], axis=0)
        / expected_mass
    )
    expected_inertia = np.sum(
        [body.get_inertia(ref=expected_cog) for body in bodies_collection.bodies],
        axis=0,
    )
    assert bodies_collection.mass == expected_mass
    assert np.allclose(bodies_collection.cog, expected_cog)
    assert np.allclose(bodies_collection.inertia, expected_inertia)


def test_bodies_add_and_remove_body(bodies_collection):
    """Test adding and removing a body from a Bodies collection."""
    new_body = Body(id="new_body", mass=1.5, cog=[2, 2, 2], inertia=[0.5, 0.5, 0.5])
    bodies_collection.add_body(new_body)
    assert len(bodies_collection.bodies) == 3
    assert bodies_collection.bodies[-1].id == "new_body"

    bodies_collection.remove_body(new_body)
    assert len(bodies_collection.bodies) == 2
    assert new_body not in bodies_collection.bodies
